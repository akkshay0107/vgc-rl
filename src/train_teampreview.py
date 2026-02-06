"""
## Dataset format

list of dicts, each dict containing:
  "opp_ids": values in [0, max_id] 
  "bring":  [0, team_size-1]
  "lead":   containing 0-based indices (subset of bring)
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from teampreview import TeamPreviewNet


@dataclass(frozen=True)
class Batch:
    opp_ids: torch.Tensor
    bring: torch.Tensor
    lead: torch.Tensor


class TeamPreviewDataset(Dataset):
    def __init__(self, data_dir: str | Path):
        self.team_size = 6
        self.samples: list[dict[str, torch.Tensor]] = []

        data_dir = Path(data_dir)
        files = sorted(data_dir.glob("*.pt"))
        if not files:
            raise FileNotFoundError(f"No '*.pt' files found in {data_dir}")

        for fp in files:
            obj = torch.load(fp, map_location="cpu")
            self.ingest(obj, fp)

        if not self.samples:
            raise RuntimeError(f"Loaded 0 samples from {data_dir}")

    def ingest(self, obj, fp: Path) -> None:
        def norm_one(d: dict) -> dict[str, torch.Tensor]:
            opp_ids = torch.as_tensor(d["opp_ids"], dtype=torch.long).view(6)
            bring = torch.as_tensor(d["bring"], dtype=torch.long).view(-1)
            lead = torch.as_tensor(d["lead"], dtype=torch.long).view(-1)
            return {"opp_ids": opp_ids, "bring": bring, "lead": lead}

        if isinstance(obj, list):
            for d in obj:
                self.samples.append(norm_one(d))
            return

        if isinstance(obj, dict) and {"opp_ids", "bring", "lead"} <= set(obj.keys()):
            opp_ids = torch.as_tensor(obj["opp_ids"], dtype=torch.long)
            bring = torch.as_tensor(obj["bring"], dtype=torch.long)
            lead = torch.as_tensor(obj["lead"], dtype=torch.long)

            if opp_ids.dim() != 2 or opp_ids.shape[1] != 6:
                raise ValueError(f"{fp}: expected opp_ids (N, 6), got {tuple(opp_ids.shape)}")
            if bring.dim() != 2:
                raise ValueError(f"{fp}: expected bring (N, K), got {tuple(bring.shape)}")
            if lead.dim() != 2:
                raise ValueError(f"{fp}: expected lead (N, K), got {tuple(lead.shape)}")
            if not (opp_ids.shape[0] == bring.shape[0] == lead.shape[0]):
                raise ValueError(f"{fp}: N mismatch between opp_ids/bring/lead")

            for i in range(int(opp_ids.shape[0])):
                self.samples.append(
                    norm_one({"opp_ids": opp_ids[i], "bring": bring[i], "lead": lead[i]})
                )
            return

        raise TypeError(
            f"{fp}: unsupported object type {type(obj)}; expected list[dict] or dict of tensors."
        )

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        return self.samples[idx]

    def batch(self, items: list[dict[str, torch.Tensor]]) -> Batch:
        opp = torch.stack([it["opp_ids"] for it in items], dim=0)  # (B, 6)

        bring = torch.zeros((len(items), self.team_size), dtype=torch.float32)
        lead = torch.zeros((len(items), self.team_size), dtype=torch.float32)

        for b, it in enumerate(items):
            bring_idx = it["bring"].tolist()
            lead_idx = it["lead"].tolist()

            for i in bring_idx:
                if 0 <= int(i) < self.team_size:
                    bring[b, int(i)] = 1.0
            for i in lead_idx:
                if 0 <= int(i) < self.team_size:
                    lead[b, int(i)] = 1.0

        return Batch(opp_ids=opp, bring= bring, lead=lead)


@torch.no_grad()
def topk_idx(logits: torch.Tensor, k: int) -> torch.Tensor:
    k = min(int(k), int(logits.shape[1]))
    return torch.topk(logits, k=k, dim=-1).indices 


@torch.no_grad()
def set_accuracy(pred_idx: torch.Tensor, true_mh: torch.Tensor) -> float:
    """
    pred_idx: (B, k) indices
    true_mh: (B, team_size) {0,1}
    Returns mean exact-match accuracy of the *set* (order ignored).
    """

    B, k = pred_idx.shape
    hit = 0
    for b in range(B):
        pred = set(int(i) for i in pred_idx[b].tolist())
        true = set(int(i) for i in torch.where(true_mh[b] > 0.5)[0].tolist())
        if pred == true:
            hit += 1
    return hit / max(1, B)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", type=str, required=True, help="Directory with *.tp.pt files")
    ap.add_argument("--out", type=str, default="teampreview_checkpoint.pt")
    ap.add_argument("--team_size", type=int, default=6)
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--batch_size", type=int, default=256)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--weight_decay", type=float, default=1e-4)
    args = ap.parse_args()


    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ds = TeamPreviewDataset(args.data_dir)
    loader = DataLoader(
        ds,
        batch_size=int(args.batch_size),
        shuffle=True,
        num_workers=0,
        collate_fn=ds.batch,
        drop_last=False,
    )

    model = TeamPreviewNet().to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=float(args.lr), weight_decay=float(args.weight_decay))

    bce = nn.BCEWithLogitsLoss()

    for epoch in range(1, int(args.epochs) + 1):
        model.train()
        total_loss = 0.0
        n_seen = 0

        for batch in loader:
            opp_ids = batch.opp_ids.to(device)
            bring_t = batch.bring.to(device)
            lead_t = batch.lead.to(device)

            bring_logits, lead_logits = model(opp_ids)

            lead_loss = bce(lead_logits, lead_t)

            bring_loss = bce(bring_logits, bring_t)
            loss = bring_loss + lead_loss

            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()

            bs = int(opp_ids.shape[0])
            total_loss += float(loss.item()) * bs
            n_seen += bs
        model.eval()
        with torch.no_grad():
            batch = next(iter(loader))
            opp_ids = batch.opp_ids.to(device)
            bring_t = batch.bring_mh.to(device)
            lead_t = batch.lead_mh.to(device)
            bring_logits, lead_logits = model(opp_ids)

            bring_pred = topk_idx(bring_logits, k=4)

            # mask lead predictions to brought team members
            masked = lead_logits.clone()
            mask = torch.ones_like(masked, dtype=torch.bool)
            for b in range(masked.shape[0]):
                mask[b, bring_pred[b]] = False
            masked[mask] = float("-inf")
            lead_pred = topk_idx(masked, k=2)

            bring_acc = set_accuracy(bring_pred, bring_t)
            lead_acc = set_accuracy(lead_pred, lead_t)

        avg_loss = total_loss / n_seen
        print(
            f"epoch {epoch:03d} | loss={avg_loss:.4f} | bring_set_acc={bring_acc:.3f} | lead_set_acc={lead_acc:.3f}"
        )

        torch.save(
            {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": opt.state_dict(),
            },
            args.out,
        )


if __name__ == "__main__":
    main()

