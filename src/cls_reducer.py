import torch
import torch.nn as nn
import torch.nn.init as init


class CLSReducer(nn.Module):
    CLS, FIELD, ALLY, FOE = 0, 1, 2, 3
    TEXT_A, TEXT_B, NUM = 0, 1, 2

    def __init__(
        self,
        seq_len: int,
        feat_dim: int,
        d_model: int = 256,
        nhead: int = 8,
        nlayer: int = 2,
        dim_feedforward: int = 1024,
        dropout: float = 0.0,
        emb_std: float = 0.02,
    ):
        super().__init__()
        if seq_len != 38:
            raise ValueError("This CLSReducer assumes seq_len=37 (field + 12*(textA,textB,num)).")

        self.seq_len = seq_len
        self.feat_dim = feat_dim
        self.d_model = d_model
        self.emb_std = emb_std

        self.in_proj = nn.Linear(feat_dim, d_model)

        self.cls = nn.Parameter(torch.zeros(1, 1, d_model))
        self.pos_emb = nn.Parameter(torch.zeros(1, seq_len + 1, d_model))

        self.type_emb = nn.Embedding(4, d_model)
        self.part_emb = nn.Embedding(3, d_model)

        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
            norm_first=True,
            activation="gelu",
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=nlayer)

        type_ids, part_ids = self._build_ids()
        self.register_buffer("type_ids", type_ids, persistent=False)
        self.register_buffer("part_ids", part_ids, persistent=False)

        self._init_weights()

    def _build_ids(self):
        assert self.seq_len == 37
        type_ids = torch.empty(self.seq_len + 1, dtype=torch.long)
        part_ids = torch.empty(self.seq_len + 1, dtype=torch.long)

        type_ids[0] = self.CLS
        part_ids[0] = self.TEXT_A

        type_ids[1] = type_ids[2] = self.FIELD
        part_ids[1] = part_ids[2] = self.TEXT_A

        part_ids_slice = torch.tensor([self.TEXT_A, self.TEXT_B, self.NUM], dtype=torch.long)

        # Ally pokemon (6 pokemon * 3 parts)
        type_ids[3:21] = self.ALLY
        part_ids[3:21] = part_ids_slice.repeat(6)

        # Foe pokemon (6 pokemon * 3 parts)
        type_ids[21:39] = self.FOE
        part_ids[21:39] = part_ids_slice.repeat(6)

        return type_ids, part_ids

    @torch.no_grad()
    def _init_weights(self):
        init.orthogonal_(self.in_proj.weight, gain=1.0)
        init.zeros_(self.in_proj.bias)

        init.normal_(self.cls, std=self.emb_std)
        init.normal_(self.pos_emb, std=self.emb_std)
        init.normal_(self.type_emb.weight, std=self.emb_std)
        init.normal_(self.part_emb.weight, std=self.emb_std)

        for p in self.encoder.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
            else:
                nn.init.zeros_(p)

    def forward(
        self, obs: torch.Tensor, key_padding_mask: torch.Tensor | None = None
    ) -> torch.Tensor:
        if obs.dim() == 2:
            obs = obs.unsqueeze(0)
        B, S, F = obs.shape
        if S != self.seq_len or F != self.feat_dim:
            raise ValueError(f"Got shape ({S}, {F}). Expected ({self.seq_len}, {self.feat_dim})")

        if key_padding_mask is not None:
            if key_padding_mask.dim() != 2 or key_padding_mask.size(0) != B:
                raise ValueError("key_padding_mask must be (B, S) or (B, S+1).")
            if key_padding_mask.size(1) == S:
                cls_pad = torch.zeros((B, 1), dtype=torch.bool, device=key_padding_mask.device)
                key_padding_mask = torch.cat([cls_pad, key_padding_mask], dim=1)
            elif key_padding_mask.size(1) != S + 1:
                raise ValueError("key_padding_mask must be (B, S) or (B, S+1).")

        x = self.in_proj(obs)
        cls = self.cls.expand(B, -1, -1)
        x = torch.cat([cls, x], dim=1)

        type_e = self.type_emb(self.type_ids).unsqueeze(0).expand(B, -1, -1)
        part_e = self.part_emb(self.part_ids).unsqueeze(0).expand(B, -1, -1)
        x = x + self.pos_emb + type_e + part_e

        x = self.encoder(x, src_key_padding_mask=key_padding_mask)
        return x[:, 0]
