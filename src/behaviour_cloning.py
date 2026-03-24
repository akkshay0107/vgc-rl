from pathlib import Path

import torch
from torch.utils.data import DataLoader, Dataset

from policy import PolicyNet


# Loads all files from replays immediately
# TODO: replace with lazy loading once we have larger replay datasets
class ReplayDataset(Dataset):
    def __init__(self, replays_dir: str):
        self.samples = []
        path = Path(replays_dir)

        # Load and flatten immediately
        for replay_file in sorted(path.rglob("*.replay")):
            try:
                episode_data = torch.load(replay_file, weights_only=False)
                # episode_data is a list of dicts (check terminal_player)
                self.samples.extend(episode_data)
            except Exception as e:
                print(f"Could not load file {replay_file}: {e}")

        print(f"Loaded {len(self.samples)} samples from {path}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]


def train_behavior_cloning(
    dataset,
    batch_size: int = 64,
    num_epochs: int = 10,
    learning_rate: float = 3e-4,
    val_split_ratio: float = 0.2,
    policy: PolicyNet | None = None,
) -> PolicyNet | None:
    if len(dataset) == 0:
        print("No data available for training.")
        return None

    val_size = int(val_split_ratio * len(dataset))
    train_size = len(dataset) - val_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)

    if policy is None:
        policy = PolicyNet()

    device = policy.device
    optimizer = torch.optim.AdamW(policy.parameters(), lr=learning_rate, eps=1e-5)

    for epoch in range(num_epochs):
        print(f"Epoch {epoch + 1}/{num_epochs}")
        policy.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0

        for batch in train_loader:
            obs = batch["obs"].to(device)
            mask = batch["mask"].to(device)
            target = batch["action"].to(device)

            optimizer.zero_grad()

            log_prob, _, _ = policy.evaluate_actions(obs, target, mask)
            loss = -log_prob.mean()
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

            with torch.no_grad():
                logits = policy.get_policy_masked_logits(obs, target, mask)
                _, preds = torch.max(logits, dim=2)
                train_total += target.shape[0] * 2  # two pokemon actions
                train_correct += (preds == target).sum().item()

        if train_total > 0:
            print(
                f"Training accuracy: {train_correct / train_total:.4f} | Loss: {train_loss / len(train_loader):.4f}"
            )
        else:
            print("Training set is empty.")

        policy.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0

        with torch.no_grad():
            for batch in val_loader:
                obs = batch["obs"].to(device)
                mask = batch["mask"].to(device)
                target = batch["action"].to(device)

                log_prob, _, _ = policy.evaluate_actions(obs, target, mask)
                loss = -log_prob.mean()
                val_loss += loss.item()

                logits = policy.get_policy_masked_logits(obs, target, mask)
                _, preds = torch.max(logits, dim=2)
                val_total += target.shape[0] * 2  # two pokemon actions
                val_correct += (preds == target).sum().item()

        if val_total > 0:
            print(
                f"Validation accuracy: {val_correct / val_total:.4f} | Loss: {val_loss / len(val_loader):.4f}"
            )
        else:
            print("Validation set is empty.")
        print("")

    return policy


def save_checkpoint(path, model, epoch):
    checkpoint = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
    }
    torch.save(checkpoint, path)
