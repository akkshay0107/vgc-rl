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
        for replay_file in sorted(path.glob("*.replay")):
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
        sample = self.samples[idx]

        return {
            "obs": sample["obs"],
            "mask": sample["mask"].to(torch.bool),
            # converting numpy array action to a tensor
            "action": torch.tensor(sample["action"], dtype=torch.long),
        }


def train_behavior_cloning(replays_path):
    # Configs
    BATCH_SIZE = 32
    NUM_EPOCHS = 10
    LEARNING_RATE = 3e-4

    # Load dataset
    dataset = ReplayDataset(replays_path)

    if len(dataset) == 0:
        print("No data available for training.")
        return

    train_size = int(0.8 * len(dataset))
    train_dataset = torch.utils.data.Subset(dataset, range(train_size))
    val_dataset = torch.utils.data.Subset(dataset, range(train_size, len(dataset)))

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    policy = PolicyNet()
    device = policy.device
    optimizer = torch.optim.AdamW(policy.parameters(), lr=LEARNING_RATE, eps=1e-5)

    for epoch in range(NUM_EPOCHS):
        print(f"Epoch {epoch + 1}/{NUM_EPOCHS}")
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


if __name__ == "__main__":
    root_dir = Path(__file__).resolve().parent.parent
    replays_dir = root_dir / "replays"
    checkpoints_dir = root_dir / "checkpoints"
    checkpoints_dir.mkdir(exist_ok=True, parents=True)

    if not replays_dir.exists():
        print(f"Replays directory not found: {replays_dir}")
        exit(0)

    model = train_behavior_cloning(str(replays_dir))
    if model:
        save_path = checkpoints_dir / "behavior_cloning_checkpoint.pt"
        save_checkpoint(save_path, model, 10)
        print(f"Checkpoint saved to {save_path}")
