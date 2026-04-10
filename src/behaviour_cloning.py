import random
from pathlib import Path

import torch
from torch.utils.data import Dataset

from policy import PolicyNet


# Loads all files from replays immediately
# TODO: replace with lazy loading once we have larger replay datasets
class ReplayDataset(Dataset):
    def __init__(self, replays_dir: str):
        self.episodes = []
        path = Path(replays_dir)

        # Load episodes
        for replay_file in sorted(path.rglob("*.replay")):
            try:
                episode_data = torch.load(replay_file, weights_only=False)
                self.episodes.append(episode_data)
            except Exception as e:
                print(f"Could not load file {replay_file}: {e}")

        print(f"Loaded {len(self.episodes)} episodes from {path}")

    def __len__(self):
        return len(self.episodes)

    def __getitem__(self, idx):
        return self.episodes[idx]


def train_behavior_cloning(
    dataset,
    batch_size: int = 1,  # Process one episode at a time for simplicity
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
    train_indices = torch.randperm(len(dataset))[:train_size]
    val_indices = torch.randperm(len(dataset))[train_size:]

    train_subset = [dataset[i] for i in train_indices]
    val_subset = [dataset[i] for i in val_indices]

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

        random.shuffle(train_subset)
        for episode in train_subset:
            state = None
            optimizer.zero_grad()
            episode_loss = 0.0

            for sample in episode:
                obs = sample["obs"].to(device).unsqueeze(0)
                mask = sample["mask"].to(device).unsqueeze(0)
                target = torch.from_numpy(sample["action"]).to(device).unsqueeze(0)

                log_prob, entropy, value = policy.evaluate_actions(obs, target, mask, state=state)
                # We need to compute next_state separately
                _, _, _, _, next_state = policy(obs, state, mask, sample_actions=False)

                loss = -log_prob.mean()
                episode_loss += loss

                with torch.no_grad():
                    logits = policy.get_policy_masked_logits(obs, target, mask, state=state)
                    _, preds = torch.max(logits, dim=2)
                    train_total += target.shape[0] * 2
                    train_correct += (preds == target).sum().item()

                state = (
                    next_state[0].detach(),
                    next_state[1].detach(),
                )  # TBPTT if we don't detach

            if len(episode) > 0:
                episode_loss /= len(episode)
                episode_loss.backward()
                optimizer.step()
                train_loss += episode_loss.item()

        if train_total > 0:
            print(
                f"Training accuracy: {train_correct / train_total:.4f} | Loss: {train_loss / len(train_subset):.4f}"
            )

        policy.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0

        with torch.no_grad():
            for episode in val_subset:
                state = None
                episode_loss = 0.0
                for sample in episode:
                    obs = sample["obs"].to(device).unsqueeze(0)
                    mask = sample["mask"].to(device).unsqueeze(0)
                    target = torch.from_numpy(sample["action"]).to(device).unsqueeze(0)

                    log_prob, entropy, value = policy.evaluate_actions(
                        obs, target, mask, state=state
                    )
                    _, _, _, _, state = policy(obs, state, mask, sample_actions=False)

                    episode_loss += -log_prob.mean().item()
                    logits = policy.get_policy_masked_logits(obs, target, mask, state=state)
                    _, preds = torch.max(logits, dim=2)
                    val_total += target.shape[0] * 2
                    val_correct += (preds == target).sum().item()

                if len(episode) > 0:
                    val_loss += episode_loss / len(episode)

        if val_total > 0:
            print(
                f"Validation accuracy: {val_correct / val_total:.4f} | Loss: {val_loss / len(val_subset):.4f}"
            )
        print("")

    return policy


def save_checkpoint(path, model, epoch):
    checkpoint = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
    }
    torch.save(checkpoint, path)
