import random
from pathlib import Path

import torch
from torch.distributions import Categorical
from torch.utils.data import Dataset

from policy import PolicyNet


def unwrap_policy(policy: PolicyNet) -> PolicyNet:
    return getattr(policy, "_orig_mod", policy)


class ReplayDataset(Dataset):
    def __init__(self, replays_dir: str):
        self.episodes = []
        path = Path(replays_dir)

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


def _to_batched_obs(obs: torch.Tensor, device: torch.device) -> torch.Tensor:
    obs = obs.to(device)
    if obs.dim() == 2:
        obs = obs.unsqueeze(0)
    return obs


def _to_batched_mask(mask: torch.Tensor, device: torch.device) -> torch.Tensor:
    mask = mask.to(device)
    if mask.dim() == 2:
        mask = mask.unsqueeze(0)
    return mask


def _to_batched_action(action, device: torch.device) -> torch.Tensor:
    action = torch.as_tensor(action, dtype=torch.long, device=device)
    if action.dim() == 1:
        action = action.unsqueeze(0)
    return action


def _behavior_cloning_step(
    policy: PolicyNet,
    obs: torch.Tensor,
    mask: torch.Tensor,
    target: torch.Tensor,
    state,
):
    policy_logits, _, _, _, next_state = policy(obs, state, mask, sample_actions=False)

    logits = policy._apply_masks(policy_logits, mask)
    logits = policy._apply_sequential_masks(logits, target[:, 0], mask)

    cat1 = Categorical(logits=logits[:, 0])
    cat2 = Categorical(logits=logits[:, 1])

    log_prob = cat1.log_prob(target[:, 0]) + cat2.log_prob(target[:, 1])
    loss = -log_prob.mean()

    preds = torch.stack(
        [
            logits[:, 0].argmax(dim=-1),
            logits[:, 1].argmax(dim=-1),
        ],
        dim=-1,
    )

    return loss, preds, next_state


def train_behavior_cloning(
    dataset,
    batch_size: int = 1,
    num_epochs: int = 10,
    learning_rate: float = 3e-4,
    val_split_ratio: float = 0.2,
    policy: PolicyNet | None = None,
) -> PolicyNet | None:
    if len(dataset) == 0:
        print("No data available for training.")
        return None

    n = len(dataset)
    if n == 1:
        train_indices = [0]
        val_indices = []
    else:
        val_size = int(round(val_split_ratio * n))
        val_size = max(1, val_size) if val_split_ratio > 0 else 0
        val_size = min(val_size, n - 1)

        perm = torch.randperm(n).tolist()
        val_indices = perm[:val_size]
        train_indices = perm[val_size:]

    train_subset = [dataset[i] for i in train_indices]
    val_subset = [dataset[i] for i in val_indices]

    if policy is None:
        policy = PolicyNet()

    device = policy.device
    optimizer = torch.optim.AdamW(policy.parameters(), lr=learning_rate, eps=1e-5)

    for epoch in range(num_epochs):
        print(f"Epoch {epoch + 1}/{num_epochs}")

        policy.train()
        random.shuffle(train_subset)

        train_loss_sum = 0.0
        train_correct = 0
        train_total = 0
        train_episodes_used = 0
        episodes_since_step = 0

        optimizer.zero_grad(set_to_none=True)

        for episode in train_subset:
            if len(episode) == 0:
                continue

            state = None
            episode_loss = 0.0

            for sample in episode:
                obs = _to_batched_obs(sample["obs"], device)
                mask = _to_batched_mask(sample["mask"], device)
                target = _to_batched_action(sample["action"], device)

                loss, preds, next_state = _behavior_cloning_step(
                    policy,
                    obs,
                    mask,
                    target,
                    state,
                )

                episode_loss = episode_loss + loss
                train_total += target.numel()
                train_correct += (preds == target).sum().item()

                state = (
                    next_state[0].detach(),
                    next_state[1].detach(),
                )

            episode_loss = episode_loss / len(episode)
            (episode_loss / batch_size).backward()

            train_loss_sum += episode_loss.item()
            train_episodes_used += 1
            episodes_since_step += 1

            if episodes_since_step >= batch_size:
                torch.nn.utils.clip_grad_norm_(policy.parameters(), 1.0)
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                episodes_since_step = 0

        if episodes_since_step > 0:
            torch.nn.utils.clip_grad_norm_(policy.parameters(), 1.0)
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)

        if train_total > 0 and train_episodes_used > 0:
            print(
                f"Training accuracy: {train_correct / train_total:.4f} | "
                f"Loss: {train_loss_sum / train_episodes_used:.4f}"
            )

        policy.eval()
        val_loss_sum = 0.0
        val_correct = 0
        val_total = 0
        val_episodes_used = 0

        with torch.inference_mode():
            for episode in val_subset:
                if len(episode) == 0:
                    continue

                state = None
                episode_loss = 0.0

                for sample in episode:
                    obs = _to_batched_obs(sample["obs"], device)
                    mask = _to_batched_mask(sample["mask"], device)
                    target = _to_batched_action(sample["action"], device)

                    loss, preds, next_state = _behavior_cloning_step(
                        policy,
                        obs,
                        mask,
                        target,
                        state,
                    )

                    episode_loss += loss.item()
                    val_total += target.numel()
                    val_correct += (preds == target).sum().item()

                    state = next_state

                val_loss_sum += episode_loss / len(episode)
                val_episodes_used += 1

        if val_total > 0 and val_episodes_used > 0:
            print(
                f"Validation accuracy: {val_correct / val_total:.4f} | "
                f"Loss: {val_loss_sum / val_episodes_used:.4f}"
            )
        elif len(val_subset) == 0:
            print("Validation skipped: no validation split.")

        print("")

    return policy


def save_checkpoint(path, model, epoch):
    model = unwrap_policy(model)
    checkpoint = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
    }
    torch.save(checkpoint, path)
