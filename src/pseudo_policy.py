import torch
import torch.nn as nn

# Define action space parameters (from gen9vgcenv.py)
NUM_SWITCHES = 6
NUM_MOVES = 4
NUM_TARGETS = 5
NUM_GIMMICKS = 1  # Tera
ACT_SIZE = 1 + NUM_SWITCHES + NUM_MOVES * NUM_TARGETS * (NUM_GIMMICKS + 1)


class PseudoPolicy(nn.Module):
    def __init__(self, observation_dim, action_dim, hidden_size=128):
        super().__init__()
        self.action_dim = action_dim
        self.network = nn.Sequential(
            nn.Linear(observation_dim, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 2 * action_dim),
        )
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.to(self.device)

    def forward(self, obs_tensor: torch.Tensor, action_mask: torch.Tensor):
        """
        Args:
            obs_tensor: Tensor of shape (batch_size, 2, 5, 30) - flattened before network
            action_mask: (optional) Tensor of shape (batch_size, 2, action_dim), 1 = legal, 0 = illegal

        Returns:
            actions: np.ndarray shape (batch_size, 2)
            cat1: torch.distributions.Categorical for mon 1
            cat2: torch.distributions.Categorical for mon 2
        """
        batch_size = obs_tensor.shape[0]
        obs_flat = obs_tensor.view(batch_size, -1).to(self.device)
        logits = self.network(obs_flat)  # shape: (batch_size, 2 * action_dim)
        logits = logits.view(batch_size, 2, self.action_dim)  # (B, 2, A)

        # --- Masking logic ---
        # action_mask: shape (batch_size, 2, action_dim), values: 1 (legal), 0 (illegal)
        mask = action_mask == 0
        logits = logits.masked_fill(mask.to(self.device), float("-inf"))

        # Distribution for both pokemon
        cat1 = torch.distributions.Categorical(logits=logits[:, 0, :])  # (B, A)
        # Sample action for mon 1
        action1 = cat1.sample()

        # Now mask out actions for mon 2 that depend on mon 1 selection if needed
        # Deep copy base mask for mon 2 and update based on mon 1's action
        mask2 = action_mask[:, 1, :].clone()

        # VGC mutual exclusivity adjustments
        idx = action1  # For each in batch

        for b in range(batch_size):
            if 1 <= idx[b] <= 6:
                # If switched to slot idx, mask for other mon
                mask2[b, idx[b]] = 0
            if 26 < idx[b] <= 46:  # Tera range for mon 1
                mask2[b, 27:47] = 0
        mask2 = mask2 == 0
        logits[:, 1, :].masked_fill_(mask2.to(self.device), float("-inf"))

        cat2 = torch.distributions.Categorical(logits=logits[:, 1, :])
        action2 = cat2.sample()

        actions = torch.stack([action1, action2], dim=1).cpu().numpy()
        return actions, cat1, cat2


if __name__ == "__main__":
    net = PseudoPolicy(observation_dim=300, action_dim=ACT_SIZE)
    action_mask = torch.ones((1, 2, ACT_SIZE))
    res = net.forward(torch.rand((1, 2, 5, 30)), action_mask)
    print(res)
