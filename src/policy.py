import torch
import torch.nn as nn
from torch.distributions import Categorical

from encoder import ACT_SIZE, OBS_DIM


class PolicyNet(nn.Module):
    """
    Actor-critic model

    Input:
        obs: (B, 11, 650) or (11, 650) observation from Encoder.encode_battle_state
        action_mask: (B, 2, ACT_SIZE) or (2, ACT_SIZE), 1 = legal, 0 = illegal

    Output:
        policy_logits: (B, 2, ACT_SIZE)
        policy_log_probs: (B, 2) - log probs of sampled actions
        sampled_actions: (B, 2) - sampled actions respecting mutual exclusivity
        value: (B,)
    """

    def __init__(
        self,
        obs_dim=OBS_DIM,  # (11, 650)
        act_size=ACT_SIZE,
        hidden_size: int = 256,
        num_layers: int = 3,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.seq_len, self.feat_dim = obs_dim
        self.act_size = act_size
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Flatten input: (B, 11, 650) -> (B, 11 * 650)
        input_dim = self.seq_len * self.feat_dim

        layers = []
        layers.append(nn.Linear(input_dim, hidden_size))
        layers.append(nn.LayerNorm(hidden_size))
        layers.append(nn.ReLU())
        layers.append(nn.Dropout(dropout))

        for _ in range(num_layers - 1):
            layers.append(nn.Linear(hidden_size, hidden_size))
            layers.append(nn.LayerNorm(hidden_size))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))

        self.shared_backbone = nn.Sequential(*layers)

        # outputs logits for both Pokemon positions
        self.policy_head = nn.Linear(hidden_size, 2 * act_size)

        # outputs scalar value estimate
        self.value_head = nn.Linear(hidden_size, 1)

        self.to(self.device)

    def forward(
        self,
        obs: torch.Tensor,
        action_mask: torch.Tensor | None = None,
        sample_actions: bool = True,
    ):
        """
        Forward pass with masking, action sampling, and log-prob computation for SAC.

        Args:
            obs: (B, 11, 650) or (11, 650)
            action_mask: (B, 2, ACT_SIZE) or (2, ACT_SIZE), 1 = legal, 0 = illegal
            sample_actions: If True, sample actions and return log_probs/actions

        Returns:
            policy_logits: (B, 2, ACT_SIZE)
            policy_log_probs: (B, 2) - log probs of sampled actions
            sampled_actions: (B, 2) - sampled actions respecting mutual exclusivity
            value: (B,)
        """
        if obs.dim() == 2:
            obs = obs.unsqueeze(0)

        B, S, F = obs.shape
        assert S == self.seq_len and F == self.feat_dim, (
            f"Expected {(self.seq_len, self.feat_dim)}, got {(S, F)}"
        )

        obs_flat = obs.view(B, -1).to(self.device)
        x = self.shared_backbone(obs_flat)  # (B, hidden_size)

        # Policy logits
        policy_logits = self.policy_head(x)  # (B, 2 * ACT_SIZE)
        policy_logits = policy_logits.view(B, 2, self.act_size)  # (B, 2, ACT_SIZE)

        if action_mask is None or not sample_actions:
            # Return logits only (no masking or sampling)
            value = self.value_head(x).squeeze(-1)  # (B,)
            return policy_logits, None, None, value

        # Expand mask if needed and move to device
        if action_mask.dim() == 2:
            action_mask = action_mask.unsqueeze(0).expand(B, -1, -1)
        action_mask = action_mask.to(self.device)

        mask = action_mask == 0
        logits = policy_logits.masked_fill(mask, float("-inf"))

        # Sample action for mon 1
        cat1 = Categorical(logits=logits[:, 0, :])
        action1 = cat1.sample()  # Keep gradients for SAC training
        log_prob1 = cat1.log_prob(action1)

        mask2 = action_mask[:, 1, :].clone()

        # VGC mutual exclusivity adjustments
        idx = action1  # (B,)
        for b in range(B):
            if 1 <= idx[b] <= 6:
                # If switched to slot idx, mask for other mon
                mask2[b, idx[b]] = 0
            if 26 < idx[b] <= 46:  # Tera range for mon 1
                mask2[b, 27:47] = 0
            if idx[b] == 0:  # Pass
                mask2[b, 0] = 0
                if mask2[b].sum() == 0:  # no valid action forced to double pass
                    mask2[b, 0] = 1

        mask2 = mask2 == 0
        logits[:, 1, :].masked_fill_(mask2, float("-inf"))

        # Sample action for mon 2 from adjusted distribution
        cat2 = Categorical(logits=logits[:, 1, :])
        action2 = cat2.sample()
        log_prob2 = cat2.log_prob(action2)

        # Stack actions and log probs
        sampled_actions = torch.stack([action1, action2], dim=-1)  # (B, 2)
        policy_log_probs = torch.stack([log_prob1, log_prob2], dim=-1)  # (B, 2)

        # Value estimate
        value = self.value_head(x).squeeze(-1)  # (B,)

        return policy_logits, policy_log_probs, sampled_actions, value


def evaluate_actions(
    self, obs: torch.Tensor, actions: torch.Tensor, action_mask: torch.Tensor | None = None
):
    """
    Evaluate given actions for SAC loss computation (log_prob only, no sampling).
    """
    policy_logits, _, _, value = self(obs, action_mask, sample_actions=False)

    B = obs.shape[0] if obs.dim() == 3 else 1
    action1, action2 = actions[:, 0], actions[:, 1]

    # Apply masking to logits
    if action_mask is not None:
        if action_mask.dim() == 2:
            action_mask = action_mask.unsqueeze(0).expand(B, -1, -1)
        action_mask = action_mask.to(self.device)
        mask = action_mask == 0
        masked_logits = policy_logits.masked_fill(mask, float("-inf"))
    else:
        masked_logits = policy_logits

    # Mutual exclusivity masking (deterministic for given actions)
    if action_mask is not None:
        mask2 = action_mask[:, 1, :].clone()
    else:
        mask2 = torch.ones(B, self.act_size, device=self.device, dtype=torch.bool)

    for b in range(B):
        idx = action1[b]
        if 1 <= idx <= 6:
            mask2[b, idx] = 0
        if 26 < idx <= 46:
            mask2[b, 27:47] = 0
        if idx == 0:
            mask2[b, 0] = 0
            if mask2[b].sum() == 0:
                mask2[b, 0] = 1

    mask2 = mask2 == 0
    masked_logits[:, 1, :].masked_fill_(mask2, float("-inf"))

    # Compute log probs for given actions
    cat1 = Categorical(logits=masked_logits[:, 0, :])
    cat2 = Categorical(logits=masked_logits[:, 1, :])
    log_prob1 = cat1.log_prob(action1)
    log_prob2 = cat2.log_prob(action2)
    log_probs = torch.stack([log_prob1, log_prob2], dim=-1)

    return log_probs, value
