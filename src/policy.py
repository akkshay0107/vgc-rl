import torch
import torch.nn as nn

from encoder import ACT_SIZE, OBS_DIM


class PolicyNet(nn.Module):
    """
    Actor-critic model (pretty simple, not optimized)

    Input:
        obs: (B, 11, 650) or (11, 650) observation from Encoder.encode_battle_state

    Output:
        policy_logits: (B, 2, ACT_SIZE)
        value:         (B,)
    """

    def __init__(
        self,
        obs_dim=OBS_DIM,  # (11, 650)
        act_size=ACT_SIZE,
        d_model: int = 256,
        n_heads: int = 4,
        n_layers: int = 2,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.seq_len, self.feat_dim = obs_dim
        self.act_size = act_size

        self.input_proj = nn.Linear(self.feat_dim, d_model)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=4 * d_model,
            dropout=dropout,
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)

        self.norm = nn.LayerNorm(d_model)

        # Policy head
        self.policy_head = nn.Linear(d_model, 2 * act_size)

        # Value head
        self.value_head = nn.Linear(d_model, 1)

    def forward(self, obs: torch.Tensor, action_mask: torch.Tensor | None = None):
        if obs.dim() == 2:
            obs = obs.unsqueeze(0)

        B, S, F = obs.shape
        assert S == self.seq_len and F == self.feat_dim, (
            f"Expected {(self.seq_len, self.feat_dim)}, got {(S, F)}"
        )

        x = self.input_proj(obs)

        # Encode sequence
        x = self.encoder(x)

        # Pool across 11 tokens
        x_pooled = x.mean(dim=1)
        x_pooled = self.norm(x_pooled)

        # Policy logits
        policy_logits = self.policy_head(x_pooled)
        policy_logits = policy_logits.view(B, 2, self.act_size)

        if action_mask is not None:
            if action_mask.dim() == 2:
                # (2, ACT_SIZE) -> (B, 2, ACT_SIZE) by broadcasting
                action_mask = action_mask.unsqueeze(0).expand(B, -1, -1)
            # Used a large negative number to zero-out probabilities
            policy_logits = policy_logits.masked_fill(action_mask == 0, -1e9)

        value = self.value_head(x_pooled).squeeze(-1)  # (B,)

        return policy_logits, value
