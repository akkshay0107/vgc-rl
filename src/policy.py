import torch
import torch.nn as nn
from torch.distributions import Categorical

from encoder import ACT_SIZE, OBS_DIM


class PolicyNet(nn.Module):
    """
    Actor-critic model (policy-evaluation network)
    
    Input:
        obs: (B, 11, 650) or (11, 650) observation from Encoder.encode_battle_state
        action_mask: (B, 2, ACT_SIZE) or (2, ACT_SIZE), 1 = legal, 0 = illegal
        
    Output:
        policy_logits: (B, 2, ACT_SIZE)
        value:         (B,)
        
    The (11, 650) observation structure:
        - Row 0: Field conditions (624 TinyBERT + 26 padding)
        - Rows 1-4: Up to 4 ally Pokemon (624 TinyBERT + 26 numerical features)
        - Rows 5-10: Up to 6 opponent Pokemon (624 TinyBERT + 26 numerical features)
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
        input_dim = self.seq_len * self.feat_dim  # 11 * 650 = 7150
        
        # Build feed-forward network
        layers = []
        layers.append(nn.Linear(input_dim, hidden_size))
        layers.append(nn.ReLU())
        layers.append(nn.Dropout(dropout))
        
        for _ in range(num_layers - 1):
            layers.append(nn.Linear(hidden_size, hidden_size))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
        
        self.shared_backbone = nn.Sequential(*layers)
        
        # Policy head: outputs logits for both Pokemon positions
        self.policy_head = nn.Linear(hidden_size, 2 * act_size)
        
        # Value head: outputs scalar value estimate
        self.value_head = nn.Linear(hidden_size, 1)
        
        self.to(self.device)

    def forward(self, obs: torch.Tensor, action_mask: torch.Tensor | None = None):
        """
        Forward pass with masking logic from pseudo_policy.py
        
        Args:
            obs: (B, 11, 650) or (11, 650)
            action_mask: (B, 2, ACT_SIZE) or (2, ACT_SIZE), 1 = legal, 0 = illegal
            
        Returns:
            policy_logits: (B, 2, ACT_SIZE)
            value: (B,)
        """
        if obs.dim() == 2:
            obs = obs.unsqueeze(0)
        
        B, S, F = obs.shape
        assert S == self.seq_len and F == self.feat_dim, (
            f"Expected {(self.seq_len, self.feat_dim)}, got {(S, F)}"
        )
        
        # Flatten observation: (B, 11, 650) -> (B, 7150)
        obs_flat = obs.view(B, -1).to(self.device)
        
        # Pass through shared backbone
        x = self.shared_backbone(obs_flat)  # (B, hidden_size)
        
        # Policy logits
        policy_logits = self.policy_head(x)  # (B, 2 * ACT_SIZE)
        policy_logits = policy_logits.view(B, 2, self.act_size)  # (B, 2, ACT_SIZE)
        
        # Apply masking logic from pseudo_policy.py
        if action_mask is not None:
            if action_mask.dim() == 2:
                # (2, ACT_SIZE) -> (B, 2, ACT_SIZE) by broadcasting
                action_mask = action_mask.unsqueeze(0).expand(B, -1, -1)
            action_mask = action_mask.to(self.device)
            
            # --- Masking logic from pseudo_policy.py (without changes) ---
            # action_mask: shape (batch_size, 2, action_dim), values: 1 (legal), 0 (illegal)
            mask = action_mask == 0
            logits = policy_logits.masked_fill(mask, float("-inf"))
            
            # Sample action for mon 1 (detached, only for mask adjustment)
            cat1 = Categorical(logits=logits[:, 0, :])  # (B, ACT_SIZE)
            action1 = cat1.sample().detach()  # Detached so it doesn't affect gradients
            
            # Deep copy base mask for mon 2 and update based on mon 1's action
            mask2 = action_mask[:, 1, :].clone()
            
            # VGC mutual exclusivity adjustments
            idx = action1  # For each in batch
            
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
            
            # Update policy_logits with masked values
            policy_logits = logits
        
        # Value estimate
        value = self.value_head(x).squeeze(-1)  # (B,)
        
        return policy_logits, value
