import torch
import torch.nn as nn
import torch.nn.init as init
from torch.distributions import Categorical

from encoder import ACT_SIZE, OBS_DIM


class PolicyNet(nn.Module):
    def __init__(self, obs_dim=OBS_DIM, act_size=ACT_SIZE, net_arch=[256, 256, 128]):
        super().__init__()
        self.seq_len, self.feat_dim = obs_dim
        self.act_size = act_size
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        input_dim = self.seq_len * self.feat_dim

        layers = [nn.Linear(input_dim, net_arch[0]), nn.ReLU()]
        for h_in, h_out in zip(net_arch[:-1], net_arch[1:]):
            layers.extend([nn.Linear(h_in, h_out), nn.ReLU()])
        self.shared_backbone = nn.Sequential(*layers)

        # outputs logits for each possible action for each pokemon
        self.policy_head = nn.Linear(net_arch[-1], 2 * act_size)
        # outputs scalar value from the state V(s)
        self.value_head = nn.Linear(net_arch[-1], 1)

        self.to(self.device)
        self._init_weights()

    def _init_weights(self):
        # orthogonal initialization of the network
        gain_hidden = init.calculate_gain("relu")
        init.orthogonal_(self.policy_head.weight, gain=0.01)
        init.zeros_(self.policy_head.bias)
        init.orthogonal_(self.value_head.weight, gain=1.0)
        init.zeros_(self.value_head.bias)
        for module in self.shared_backbone:
            if isinstance(module, nn.Linear):
                init.orthogonal_(module.weight, gain=gain_hidden)
                init.zeros_(module.bias)

    def forward(
        self,
        obs: torch.Tensor,
        action_mask: torch.Tensor | None = None,
        sample_actions: bool = True,
    ) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor | None, torch.Tensor]:
        # Add batch dimension if missing
        if obs.dim() == 2:
            obs = obs.unsqueeze(0)
        B, S, F = obs.shape

        # changed assertion to error for debugging
        if S != self.seq_len or F != self.feat_dim:
            raise ValueError(f"Got shape ({S}, {F}). Expected({self.seq_len}, {self.feat_dim})")

        # Flatten observation (B, S, F) -> (B, S*F), move to device
        x = self.shared_backbone(obs.view(B, -1).to(self.device))

        policy_logits = self.policy_head(x).view(B, 2, self.act_size)
        value = self.value_head(x).squeeze(-1)

        # Return raw logits if no masking or sampling needed
        if not sample_actions or action_mask is None:
            return policy_logits, None, None, value

        # Expand action_mask to match batch size if 2D, and move to device
        action_mask = (
            action_mask.unsqueeze(0).expand(B, -1, -1).to(self.device)
            if action_mask.dim() == 2
            else action_mask.to(self.device)
        )

        # Mask logits with -inf where actions are illegal
        logits = self._apply_masks(policy_logits.clone(), action_mask)

        # Sample actions for first Pokemon using masked logits distribution
        cat1 = Categorical(logits=logits[:, 0])
        action1 = cat1.sample()
        log_prob1 = cat1.log_prob(action1)

        # Adjust logits for the second Pokemon to enforce mutual exclusivity with action1
        logits = self._apply_sequential_masks(logits, action1, action_mask)
        cat2 = Categorical(logits=logits[:, 1])
        action2 = cat2.sample()
        log_prob2 = cat2.log_prob(action2)

        return (
            policy_logits,
            torch.stack([log_prob1, log_prob2], -1),
            torch.stack([action1, action2], -1),
            value,
        )

    def evaluate_actions(
        self, obs: torch.Tensor, actions: torch.Tensor, action_mask: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # Get logits and value prediction without sampling
        policy_logits, _, _, value = self(obs, action_mask, sample_actions=False)
        B = obs.shape[0] if obs.dim() == 3 else 1

        if action_mask is not None:
            # Expand and mask logits for illegal actions
            action_mask = (
                action_mask.unsqueeze(0).expand(B, -1, -1).to(self.device)
                if action_mask.dim() == 2
                else action_mask.to(self.device)
            )

            logits = self._apply_masks(policy_logits.clone(), action_mask)
            # Apply mutual exclusivity mask deterministically for given actions
            logits = self._apply_sequential_masks(logits, actions[:, 0], action_mask)
        else:
            logits = policy_logits

        # Calculate log probabilities of provided actions under the masked distributions
        cat1 = Categorical(logits=logits[:, 0])
        cat2 = Categorical(logits=logits[:, 1])
        log_prob1 = cat1.log_prob(actions[:, 0])
        log_prob2 = cat2.log_prob(actions[:, 1])

        return torch.stack([log_prob1, log_prob2], -1), value

    def _apply_masks(self, logits: torch.Tensor, action_mask: torch.Tensor) -> torch.Tensor:
        # Replace logits of illegal actions with -inf so they have zero probability
        mask = action_mask == 0
        return logits.masked_fill(mask, float("-inf"))

    def _apply_sequential_masks(
        self, logits: torch.Tensor, action1: torch.Tensor, action_mask: torch.Tensor
    ) -> torch.Tensor:
        mask2 = action_mask[:, 1].clone()

        # Create boolean masks for each action types of Pokemon 1
        switch_mask = (1 <= action1) & (action1 <= 6)
        tera_mask = (26 < action1) & (action1 <= 46)
        pass_mask = action1 == 0

        # If Pokemon 1 switches to slot idx, Pokemon 2 cannot switch to the same slot
        mask2[switch_mask, action1[switch_mask]] = 0

        # If Pokemon 1 uses terastallize in certain moves, Pokemon 2 cannot also tera in that range
        mask2[tera_mask, 27:47] = 0

        # If Pokemon 1 passes, Pokemon 2 cannot pass as well unless no valid moves left
        mask2[pass_mask, 0] = 0

        # If no valid action remains, force pass action to be valid for Pokemon 2
        no_valid = mask2.sum(-1) == 0
        mask2[no_valid, 0] = 1

        logits[:, 1].masked_fill_(mask2 == 0, float("-inf"))
        return logits
