import torch
import torch.nn as nn
import torch.nn.init as init
from torch.distributions import Categorical

from cls_reducer import CLSReducer
from lookups import ACT_SIZE, OBS_DIM


# Needs all inputs to be on the same device as the model
class PolicyNet(nn.Module):
    def __init__(
        self,
        obs_dim=OBS_DIM,
        act_size=ACT_SIZE,
        d_model=256,
        nhead=8,
        nlayer=3,
        net_arch=(256, 256, 128),
        n_hg=4,
    ):
        super().__init__()
        self.seq_len, self.feat_dim = obs_dim
        self.act_size = act_size
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.reducer = CLSReducer(self.seq_len, self.feat_dim, d_model, nhead, nlayer, n_hg=n_hg)

        layers = [nn.Linear(d_model, net_arch[0]), nn.GELU()]
        for h_in, h_out in zip(net_arch[:-1], net_arch[1:]):
            layers.extend([nn.Linear(h_in, h_out), nn.GELU()])
        self.shared_backbone = nn.Sequential(*layers)

        # outputs logits for each possible action for each pokemon
        self.policy_head = nn.Linear(net_arch[-1], 2 * act_size)
        # outputs scalar value from the state V(s)
        self.value_head = nn.Linear(net_arch[-1], 1)

        self.to(self.device)
        self._init_weights()

    @torch.no_grad()
    def _init_weights(self):
        # orthogonal initialization of the network
        init.orthogonal_(self.policy_head.weight, gain=0.1)
        init.zeros_(self.policy_head.bias)

        init.orthogonal_(self.value_head.weight, gain=0.05)
        init.zeros_(self.value_head.bias)

        for module in self.shared_backbone:
            if isinstance(module, nn.Linear):
                init.orthogonal_(module.weight, gain=1.0)
                init.zeros_(module.bias)

    def forward(
        self,
        obs: torch.Tensor,
        state: tuple[torch.Tensor, torch.Tensor] | None = None,
        action_mask: torch.Tensor | None = None,
        sample_actions: bool = True,
    ) -> tuple[
        torch.Tensor,
        torch.Tensor | None,
        torch.Tensor | None,
        torch.Tensor,
        tuple[torch.Tensor, torch.Tensor],
    ]:
        # Add batch dimension if missing
        if obs.dim() == 2:
            obs = obs.unsqueeze(0)
        B, S, F = obs.shape

        # changed assertion to error for debugging
        if S != self.seq_len or F != self.feat_dim:
            raise ValueError(f"Got shape ({S}, {F}). Expected({self.seq_len}, {self.feat_dim})")

        # Detect phase from observation flag (Row 41, Index 18)
        is_tp = obs[:, 41, 18] > 0.5

        z, next_state = self.reducer(obs, state)
        x = self.shared_backbone(z)

        policy_logits = self.policy_head(x).reshape(B, 2, self.act_size)
        value = self.value_head(x).squeeze(-1)

        # Return raw logits if no masking or sampling needed
        if not sample_actions or action_mask is None:
            return policy_logits, None, None, value, next_state

        # Mask logits with -inf where actions are illegal
        logits = self._apply_masks(policy_logits, action_mask)

        cat1 = Categorical(logits=logits[:, 0])
        action1 = cat1.sample()
        log_prob1 = cat1.log_prob(action1)

        # Adjust logits for the second Pokemon
        logits = self._apply_sequential_masks(logits, action1, action_mask, is_tp)
        cat2 = Categorical(logits=logits[:, 1])
        action2 = cat2.sample()
        log_prob2 = cat2.log_prob(action2)

        return (
            logits,
            log_prob1 + log_prob2,  # log prob of choosing this action pair
            torch.stack([action1, action2], -1),
            value,
            next_state,
        )

    def get_policy_masked_logits(
        self,
        obs: torch.Tensor,
        action_taken: torch.Tensor,
        action_mask: torch.Tensor | None,
        state: tuple[torch.Tensor, torch.Tensor] | None = None,
    ):
        if obs.dim() == 2:
            obs = obs.unsqueeze(0)
        is_tp = obs[:, 41, 18] > 0.5

        policy_logits, _, _, _, _ = self(obs, state, action_mask, sample_actions=False)

        if action_mask is None:
            return policy_logits

        logits = self._apply_masks(policy_logits, action_mask)  # (B, 2, A)
        logits = self._apply_sequential_masks(logits, action_taken[:, 0], action_mask, is_tp)
        return logits

    def evaluate_actions(
        self,
        obs: torch.Tensor,
        actions: torch.Tensor,
        action_mask: torch.Tensor | None = None,
        state: tuple[torch.Tensor, torch.Tensor] | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        if obs.dim() == 2:
            obs = obs.unsqueeze(0)
        is_tp = obs[:, 41, 18] > 0.5

        policy_logits, _, _, value, next_state = self(obs, state, action_mask, sample_actions=False)

        if action_mask is not None:
            logits = self._apply_masks(policy_logits, action_mask)
            logits = self._apply_sequential_masks(logits, actions[:, 0], action_mask, is_tp)
        else:
            logits = policy_logits

        cat1 = Categorical(logits=logits[:, 0])
        cat2 = Categorical(logits=logits[:, 1])

        log_prob1 = cat1.log_prob(actions[:, 0])
        log_prob2 = cat2.log_prob(actions[:, 1])
        log_prob = log_prob1 + log_prob2

        entropy = cat1.entropy() + cat2.entropy()

        return log_prob, entropy, value, next_state

    def _apply_masks(self, logits: torch.Tensor, action_mask: torch.Tensor) -> torch.Tensor:
        # Replace logits of illegal actions with -inf so they have zero probability
        mask = action_mask == 0
        return logits.masked_fill(mask, float("-inf"))

    def _apply_sequential_masks(
        self,
        logits: torch.Tensor,
        action1: torch.Tensor,
        action_mask: torch.Tensor,
        is_tp: torch.Tensor,
    ):
        mask2 = action_mask[:, 1].clone().bool()

        # If Pokemon 1 switches to slot idx, Pokemon 2 cannot switch to the same slot
        switch_mask = (1 <= action1) & (action1 <= 6) & (~is_tp)
        mask2[switch_mask, action1[switch_mask]] = 0

        # If Pokemon 1 uses terastallize, Pokemon 2 cannot also tera
        tera_mask = (26 < action1) & (action1 <= 46) & (~is_tp)
        mask2[tera_mask, 27:47] = 0

        # If Pokemon 1 passes, Pokemon 2 cannot pass as well unless no valid moves left
        pass_mask = (action1 == 0) & (~is_tp)
        mask2[pass_mask, 0] = 0

        # Ensure all 4 selected Pokemon are unique (no overlap between Lead and Back)
        if is_tp.any():
            tp_indices = torch.where(is_tp)[0]
            tp_actions = action1[tp_indices]

            p1_1 = tp_actions // 6 + 1
            p2_1 = tp_actions % 6 + 1

            all_a = torch.arange(36, device=logits.device)
            p1_2 = all_a // 6 + 1
            p2_2 = all_a % 6 + 1

            overlap = (
                (p1_2.unsqueeze(0) == p1_1.unsqueeze(1))
                | (p1_2.unsqueeze(0) == p2_1.unsqueeze(1))
                | (p2_2.unsqueeze(0) == p1_1.unsqueeze(1))
                | (p2_2.unsqueeze(0) == p2_1.unsqueeze(1))
            )

            mask2[tp_indices.unsqueeze(1), all_a] &= ~overlap

        # If no valid action remains, force pass action to be valid for Pokemon 2
        no_valid = mask2.sum(-1) == 0
        mask2[no_valid, 0] = 1

        return torch.stack(
            [
                logits[:, 0],  # first row actions unchanged
                logits[:, 1].masked_fill(
                    ~mask2.bool(), float("-inf")
                ),  # masked row for second pokemon
            ],
            dim=1,
        )
