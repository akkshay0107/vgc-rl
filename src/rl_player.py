import torch
from poke_env.battle import AbstractBattle, DoubleBattle
from poke_env.player import DefaultBattleOrder, Player
from torch.distributions import Categorical

import observation_builder
from env import Gen9VGCEnv
from policy import PolicyNet


class RLPlayer(Player):
    """
    Class that plays moves as per the trained policy net.
    """

    def __init__(
        self,
        policy: PolicyNet,
        p: float = 0.9,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.policy = policy

        assert 0.0 <= p <= 1.0
        self.p = p
        self.state = None

    def _apply_top_p(self, logits: torch.Tensor) -> torch.Tensor:
        sorted_logits, sorted_indices = torch.sort(logits, descending=True, dim=-1)
        sorted_probs = torch.softmax(sorted_logits, dim=-1)
        cumulative_probs = torch.cumsum(sorted_probs, dim=-1)

        sorted_indices_to_remove = cumulative_probs > self.p
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = False

        sorted_logits[sorted_indices_to_remove] = float("-inf")

        return sorted_logits.scatter(-1, sorted_indices, sorted_logits)

    def _top_p(self, obs, action_mask):
        policy_logits, _, _, _, self.state = self.policy(
            obs, self.state, action_mask, sample_actions=False
        )
        logits = self.policy._apply_masks(policy_logits, action_mask)

        p1_logits = self._apply_top_p(logits[:, 0])
        cat1 = Categorical(logits=p1_logits)
        action1 = cat1.sample()  # (B,)

        logits = self.policy._apply_sequential_masks(logits, action1, action_mask)
        p2_logits = self._apply_top_p(logits[:, 1])
        cat2 = Categorical(logits=p2_logits)
        action2 = cat2.sample()  # (B,)

        return torch.stack([action1, action2], dim=-1)

    def _get_action(self, battle: AbstractBattle):
        obs = self.get_observation(battle)
        action_mask = observation_builder.get_action_mask(battle)
        with torch.no_grad():
            actions = self._top_p(
                obs.unsqueeze(0).to(self.policy.device),
                action_mask.unsqueeze(0).to(self.policy.device),
            )
        return actions[0].cpu().numpy()

    def choose_move(self, battle: AbstractBattle):
        assert isinstance(battle, DoubleBattle)
        if battle._wait:
            return DefaultBattleOrder()
        return Gen9VGCEnv.action_to_order(self._get_action(battle), battle)

    def get_observation(self, battle: AbstractBattle):
        assert isinstance(battle, DoubleBattle)
        return observation_builder.from_battle(battle)

    def teampreview(self, battle: AbstractBattle) -> str:
        assert isinstance(battle, DoubleBattle)
        # Team preview is the start of the battle, so we reset the state here
        self.state = None
        action = self._get_action(battle)
        order = Gen9VGCEnv.action_to_order(action, battle)
        return order.message

    def _battle_finished_callback(self, battle: AbstractBattle):
        # Reset state at the end of the battle to prevent memory leaks or state carry-over
        self.state = None
