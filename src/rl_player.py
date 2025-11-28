from poke_env.battle import AbstractBattle, DoubleBattle
from poke_env.player import Player, DefaultBattleOrder
from poke_env.environment import DoublesEnv

from encoder import Encoder
import torch
from policy import PolicyNet


class RLPlayer(Player):
    def __init__(self, policy: PolicyNet, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.policy = policy

    def _get_action(self, battle: AbstractBattle):
        obs = self.get_observation(battle)
        action_mask = Encoder.get_action_mask(battle)
        with torch.no_grad():
            obs_tensor = torch.as_tensor(obs, device=self.policy.device).unsqueeze(0)
            action_mask_tensor = action_mask.unsqueeze(0)
            _, _, actions, _ = self.policy.forward(obs_tensor, action_mask_tensor, sample_actions=True)
        return actions[0].cpu().numpy()

    def choose_move(self, battle: AbstractBattle):
        assert isinstance(battle, DoubleBattle)
        if battle._wait:
            return DefaultBattleOrder()
        return DoublesEnv.action_to_order(self._get_action(battle), battle)

    def get_observation(self, battle: AbstractBattle):
        assert isinstance(battle, DoubleBattle)
        return Encoder.encode_battle_state(battle)

