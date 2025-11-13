from poke_env.battle import AbstractBattle, DoubleBattle
from poke_env.player import Player, DefaultBattleOrder
from encoder import Encoder, BATTLE_STATE_DIMS
from gen9vgcenv import Gen9VGCEnv
import torch
from pseudo_policy import PseudoPolicy


class RLPlayer(Player):
    def __init__(self, policy: PseudoPolicy, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.policy = policy

    def _get_action(self, battle: AbstractBattle):
        obs = self.get_observation(battle)
        action_mask = Encoder.get_action_mask(battle)
        with torch.no_grad():
            obs_tensor = torch.as_tensor(obs, device=self.policy.device).unsqueeze(0)
            action_mask_tensor = action_mask.unsqueeze(0)
            action_pair_np, _, _ = self.policy.forward(obs_tensor, action_mask_tensor)
        return action_pair_np[0]

    def choose_move(self, battle: AbstractBattle):
        assert isinstance(battle, DoubleBattle)
        if battle._wait:
            return DefaultBattleOrder()
        return Gen9VGCEnv.action_to_order(self._get_action(battle), battle)

    def get_observation(self, battle: AbstractBattle):
        assert isinstance(battle, DoubleBattle)
        obs = torch.zeros(BATTLE_STATE_DIMS, dtype=torch.float32)
        Encoder.encode_battle_state(battle, obs)
        return obs

    def teampreview(self, battle: AbstractBattle) -> str:
        # defaults to random team preview
        return super().teampreview(battle)
