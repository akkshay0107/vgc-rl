from poke_env.battle import AbstractBattle, DoubleBattle
from poke_env.player import Player, DefaultBattleOrder
from encoder import Encoder, BATTLE_STATE_DIMS
from env import Gen9VGCEnv
from teams import RandomTeamFromPool
import torch


class RLPlayer(Player):
    def __init__(self, policy, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.policy = policy
        self.team = RandomTeamFromPool

    def choose_move(self, battle: AbstractBattle):
        assert isinstance(battle, DoubleBattle)
        if battle._wait:
            return DefaultBattleOrder()
        obs = self.get_observation(battle)
        with torch.no_grad():
            obs_tensor = torch.as_tensor(obs, device=self.policy.device).unsqueeze(0)
            action, _, _ = self.policy.forward(obs_tensor)
        action = action.cpu().numpy()[0]
        return Gen9VGCEnv.action_to_order(action, battle)

    def get_observation(self, battle: DoubleBattle):
        obs = torch.Tensor(BATTLE_STATE_DIMS)
        Encoder.encode_battle_state(battle, obs)
        return obs
