from encoder import BATTLE_STATE_DIMS, Encoder
from gen9vgcenv import Gen9VGCEnv
from poke_env.battle import AbstractBattle, DoubleBattle
import torch

class SimEnv(Gen9VGCEnv):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def calc_reward(self, battle: AbstractBattle) -> float:
        if not battle.finished:
            return 0
        elif battle.won:
            return 1
        elif battle.lost:
            return -1
        else:
            return 0

    def embed_battle(self, battle: AbstractBattle):
        assert isinstance(battle, DoubleBattle)
        obs = torch.Tensor(BATTLE_STATE_DIMS)
        Encoder.encode_battle_state(battle, obs)
        return obs
