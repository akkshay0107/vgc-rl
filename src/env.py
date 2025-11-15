import torch
from pathlib import Path

from poke_env.battle import AbstractBattle, DoubleBattle
from poke_env.environment import DoublesEnv

from encoder import BATTLE_STATE_DIMS, Encoder
from teams import RandomTeamFromPool


class SimEnv(DoublesEnv):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    @classmethod
    def build_env(cls):
        teams_dir = "./teams"
        team_files = [
            path.read_text(encoding="utf-8") for path in Path(teams_dir).iterdir() if path.is_file()
        ]
        team = RandomTeamFromPool(team_files)
        return cls(
            battle_format="gen9vgc2025regh",
            accept_open_team_sheet=True,
            start_timer_on_battle_start=True,
            log_level=25,
            team=team,
        )

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
        obs = torch.zeros(BATTLE_STATE_DIMS, dtype=torch.float32)
        Encoder.encode_battle_state(battle, obs)
        return obs
