import asyncio
import uuid
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
from poke_env import AccountConfiguration, LocalhostServerConfiguration
from poke_env.battle import AbstractBattle, DoubleBattle
from poke_env.player import DefaultBattleOrder, Player, SimpleHeuristicsPlayer

from encoder import Encoder
from env import Gen9VGCEnv
from teams import RandomTeamFromPool


class TerminalPlayer(Player):
    def __init__(self, save_dir, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.save_dir = save_dir
        self.episode_data = []

    def _get_action(self, battle: AbstractBattle):
        obs = self.get_observation(battle).detach().cpu()
        action_mask = Encoder.get_action_mask(battle).detach().cpu()
        action = [int(x) for x in input().split()][:2]
        action_np = np.array(action)

        self.episode_data.append(
            {
                "obs": obs,
                "mask": action_mask,
                "action": action_np,
            }
        )

        return action_np

    def _save_episode_data(self):
        save_dir = Path(self.save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)

        uid = f"{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')}_{uuid.uuid4().hex[:10]}"
        save_path = save_dir / f"{uid}.replay"
        tmp_path = save_dir / f"{uid}.replay.tmp"

        torch.save(self.episode_data, tmp_path)
        tmp_path.replace(save_path)

        self.episode_data = []
        return str(save_path)

    def choose_move(self, battle: AbstractBattle):
        assert isinstance(battle, DoubleBattle)
        if battle._wait:
            return DefaultBattleOrder()
        return Gen9VGCEnv.action_to_order(self._get_action(battle), battle)

    def get_observation(self, battle: AbstractBattle):
        assert isinstance(battle, DoubleBattle)
        return Encoder.encode_battle_state(battle)

    def teampreview(self, battle: AbstractBattle) -> str:
        return super().random_teampreview(battle)


async def main():
    teams_dir = "./teams"
    team_files = [
        path.read_text(encoding="utf-8") for path in Path(teams_dir).iterdir() if path.is_file()
    ]
    team = RandomTeamFromPool(team_files)
    fmt = "gen9vgc2025regh"

    save_dir = "./replays"

    term_player = TerminalPlayer(
        save_dir=save_dir,
        account_configuration=AccountConfiguration("TermPlayer", None),
        battle_format=fmt,
        server_configuration=LocalhostServerConfiguration,
        team=team,
        accept_open_team_sheet=True,
        max_concurrent_battles=1,
    )

    heuristic_player = SimpleHeuristicsPlayer(
        account_configuration=AccountConfiguration("HeuristicPlayer", None),
        battle_format=fmt,
        server_configuration=LocalhostServerConfiguration,
        team=team,
        accept_open_team_sheet=True,
    )

    await term_player.battle_against(heuristic_player, n_battles=1)
    term_player._save_episode_data()

    # Cleanup
    await term_player.ps_client.stop_listening()
    await heuristic_player.ps_client.stop_listening()


if __name__ == "__main__":
    asyncio.run(main())
