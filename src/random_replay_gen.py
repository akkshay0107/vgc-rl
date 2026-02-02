import asyncio
import uuid
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
from poke_env import AccountConfiguration, LocalhostServerConfiguration
from poke_env.battle import AbstractBattle, DoubleBattle
from poke_env.player import DefaultBattleOrder, Player, SimpleHeuristicsPlayer

import observation_builder
from env import Gen9VGCEnv
from teams import RandomTeamFromPool

class RandomReplayGenerator(Player):
    def __init__(self, save_dir, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.save_dir = save_dir
        self.episode_data = []

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

    def sequential_mask(self, action_mask, action1):
        mask2 = action_mask[1]  # using view of action mask
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

        return mask2

    def logits_from_mask(self, action_mask):
        logits = torch.ones_like(action_mask, dtype=torch.float32)
        return logits.masked_fill(~action_mask.bool(), float("-inf"))

    async def choose_move(self, battle: AbstractBattle):
        assert isinstance(battle, DoubleBattle)
        if battle._wait:
            return DefaultBattleOrder()

        obs = self.get_observation(battle)
        action_mask = observation_builder.get_action_mask(battle)
        # print_valid_actions_from_mask(battle, action_mask.detach().cpu())

        logits1 = self.logits_from_mask(action_mask[0])
        cat1 = torch.distributions.Categorical(logits=logits1)
        action1 = cat1.sample()

        mask2 = self.sequential_mask(action_mask, action1)
        logits2 = self.logits_from_mask(mask2)
        cat2 = torch.distributions.Categorical(logits=logits2)
        action2 = cat2.sample()
        action_np = np.asarray([action1.item(), action2.item()], dtype=np.int64)

        self.episode_data.append({"obs": obs, "mask": action_mask, "action": action_np})

        return Gen9VGCEnv.action_to_order(action_np, battle)

    def get_observation(self, battle: AbstractBattle):
        assert isinstance(battle, DoubleBattle)
        return observation_builder.from_battle(battle)

    def teampreview(self, battle: AbstractBattle) -> str:
        return super().random_teampreview(battle)


async def main():
    teams_dir = "./teams"
    team_files = [
        path.read_text(encoding="utf-8") for path in Path(teams_dir).iterdir() if path.is_file()
    ]
    team = RandomTeamFromPool(team_files)
    fmt = "gen9vgc2025regh"

    save_dir = "./test_replays"

    rand_replay = RandomReplayGenerator(
        save_dir=save_dir,
        account_configuration=AccountConfiguration("RandomReplayGen", None),
        battle_format=fmt,
        server_configuration=LocalhostServerConfiguration,
        team=team,
        accept_open_team_sheet=True,
        max_concurrent_battles=5,
    )

    heuristic_player = SimpleHeuristicsPlayer(
        account_configuration=AccountConfiguration("HeuristicPlayer", None),
        battle_format=fmt,
        server_configuration=LocalhostServerConfiguration,
        team=team,
        accept_open_team_sheet=True,
    )

    await rand_replay.battle_against(heuristic_player, n_battles=100)
    rand_replay._save_episode_data()

    # Cleanup
    await rand_replay.ps_client.stop_listening()
    await heuristic_player.ps_client.stop_listening()


if __name__ == "__main__":
    asyncio.run(main())
