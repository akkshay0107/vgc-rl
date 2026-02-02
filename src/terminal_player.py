import asyncio
import random
import uuid
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
from poke_env import AccountConfiguration, LocalhostServerConfiguration
from poke_env.battle import AbstractBattle, DoubleBattle
from poke_env.player import (
    DefaultBattleOrder,
    MaxBasePowerPlayer,
    Player,
    RandomPlayer,
    SimpleHeuristicsPlayer,
)
from poke_env.teambuilder import Teambuilder

import observation_builder
from env import Gen9VGCEnv
from observation_builder import ACT_SIZE
from teams import RandomTeamFromPool

_TARGET_NAME = {-2: "default", -1: "self", 0: "ally", 1: "opp1", 2: "opp2"}

# TODO: improve this to give all the important information
def print_valid_actions_from_mask(battle, action_mask):
    m = action_mask
    if torch.is_tensor(m):
        m = m.detach().cpu().numpy()
    m = np.asarray(m)
    if m.ndim == 1:
        m = m.reshape(2, -1)

    team_list = list(battle.team.values())

    for pos in range(2):
        active = battle.active_pokemon[pos]
        if active is None:
            print("== EMPTY SLOT ==")
            for i in range(6):
                a = 1 + i
                if a < m.shape[1] and m[pos, a]:
                    mon = team_list[i]
                    print(f"{a}: SWITCH -> {mon.species}")
            continue

        mvs = (
            battle.available_moves[pos]
            if len(battle.available_moves[pos]) == 1
            and battle.available_moves[pos][0].id in ["struggle", "recharge"]
            else list(active.moves.values())
        )

        print(f"\nPos {pos}: {active.species}")

        if m[pos, 0]:
            print("0: PASS")

        for i in range(6):
            a = 1 + i
            if a < m.shape[1] and m[pos, a]:
                mon = team_list[i]
                print(f"{a}: SWITCH -> {mon.species}")

        for a in range(7, ACT_SIZE):
            if not m[pos, a]:
                continue
            gimmick = (a - 7) // 20
            move_i = ((a - 7) % 20) // 5
            target = ((a - 7) % 5) - 2
            if move_i >= len(mvs):
                continue
            mv = mvs[move_i]
            tera = gimmick == 1
            print(f"{a}: MOVE -> {mv.id} | target={_TARGET_NAME[target]} | tera={tera}")


# this function only exists to give each player a seperate name
# which is random before the battle with the bot (although unnecessarily expensive)
# this way I shouldn't be able to tell which opponent is of which
# type before the game starts
def get_opponent(num: float, fmt: str, team: Teambuilder) -> Player:
    name = uuid.uuid4().hex[:7]
    if num < 0.7:
        return SimpleHeuristicsPlayer(
            account_configuration=AccountConfiguration(name, None),
            battle_format=fmt,
            server_configuration=LocalhostServerConfiguration,
            team=team,
            accept_open_team_sheet=True,
        )
    elif num < 0.9:
        return MaxBasePowerPlayer(
            account_configuration=AccountConfiguration(name, None),
            battle_format=fmt,
            server_configuration=LocalhostServerConfiguration,
            team=team,
            accept_open_team_sheet=True,
        )
    else:
        return RandomPlayer(
            account_configuration=AccountConfiguration(name, None),
            battle_format=fmt,
            server_configuration=LocalhostServerConfiguration,
            team=team,
            accept_open_team_sheet=True,
        )


class TerminalPlayer(Player):
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

    async def choose_move(self, battle: AbstractBattle):
        assert isinstance(battle, DoubleBattle)
        if battle._wait:
            return DefaultBattleOrder()

        obs = self.get_observation(battle)
        action_mask = observation_builder.get_action_mask(battle).detach().cpu()
        print_valid_actions_from_mask(battle, action_mask)

        line = await asyncio.to_thread(input, "")
        action = [int(x) for x in line.split()][:2]
        action_np = np.asarray(action, dtype=np.int64)

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

    # As of right now data is collected entirely as human battles vs bots
    # Potentially add cases where bots play each other and human takes over midway
    # (for diverse initial states, instead of always being on winning positions) ??
    # Have to remove the bot on bot moves from epsiode data for that
    choice = "y"
    while choice == "y":
        num = random.random()
        selected_opp = get_opponent(num, fmt, team)

        await term_player.battle_against(selected_opp, n_battles=1)
        term_player._save_episode_data()

        # Cleanup newly created opponent
        await selected_opp.ps_client.stop_listening()

        choice = input("Play again? (y/n)")[0]
        choice = choice.lower()

    # Cleanup
    await term_player.ps_client.stop_listening()


if __name__ == "__main__":
    asyncio.run(main())
