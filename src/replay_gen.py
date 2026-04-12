import argparse
import asyncio
import random
import uuid
from abc import ABC, abstractmethod
from datetime import datetime, timezone
from pathlib import Path
from typing import Awaitable

import numpy as np
import torch
from poke_env import AccountConfiguration, LocalhostServerConfiguration
from poke_env.battle import AbstractBattle, DoubleBattle
from poke_env.player import (
    BattleOrder,
    DefaultBattleOrder,
    MaxBasePowerPlayer,
    Player,
    RandomPlayer,
    SimpleHeuristicsPlayer,
)
from poke_env.teambuilder import Teambuilder

import observation_builder
from env import Gen9VGCEnv
from heuristic import FuzzyHeuristic
from lookups import ACT_SIZE
from teams import RandomTeamFromPool

_TARGET_NAME = {-2: "pkm2", -1: "pkm1", 0: "empty", 1: "opp1", 2: "opp2"}


def print_teampreview_actions(battle):
    team = list(battle.team.values())
    print("\n=== TEAM PREVIEW ===")
    for i, mon in enumerate(team):
        print(f"{i + 1}: {mon.species}")

    print("\nActions (0-35) representing combinations {Mon1, Mon2}:")
    valid_count = 0
    for a in range(36):
        p1 = a // 6 + 1
        p2 = a % 6 + 1
        if p1 < p2 and p1 <= len(team) and p2 <= len(team):
            name1 = team[p1 - 1].species
            name2 = team[p2 - 1].species
            print(
                f"{a:2}:{{{p1},{p2}}} {name1[:7]:7}&{name2[:7]:7}",
                end=" | " if (valid_count + 1) % 3 != 0 else "\n",
            )
            valid_count += 1
    print("\n(Note: Permutations like {2,1} are automatically mapped to {1,2})")


def print_valid_actions_from_mask(battle, action_mask):
    if battle.teampreview:
        print_teampreview_actions(battle)
        return

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


def _modify_mask(action_mask: torch.Tensor, action1):
    mask2 = action_mask[1].clone().bool()
    if 1 <= action1 and action1 <= 6:
        mask2[action1] = 0
    elif (26 < action1) and (action1 <= 46):
        mask2[27:47] = 0
    elif action1 == 0:
        mask2[0] = 0

    no_valid = mask2.sum(-1) == 0
    if no_valid:
        mask2[0] = 1

    return mask2


def get_opponent(fmt: str, team: Teambuilder) -> Player:
    name = "p" + uuid.uuid4().hex[:16]
    num = random.random()
    if num < 0.35:
        return FuzzyHeuristic(
            k=2,
            account_configuration=AccountConfiguration(name, None),
            battle_format=fmt,
            server_configuration=LocalhostServerConfiguration,
            team=team,
            accept_open_team_sheet=True,
        )
    elif num < 0.7:
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


class ReplayRecordingPlayer(Player, ABC):
    def __init__(self, save_dir, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.save_dir = save_dir
        self.episode_data = []

    def _save_episode_data(self):
        if not self.episode_data:
            return None

        save_dir = Path(self.save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)

        uid = f"{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')}_{uuid.uuid4().hex[:10]}"
        save_path = save_dir / f"{uid}.replay"
        tmp_path = save_dir / f"{uid}.replay.tmp"

        torch.save(self.episode_data, tmp_path)
        tmp_path.replace(save_path)
        print(f"Replays saved to {save_path}")

        self.episode_data = []
        return str(save_path)

    def get_observation(self, battle: AbstractBattle):
        assert isinstance(battle, DoubleBattle)
        return observation_builder.from_battle(battle)

    async def choose_move(self, battle: AbstractBattle):
        assert isinstance(battle, DoubleBattle)
        if battle._wait:
            return DefaultBattleOrder()

        obs = self.get_observation(battle)
        action_mask = observation_builder.get_action_mask(battle)

        action_np = await self.get_action(battle, action_mask)
        self.episode_data.append({"obs": obs, "mask": action_mask, "action": action_np})
        return Gen9VGCEnv.action_to_order(action_np, battle)

    async def _handle_battle_request(
        self, battle: AbstractBattle, maybe_default_order: bool = False
    ):
        if battle.teampreview:
            obs = self.get_observation(battle)
            action_mask = observation_builder.get_action_mask(battle)
            action_np = await self.get_action(battle, action_mask)
            self.episode_data.append({"obs": obs, "mask": action_mask, "action": action_np})
            order = Gen9VGCEnv.action_to_order(action_np, battle)
            await self.ps_client.send_message(order.message, battle.battle_tag)
        else:
            await super()._handle_battle_request(battle, maybe_default_order)

    @abstractmethod
    async def get_action(self, battle: DoubleBattle, action_mask: torch.Tensor) -> np.ndarray:
        pass

    def teampreview(self, battle: AbstractBattle) -> str:
        # This is now handled in _handle_battle_request to allow recording
        return super().random_teampreview(battle)


class TerminalPlayer(ReplayRecordingPlayer):
    def __init__(self, save_dir, *args, **kwargs):
        super().__init__(save_dir, *args, **kwargs)

    async def get_action(self, battle: DoubleBattle, action_mask: torch.Tensor) -> np.ndarray:
        action_mask_cpu = action_mask.detach().cpu()
        print_valid_actions_from_mask(battle, action_mask_cpu)

        while True:
            line = await asyncio.to_thread(input, "Enter two actions (e.g., '10 15'): ")
            try:
                action = [int(x) for x in line.split()]
                if len(action) < 2:
                    print("Error: Please input two actions. Re-input your two choices.")
                    continue
                action = action[:2]
                action1, action2 = action
            except ValueError:
                print(
                    "Error: Invalid input. Please enter two numbers separated by a space. Re-input your two choices."
                )
                continue

            if not action_mask_cpu[0, action1] or not action_mask_cpu[1, action2]:
                print(
                    "Error: One or more of your chosen actions are invalid. Re-input your two choices."
                )
                continue

            if battle.teampreview:
                return np.asarray(action, dtype=np.int64)

            if (1 <= action1 <= 6) and action1 == action2:
                print(
                    "Error: Mutually exclusive actions selected. Both pokemon cannot switch to the same slot. Re-input your two choices."
                )
                continue

            if (27 <= action1 <= 46) and (27 <= action2 <= 46):
                print(
                    "Error: Mutually exclusive actions selected. Both pokemon cannot terastallize. Re-input your two choices."
                )
                continue

            if action1 == 0 and action2 == 0:
                m2 = _modify_mask(action_mask_cpu, action1)
                if not m2[action2]:
                    print(
                        "Error: Mutually exclusive actions selected. Both pokemon cannot pass. Re-input your two choices."
                    )
                    continue

            return np.asarray(action, dtype=np.int64)


class StrategyRecordingPlayer(ReplayRecordingPlayer):
    def __init__(self, strategy_player: Player, save_dir, *args, **kwargs):
        super().__init__(save_dir, *args, **kwargs)
        self.strategy_player = strategy_player

    async def get_action(self, battle: DoubleBattle, action_mask: torch.Tensor) -> np.ndarray:
        if battle.teampreview:
            res = self.strategy_player.teampreview(battle)
            if isinstance(res, Awaitable):
                res = await res
            order = SingleBattleOrder(res)
        else:
            order = self.strategy_player.choose_move(battle)
            if isinstance(order, Awaitable):
                order = await order

        action = Gen9VGCEnv.order_to_action(order, battle, fake=True, strict=False)

        if not battle.teampreview:
            if action[0] < 0:
                action[0] = 0
            if not action_mask[0, action[0]]:
                valid_indices = torch.where(action_mask[0])[0]
                action[0] = valid_indices[0].item()

            mask2 = _modify_mask(action_mask, action[0])

            if action[1] < 0:
                action[1] = 0
            if not mask2[action[1]]:
                valid_indices = torch.where(mask2)[0]
                action[1] = valid_indices[0].item()
        else:
            action[0] = np.clip(action[0], 0, 35)
            action[1] = np.clip(action[1], 0, 35)

        return action


async def main():
    parser = argparse.ArgumentParser(description="Replay Generator")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--mbp", action="store_true", help="Run with Max Base Power strategy")
    group.add_argument("--sh", action="store_true", help="Run with Simple Heuristic strategy")
    group.add_argument("--fuzzy", action="store_true", help="Run with Fuzzy Heuristic strategy")
    group.add_argument(
        "--interactive", action="store_true", help="Run interactively as Terminal Player"
    )
    parser.add_argument("-n", type=int, default=100, help="Number of battles to play")

    args = parser.parse_args()

    teams_dir = "./teams"
    team_files = [
        path.read_text(encoding="utf-8") for path in Path(teams_dir).iterdir() if path.is_file()
    ]
    team = RandomTeamFromPool(team_files)
    fmt = "gen9vgc2025regh"

    if args.interactive:
        save_dir = "./replays/interactive"
        player = TerminalPlayer(
            save_dir=save_dir,
            account_configuration=AccountConfiguration("TermPlayer", None),
            battle_format=fmt,
            server_configuration=LocalhostServerConfiguration,
            team=team,
            accept_open_team_sheet=True,
            max_concurrent_battles=1,
        )
    elif args.mbp:
        save_dir = "./replays/max_base_power"
        strategy = MaxBasePowerPlayer(
            account_configuration=AccountConfiguration("MBP_Strategy", None),
            battle_format=fmt,
            server_configuration=LocalhostServerConfiguration,
            team=team,
            accept_open_team_sheet=True,
            start_listening=False,
        )
        player = StrategyRecordingPlayer(
            strategy_player=strategy,
            save_dir=save_dir,
            account_configuration=AccountConfiguration("BotPlayer", None),
            battle_format=fmt,
            server_configuration=LocalhostServerConfiguration,
            team=team,
            accept_open_team_sheet=True,
            max_concurrent_battles=1,
        )
    elif args.sh:
        save_dir = "./replays/simple_heuristic"
        strategy = SimpleHeuristicsPlayer(
            account_configuration=AccountConfiguration("SH_Strategy", None),
            battle_format=fmt,
            server_configuration=LocalhostServerConfiguration,
            team=team,
            accept_open_team_sheet=True,
            start_listening=False,
        )
        player = StrategyRecordingPlayer(
            strategy_player=strategy,
            save_dir=save_dir,
            account_configuration=AccountConfiguration("BotPlayer", None),
            battle_format=fmt,
            server_configuration=LocalhostServerConfiguration,
            team=team,
            accept_open_team_sheet=True,
            max_concurrent_battles=1,
        )
    elif args.fuzzy:
        save_dir = "./replays/fuzzy_heuristic"
        strategy = FuzzyHeuristic(
            k=3,
            account_configuration=AccountConfiguration("Fuzzy_Strategy", None),
            battle_format=fmt,
            server_configuration=LocalhostServerConfiguration,
            team=team,
            accept_open_team_sheet=True,
            start_listening=False,
        )
        player = StrategyRecordingPlayer(
            strategy_player=strategy,
            save_dir=save_dir,
            account_configuration=AccountConfiguration("BotPlayer", None),
            battle_format=fmt,
            server_configuration=LocalhostServerConfiguration,
            team=team,
            accept_open_team_sheet=True,
            max_concurrent_battles=1,
        )
    else:
        return

    n_battles = args.n
    for i in range(1, n_battles + 1):
        print(f"Battle {i}/{n_battles}")
        selected_opp = get_opponent(fmt, team)
        await player.battle_against(selected_opp, n_battles=1)
        await selected_opp.ps_client.stop_listening()

        player._save_episode_data()

    player._save_episode_data()
    await player.ps_client.stop_listening()


if __name__ == "__main__":
    asyncio.run(main())
