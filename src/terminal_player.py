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
def get_opponent(fmt: str, team: Teambuilder) -> Player:
    name = uuid.uuid4().hex[:7]
    num = random.random()
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
    def __init__(self, save_dir, random_input=False, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.save_dir = save_dir
        self.episode_data = []
        self.random_input = random_input

    def _save_episode_data(self):
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

    def _get_random_move(self, action_mask):
        logits1 = self.logits_from_mask(action_mask[0])
        cat1 = torch.distributions.Categorical(logits=logits1)
        action1 = cat1.sample()

        mask2 = self.sequential_mask(action_mask, action1)
        logits2 = self.logits_from_mask(mask2)
        cat2 = torch.distributions.Categorical(logits=logits2)
        action2 = cat2.sample()
        return np.asarray([action1.item(), action2.item()], dtype=np.int64)

    async def _get_move_from_terminal(self, battle, action_mask):
        action_mask_cpu = action_mask.detach().cpu()
        print_valid_actions_from_mask(battle, action_mask_cpu)

        while True:
            line = await asyncio.to_thread(input, "")
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

            # Check for mutual exclusivity
            # Both switching to same slot
            if (1 <= action1 <= 6) and action1 == action2:
                print(
                    "Error: Mutually exclusive actions selected. Both pokemon cannot switch to the same slot. Re-input your two choices."
                )
                continue

            # Both terastallizing
            if (27 <= action1 <= 46) and (27 <= action2 <= 46):
                print(
                    "Error: Mutually exclusive actions selected. Both pokemon cannot terastallize. Re-input your two choices."
                )
                continue

            # Both passing
            if action1 == 0 and action2 == 0:
                print(
                    "Error: Mutually exclusive actions selected. Both pokemon cannot pass. Re-input your two choices."
                )
                continue

            return np.asarray(action, dtype=np.int64)

    async def choose_move(self, battle: AbstractBattle):
        assert isinstance(battle, DoubleBattle)
        if battle._wait:
            return DefaultBattleOrder()

        obs = self.get_observation(battle)
        action_mask = observation_builder.get_action_mask(battle)

        if self.random_input:
            action_np = self._get_random_move(action_mask)
            self.episode_data.append({"obs": obs, "mask": action_mask, "action": action_np})
            return Gen9VGCEnv.action_to_order(action_np, battle)
        else:
            action_np = await self._get_move_from_terminal(battle, action_mask)
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
        random_input=False,
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
    if not term_player.random_input:
        choice = "y"
        while choice == "y":
            selected_opp = get_opponent(fmt, team)
            await term_player.battle_against(selected_opp, n_battles=1)
            term_player._save_episode_data()

            # Cleanup newly created opponent
            await selected_opp.ps_client.stop_listening()

            choice = input("Play again? (y/n)")[0]
            choice = choice.lower()

        # Cleanup
        await term_player.ps_client.stop_listening()
    else:
        N_BATTLES = 100
        for _ in range(N_BATTLES):
            selected_opp = get_opponent(fmt, team)
            await term_player.battle_against(selected_opp, n_battles=1)
            term_player._save_episode_data()

            # Cleanup newly created opponent
            await selected_opp.ps_client.stop_listening()


if __name__ == "__main__":
    asyncio.run(main())
