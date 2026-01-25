from pathlib import Path
from typing import Optional, Union

import numpy as np
import numpy.typing as npt
from gymnasium.spaces import MultiDiscrete
from poke_env.battle import AbstractBattle, DoubleBattle, Pokemon
from poke_env.environment.env import ObsType, PokeEnv
from poke_env.player.battle_order import (
    BattleOrder,
    DefaultBattleOrder,
    DoubleBattleOrder,
    ForfeitBattleOrder,
    PassBattleOrder,
    SingleBattleOrder,
)
from poke_env.player.player import Player
from poke_env.ps_client import (
    AccountConfiguration,
    LocalhostServerConfiguration,
    ServerConfiguration,
)
from poke_env.teambuilder import Teambuilder

import observation_builder
from teams import RandomTeamFromPool


# modified Gen9VGCEnv from poke-env
# to remove all other gimmicks but tera
class Gen9VGCEnv(PokeEnv[ObsType, npt.NDArray[np.int64]]):
    def __init__(
        self,
        account_configuration1: Optional[AccountConfiguration] = None,
        account_configuration2: Optional[AccountConfiguration] = None,
        avatar: Optional[int] = None,
        battle_format: str = "gen9vgc2025regh",
        log_level: Optional[int] = None,
        save_replays: Union[bool, str] = False,
        server_configuration: Optional[ServerConfiguration] = LocalhostServerConfiguration,
        accept_open_team_sheet: Optional[bool] = True,
        start_timer_on_battle_start: bool = True,
        start_listening: bool = True,
        open_timeout: Optional[float] = 10.0,
        ping_interval: Optional[float] = 20.0,
        ping_timeout: Optional[float] = 20.0,
        challenge_timeout: Optional[float] = 60.0,
        team: Optional[Union[str, Teambuilder]] = None,
        fake: bool = False,
        strict: bool = True,
    ):
        super().__init__(
            account_configuration1=account_configuration1,
            account_configuration2=account_configuration2,
            avatar=avatar,
            battle_format=battle_format,
            log_level=log_level,
            save_replays=save_replays,
            server_configuration=server_configuration,
            accept_open_team_sheet=accept_open_team_sheet,
            start_timer_on_battle_start=start_timer_on_battle_start,
            start_listening=start_listening,
            open_timeout=open_timeout,
            ping_interval=ping_interval,
            ping_timeout=ping_timeout,
            challenge_timeout=challenge_timeout,
            team=team,
            fake=fake,
            strict=strict,
        )
        num_switches = 6
        num_moves = 4
        num_targets = 5
        num_gimmicks = 1
        act_size = 1 + num_switches + num_moves * num_targets * (num_gimmicks + 1)
        self.action_spaces = {
            agent: MultiDiscrete([act_size, act_size]) for agent in self.possible_agents
        }

    @staticmethod
    def action_to_order(
        action: npt.NDArray[np.int64],
        battle: DoubleBattle,
        fake: bool = False,
        strict: bool = True,
    ) -> BattleOrder:
        """Convert an action array into a :class:`BattleOrder`.

        The action is a list in doubles, and the individual action mapping is
        as follows, where each 5-long range for a move corresponds to a
        different target (-2, -1, 0, 1, 2).
        -2 = default
        -1 = self
        0 = ally
        1 = opponent1
        2 = opponent2

        element = -2: default
        element = -1: forfeit
        element = 0: pass
        1 <= element <= 6: switch
        7 <= element <= 11: move 1
        12 <= element <= 16: move 2
        17 <= element <= 21: move 3
        22 <= element <= 26: move 4
        27 <= element <= 31: move 1 and terastallize
        32 <= element <= 36: move 2 and terastallize
        37 <= element <= 41: move 3 and terastallize
        42 <= element <= 46: move 4 and terastallize

        :param action: The action to take.
        :type action: ndarray[int64]
        :param battle: The current battle state
        :type battle: AbstractBattle
        :param fake: If ``True``, return a best-effort order even if it would be
            illegal.
        :type fake: bool
        :param strict: If ``True``, raise an error when the action is illegal;
            otherwise return a default order.
        :type strict: bool

        :return: The battle order for the given action in context of the current battle.
        :rtype: BattleOrder

        """
        if action[0] == -2 and action[1] == -2:
            return DefaultBattleOrder()
        elif action[0] == -1 or action[1] == -1:
            return ForfeitBattleOrder()
        try:
            order1 = Gen9VGCEnv._action_to_order_individual(action[0], battle, fake, 0)
        except ValueError as e:
            if strict:
                raise e
            else:
                if battle.logger is not None:
                    battle.logger.warning(str(e) + " Defaulting to random move.")
                order = Player.choose_random_doubles_move(battle)
                order1 = order.first_order if not isinstance(order, DefaultBattleOrder) else order
        try:
            order2 = Gen9VGCEnv._action_to_order_individual(action[1], battle, fake, 1)
        except ValueError as e:
            if strict:
                raise e
            else:
                if battle.logger is not None:
                    battle.logger.warning(str(e) + " Defaulting to random move.")
                order = Player.choose_random_doubles_move(battle)
                order2 = order.second_order if not isinstance(order, DefaultBattleOrder) else order
        joined_orders = DoubleBattleOrder.join_orders([order1], [order2])
        if not joined_orders:
            error_msg = (
                f"Invalid action {action} from player {battle.player_username} "
                f"in battle {battle.battle_tag} - converted orders {order1} "
                f"and {order2} are incompatible!"
            )
            if strict:
                raise ValueError(error_msg)
            else:
                if battle.logger is not None:
                    battle.logger.warning(error_msg + " Defaulting to random move.")
                return Player.choose_random_doubles_move(battle)
        else:
            return joined_orders[0]

    @staticmethod
    def _action_to_order_individual(
        action: np.int64, battle: DoubleBattle, fake: bool, pos: int
    ) -> SingleBattleOrder:
        if action == -2:
            return DefaultBattleOrder()
        elif action == 0:
            order: SingleBattleOrder = PassBattleOrder()
        elif action < 7:
            order = Player.create_order(list(battle.team.values())[action - 1])
        else:
            active_mon = battle.active_pokemon[pos]
            if active_mon is None:
                raise ValueError(
                    f"Invalid order from player {battle.player_username} "
                    f"in battle {battle.battle_tag} at position {pos} - action "
                    f"specifies a move, but battle.active_pokemon is None!"
                )
            mvs = (
                battle.available_moves[pos]
                if len(battle.available_moves[pos]) == 1
                and battle.available_moves[pos][0].id in ["struggle", "recharge"]
                else list(active_mon.moves.values())
            )
            if (action - 7) % 20 // 5 not in range(len(mvs)):
                raise ValueError(
                    f"Invalid action {action} from player {battle.player_username} "
                    f"in battle {battle.battle_tag} at position {pos} - action "
                    f"specifies a move but the move index {(action - 7) % 20 // 5} "
                    f"is out of bounds for available moves {mvs}!"
                )
            order = Player.create_order(
                mvs[(action - 7) % 20 // 5],
                move_target=(action.item() - 7) % 5 - 2,
                terastallize=(action - 7) // 20 == 1,
            )
        if not fake and str(order) not in [str(o) for o in battle.valid_orders[pos]]:
            raise ValueError(
                f"Invalid action {action} from player {battle.player_username} "
                f"in battle {battle.battle_tag} at position {pos} - order {order} "
                f"not in action space {[str(o) for o in battle.valid_orders[pos]]}!"
            )
        return order

    @staticmethod
    def order_to_action(
        order: BattleOrder,
        battle: DoubleBattle,
        fake: bool = False,
        strict: bool = True,
    ) -> npt.NDArray[np.int64]:
        """Convert a :class:`BattleOrder` into an action array.

        :param order: The order to take.
        :type order: BattleOrder
        :param battle: The current battle state
        :type battle: AbstractBattle
        :param fake: If ``True``, return a best-effort action even if it would be
            illegal.
        :type fake: bool
        :param strict: If ``True``, raise an error when the order is illegal;
            otherwise return default.
        :type strict: bool

        :return: The action for the given battle order in context of the current battle.
        :rtype: ndarray[int64]
        """
        if isinstance(order, DefaultBattleOrder):
            return np.array([-2, -2])
        elif isinstance(order, ForfeitBattleOrder):
            return np.array([-1, -1])
        assert isinstance(order, DoubleBattleOrder)
        joined_orders = DoubleBattleOrder.join_orders([order.first_order], [order.second_order])
        if not fake and not joined_orders:
            error_msg = (
                f"Invalid order {order} from player {battle.player_username} "
                f"in battle {battle.battle_tag} - orders are incompatible!"
            )
            if strict:
                raise ValueError(str(error_msg) + " Defaulting to random move.")
            else:
                if battle.logger is not None:
                    battle.logger.warning(error_msg)
                return Gen9VGCEnv.order_to_action(
                    Player.choose_random_doubles_move(battle), battle, fake, strict
                )
        try:
            action1 = Gen9VGCEnv._order_to_action_individual(order.first_order, battle, fake, 0)
        except ValueError as e:
            if strict:
                raise e
            else:
                if battle.logger is not None:
                    battle.logger.warning(str(e) + " Defaulting to random move.")
                order = Player.choose_random_doubles_move(battle)
                action1 = Gen9VGCEnv._order_to_action_individual(
                    (order.first_order if not isinstance(order, DefaultBattleOrder) else order),
                    battle,
                    fake,
                    0,
                )
        try:
            action2 = Gen9VGCEnv._order_to_action_individual(order.second_order, battle, fake, 1)
        except ValueError as e:
            if strict:
                raise e
            else:
                if battle.logger is not None:
                    battle.logger.warning(str(e) + " Defaulting to random move.")
                order = Player.choose_random_doubles_move(battle)
                action2 = Gen9VGCEnv._order_to_action_individual(
                    (order.second_order if not isinstance(order, DefaultBattleOrder) else order),
                    battle,
                    fake,
                    1,
                )
        return np.array([action1, action2])

    @staticmethod
    def _order_to_action_individual(
        order: SingleBattleOrder, battle: DoubleBattle, fake: bool, pos: int
    ) -> np.int64:
        if isinstance(order.order, str):
            if isinstance(order, DefaultBattleOrder):
                return np.int64(-2)
            else:
                assert isinstance(order, PassBattleOrder)
                return np.int64(0)
        if not fake and str(order) not in [str(o) for o in battle.valid_orders[pos]]:
            raise ValueError(
                f"Invalid order from player {battle.player_username} in battle "
                f"{battle.battle_tag} at position {pos} - order {order} not in "
                f"action space {[str(o) for o in battle.valid_orders[pos]]}!"
            )
        if isinstance(order.order, Pokemon):
            action = [p.base_species for p in battle.team.values()].index(
                order.order.base_species
            ) + 1
        else:
            active_mon = battle.active_pokemon[pos]
            assert active_mon is not None
            mvs = (
                battle.available_moves[pos]
                if len(battle.available_moves[pos]) == 1
                and battle.available_moves[pos][0].id in ["struggle", "recharge"]
                else list(active_mon.moves.values())
            )
            action = [m.id for m in mvs].index(order.order.id)
            target = order.move_target + 2
            if order.terastallize:
                gimmick = 1
            else:
                gimmick = 0
            action = 1 + 6 + 5 * action + target + 20 * gimmick
        return np.int64(action)


class SimEnv(Gen9VGCEnv):
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
        return observation_builder.from_battle(battle)
