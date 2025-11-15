from typing import Optional, Union
from poke_env import Player, RandomPlayer
from poke_env import AccountConfiguration, ServerConfiguration
from teams import RandomTeamFromPool
from pathlib import Path
import asyncio

class MCTSAgent(Player):
    def __init__(
        self,
        account_configuration: Optional[AccountConfiguration] = None, *,
        avatar: Optional[str] = None,
        battle_format: str = "gen9vgc2025regh",
        log_level: Optional[int] = None,
        max_concurrent_battles: int = 1,
        accept_open_team_sheet: bool = True,
        save_replays: Union[bool, str] = False,
        server_configuration: Optional[ServerConfiguration] = None,
        start_timer_on_battle_start: bool = True,
        start_listening: bool = True,
        open_timeout: Optional[float] = 10,
        ping_interval: Optional[float] = 20,
        ping_timeout: Optional[float] = 20,
    ):
        if server_configuration is None:
            server_configuration = ServerConfiguration(
                "ws://localhost:8000/showdown/websocket",
                "https://play.pokemonshowdown.com/action.php?"
            )

        teams_dir = "./teams"
        team_files = [
            path.read_text(encoding="utf-8")
            for path in Path(teams_dir).iterdir()
            if path.is_file()
        ]

        team = RandomTeamFromPool(team_files)

        super().__init__(
            account_configuration=account_configuration,
            avatar=avatar,
            battle_format=battle_format,
            log_level=log_level,
            max_concurrent_battles=max_concurrent_battles,
            accept_open_team_sheet=accept_open_team_sheet,
            save_replays=save_replays,
            server_configuration=server_configuration,
            start_timer_on_battle_start=start_timer_on_battle_start,
            start_listening=start_listening,
            open_timeout=open_timeout,
            ping_interval=ping_interval,
            ping_timeout=ping_timeout,
            team=team
        )

    def choose_move(self, battle):
        # TODO: Replace with MCTS logic
        return self.choose_default_move()


async def main():
    server_cfg = ServerConfiguration(
        "ws://localhost:8000/showdown/websocket",
        "https://play.pokemonshowdown.com/action.php?"
    )

    p1 = MCTSAgent(
        server_configuration=server_cfg
    )

    p2 = RandomPlayer(
        battle_format="gen9vgc2025regh",
        team=p1._team,
        server_configuration=server_cfg
    )

    await p1.battle_against(p2, n_battles=1)


if __name__ == "__main__":
    asyncio.run(main())
