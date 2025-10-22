from typing import Optional, Union
from poke_env import Player
from poke_env import AccountConfiguration, ServerConfiguration
from random_team_from_pool import RandomTeamFromPool
from pathlib import Path

class MCTSAgent(Player):
    def __init__(
        self,
        account_configuration: Optional[AccountConfiguration] = None, *,
        avatar: Optional[str] = None,
        battle_format: str = "gen9randombattle",
        log_level: Optional[int] = None,
        max_concurrent_battles: int = 1,
        accept_open_team_sheet: bool = False,
        save_replays: Union[bool, str] = False,
        server_configuration: ServerConfiguration = ...,
        start_timer_on_battle_start: bool = False,
        start_listening: bool = True,
        open_timeout: Optional[float] = 10,
        ping_interval: Optional[float] = 20,
        ping_timeout: Optional[float] = 20,
    ):
        teams_dir = '../teams'
        teams = [
            path.read_text(encoding='utf-8')
            for path in Path(teams_dir).iterdir()
            if path.is_file()
        ]

        super().__init__(
            account_configuration,
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
            team=RandomTeamFromPool(teams)
        )

    def choose_move(self, battle):
        # implement move choice algorithm here
        return self.choose_default_move()


if __name__ == "__main__":
    pass
