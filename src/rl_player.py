import asyncio
from pathlib import Path

import torch
from poke_env import AccountConfiguration, LocalhostServerConfiguration
from poke_env.battle import AbstractBattle, DoubleBattle
from poke_env.player import DefaultBattleOrder, MaxBasePowerPlayer, Player, RandomPlayer

from encoder import ACT_SIZE, OBS_DIM, Encoder
from env import Gen9VGCEnv
from policy import PolicyNet
from teams import RandomTeamFromPool


class RLPlayer(Player):
    def __init__(self, policy: PolicyNet, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.policy = policy

    def _get_action(self, battle: AbstractBattle):
        obs = self.get_observation(battle)
        action_mask = Encoder.get_action_mask(battle)
        with torch.no_grad():
            obs_tensor = torch.as_tensor(obs, device=self.policy.device).unsqueeze(0)
            action_mask_tensor = action_mask.unsqueeze(0)
            _, _, actions, _ = self.policy.forward(
                obs_tensor, action_mask_tensor, sample_actions=True
            )
        return actions[0].cpu().numpy()  # type: ignore

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

    checkpoint_path = "./checkpoints/checkpoint_1210.pt"
    checkpoint = torch.load(checkpoint_path)

    policy = PolicyNet(obs_dim=OBS_DIM, act_size=ACT_SIZE)
    policy.load_state_dict(checkpoint["model_state_dict"])

    rl_player = RLPlayer(
        policy=policy,
        account_configuration=AccountConfiguration("RLPlayer", None),
        battle_format="gen9vgc2025regh",
        server_configuration=LocalhostServerConfiguration,
        max_concurrent_battles=10,
        team=team,
        accept_open_team_sheet=True,
    )

    # Create opponents
    random_player = RandomPlayer(
        account_configuration=AccountConfiguration("RandomPlayer", None),
        battle_format="gen9vgc2025regh",
        server_configuration=LocalhostServerConfiguration,
        max_concurrent_battles=10,
        team=team,
        accept_open_team_sheet=True,
    )

    max_power_player = MaxBasePowerPlayer(
        account_configuration=AccountConfiguration("MaxPowerPlayer", None),
        battle_format="gen9vgc2025regh",
        server_configuration=LocalhostServerConfiguration,
        max_concurrent_battles=10,
        team=team,
        accept_open_team_sheet=True,
    )

    # for statistics
    await rl_player.battle_against(random_player, n_battles=500)
    rand_wr = rl_player.win_rate
    rl_player.reset_battles()

    await rl_player.battle_against(max_power_player, n_battles=500)
    mbp_wr = rl_player.win_rate
    rl_player.reset_battles()

    print(f"Win rate vs RandomPlayer: {rand_wr:.4%}")
    print(f"Win rate vs MaxBasePowerPlayer: {mbp_wr:.4%}")

    # Clean up
    await rl_player.ps_client.stop_listening()
    await random_player.ps_client.stop_listening()
    await max_power_player.ps_client.stop_listening()


if __name__ == "__main__":
    asyncio.run(main())
