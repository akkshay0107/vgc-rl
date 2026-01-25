import asyncio
from pathlib import Path

import torch
from poke_env import AccountConfiguration, LocalhostServerConfiguration
from poke_env.battle import AbstractBattle, DoubleBattle
from poke_env.player import DefaultBattleOrder, MaxBasePowerPlayer, Player, RandomPlayer
from torch.distributions import Categorical

import observation_builder
from env import Gen9VGCEnv
from policy import PolicyNet
from teampreview import TeamPreviewHandler
from teams import RandomTeamFromPool


class RLPlayer(Player):
    """
    Class that plays moves as per the trained policy net.
    """

    def __init__(
        self,
        policy: PolicyNet,
        teampreview_handler: TeamPreviewHandler,
        p: float = 0.9,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.policy = policy
        self.teampreview_handler = teampreview_handler

        assert 0.0 <= p <= 1.0
        self.p = p

    def _apply_top_p(self, logits: torch.Tensor) -> torch.Tensor:
        sorted_logits, sorted_indices = torch.sort(logits, descending=True, dim=-1)
        sorted_probs = torch.softmax(sorted_logits, dim=-1)
        cumulative_probs = torch.cumsum(sorted_probs, dim=-1)

        sorted_indices_to_remove = cumulative_probs > self.p
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = False

        sorted_logits[sorted_indices_to_remove] = float("-inf")

        return sorted_logits.scatter(-1, sorted_indices, sorted_logits)

    def _top_p(self, obs, action_mask):
        policy_logits, _, _, _ = self.policy(obs, action_mask, sample_actions=False)
        logits = self.policy.actor_head._apply_masks(policy_logits, action_mask)

        p1_logits = self._apply_top_p(logits[:, 0])
        cat1 = Categorical(logits=p1_logits)
        action1 = cat1.sample()  # (B,)

        logits = self.policy.actor_head._apply_sequential_masks(logits, action1, action_mask)
        p2_logits = self._apply_top_p(logits[:, 1])
        cat2 = Categorical(logits=p2_logits)
        action2 = cat2.sample()  # (B,)

        return torch.stack([action1, action2], dim=-1)

    def _get_action(self, battle: AbstractBattle):
        obs = self.get_observation(battle)
        action_mask = observation_builder.get_action_mask(battle)
        with torch.no_grad():
            actions = self._top_p(obs, action_mask.unsqueeze(0))
        return actions[0].cpu().numpy()

    def choose_move(self, battle: AbstractBattle):
        assert isinstance(battle, DoubleBattle)
        if battle._wait:
            return DefaultBattleOrder()
        return Gen9VGCEnv.action_to_order(self._get_action(battle), battle)

    def get_observation(self, battle: AbstractBattle):
        assert isinstance(battle, DoubleBattle)
        return observation_builder.from_battle(battle)

    def teampreview(self, battle: AbstractBattle) -> str:
        return self.teampreview_handler.select_team(battle)


async def main():
    teams_dir = "./teams"
    team_files = [
        path.read_text(encoding="utf-8") for path in Path(teams_dir).iterdir() if path.is_file()
    ]
    team = RandomTeamFromPool(team_files)
    fmt = "gen9vgc2025regh"

    checkpoint_path = "PUT CHECKPOINT PATH HERE"
    checkpoint = torch.load(checkpoint_path)

    policy = PolicyNet()
    policy.load_state_dict(checkpoint["model_state_dict"])

    tp_handler = TeamPreviewHandler()

    rl_player = RLPlayer(
        policy=policy,
        teampreview_handler=tp_handler,
        account_configuration=AccountConfiguration("RLPlayer", None),
        battle_format=fmt,
        server_configuration=LocalhostServerConfiguration,
        max_concurrent_battles=10,
        team=team,
        accept_open_team_sheet=True,
    )

    # Create opponents
    random_player = RandomPlayer(
        account_configuration=AccountConfiguration("RandomPlayer", None),
        battle_format=fmt,
        server_configuration=LocalhostServerConfiguration,
        max_concurrent_battles=10,
        team=team,
        accept_open_team_sheet=True,
    )

    max_power_player = MaxBasePowerPlayer(
        account_configuration=AccountConfiguration("MaxPowerPlayer", None),
        battle_format=fmt,
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
