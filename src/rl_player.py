import asyncio
from pathlib import Path

import torch
from poke_env import AccountConfiguration, LocalhostServerConfiguration
from poke_env.battle import AbstractBattle, DoubleBattle
from poke_env.player import DefaultBattleOrder, MaxBasePowerPlayer, Player, RandomPlayer
from torch.distributions import Categorical

from encoder import ACT_SIZE, OBS_DIM, Encoder
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
        k: int = ACT_SIZE,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.policy = policy
        self.teampreview_handler = teampreview_handler

        assert k >= 1 and k <= ACT_SIZE
        self.k = k

    def _top_k(self, obs, action_mask):
        B = obs.shape[0]
        z = self.policy.reducer(obs)
        x = self.policy.shared_backbone(z)
        policy_logits = self.policy.policy_head(x).reshape(B, 2, self.policy.act_size)
        logits = self.policy._apply_masks(policy_logits, action_mask)

        p1 = logits[:, 0]
        idx1 = torch.topk(p1, self.k, dim=-1).indices  # (B, k)
        # binary mask with 1 at top-k positions
        topk_mask1 = torch.zeros_like(p1)
        topk_mask1.scatter_(1, idx1, 1.0)
        p1 = p1.masked_fill(topk_mask1 == 0, float("-inf"))

        cat1 = Categorical(logits=p1)
        action1 = cat1.sample()  # (B,)

        logits = self.policy._apply_sequential_masks(logits, action1, action_mask)
        p2 = logits[:, 1]
        idx2 = torch.topk(p2, self.k, dim=-1).indices
        topk_mask2 = torch.zeros_like(p2)
        topk_mask2.scatter_(1, idx2, 1.0)
        p2 = p2.masked_fill(topk_mask2 == 0, float("-inf"))

        cat2 = Categorical(logits=p2)
        action2 = cat2.sample()  # (B,)

        return torch.stack([action1, action2], dim=-1)

    def _get_action(self, battle: AbstractBattle):
        obs = self.get_observation(battle)
        action_mask = Encoder.get_action_mask(battle)
        with torch.no_grad():
            obs_tensor = torch.as_tensor(obs, device=self.policy.device).unsqueeze(0)
            action_mask_tensor = action_mask.unsqueeze(0)
            actions = self._top_k(obs_tensor, action_mask_tensor)
        return actions[0].cpu().numpy()

    def choose_move(self, battle: AbstractBattle):
        assert isinstance(battle, DoubleBattle)
        if battle._wait:
            return DefaultBattleOrder()
        return Gen9VGCEnv.action_to_order(self._get_action(battle), battle)

    def get_observation(self, battle: AbstractBattle):
        assert isinstance(battle, DoubleBattle)
        return Encoder.encode_battle_state(battle)

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

    policy = PolicyNet(obs_dim=OBS_DIM, act_size=ACT_SIZE)
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
