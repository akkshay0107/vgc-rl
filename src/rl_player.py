import torch
from poke_env.battle import AbstractBattle, DoubleBattle
from poke_env.player import DefaultBattleOrder, Player
from torch.distributions import Categorical

import observation_builder
from env import Gen9VGCEnv
from policy import PolicyNet
from ppo_utils import initial_state


class RLPlayer(Player):
    """
    Class that plays moves as per the trained policy net.
    """

    def __init__(
        self,
        policy: PolicyNet,
        p: float = 0.9,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.policy = policy

        assert 0.0 <= p <= 1.0
        self.p = p
        self.state = None

    def _apply_top_p(self, logits: torch.Tensor) -> torch.Tensor:
        sorted_logits, sorted_indices = torch.sort(logits, descending=True, dim=-1)
        sorted_probs = torch.softmax(sorted_logits, dim=-1)
        cumulative_probs = torch.cumsum(sorted_probs, dim=-1)

        sorted_indices_to_remove = cumulative_probs > self.p
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = False

        sorted_logits[sorted_indices_to_remove] = float("-inf")

        return sorted_logits.scatter(-1, sorted_indices, sorted_logits)

    def _top_p(self, obs, action_mask, is_tp: bool):
        if self.state is None:
            self.state = initial_state(self.policy, 1, self.policy.device)

        policy_logits, _, _, _, self.state = self.policy(
            obs, self.state, action_mask, sample_actions=False
        )
        logits = self.policy._apply_masks(policy_logits, action_mask)

        p1_logits = self._apply_top_p(logits[:, 0])
        cat1 = Categorical(logits=p1_logits)
        action1 = cat1.sample()  # (B,)

        is_tp_t = torch.tensor([is_tp], device=self.policy.device, dtype=torch.bool)
        logits = self.policy._apply_sequential_masks(logits, action1, action_mask, is_tp_t)
        p2_logits = self._apply_top_p(logits[:, 1])
        cat2 = Categorical(logits=p2_logits)
        action2 = cat2.sample()  # (B,)

        return torch.stack([action1, action2], dim=-1)

    def _get_action(self, battle: AbstractBattle, is_tp: bool):
        obs = self.get_observation(battle)
        action_mask = observation_builder.get_action_mask(battle)
        with torch.no_grad():
            actions = self._top_p(
                obs.unsqueeze(0).to(self.policy.device),
                action_mask.unsqueeze(0).to(self.policy.device),
                is_tp,
            )
        return actions[0].cpu().numpy()

    def choose_move(self, battle: AbstractBattle):
        assert isinstance(battle, DoubleBattle)
        if battle._wait:
            return DefaultBattleOrder()
        return Gen9VGCEnv.action_to_order(self._get_action(battle, False), battle)

    def get_observation(self, battle: AbstractBattle):
        assert isinstance(battle, DoubleBattle)
        return observation_builder.from_battle(battle)

    def teampreview(self, battle: AbstractBattle) -> str:
        assert isinstance(battle, DoubleBattle)
        # Team preview is the start of the battle, so we reset the state here
        self.state = None
        action = self._get_action(battle, True)
        order = Gen9VGCEnv.action_to_order(action, battle)
        return order.message

    def _battle_finished_callback(self, battle: AbstractBattle):
        # Reset state at the end of the battle to prevent memory leaks or state carry-over
        self.state = None


if __name__ == "__main__":
    import asyncio
    import sys
    from pathlib import Path

    from poke_env import AccountConfiguration, LocalhostServerConfiguration

    sys.path.append(str(Path(__file__).parent))
    from ppo_utils import load_checkpoint
    from teams import RandomTeamFromPool

    async def main():
        root_dir = Path(__file__).resolve().parent.parent
        teams_dir = root_dir / "teams"

        if not teams_dir.exists():
            print(f"Teams directory not found: {teams_dir}")
            return

        team_files = [
            path.read_text(encoding="utf-8")
            for path in teams_dir.iterdir()
            if path.is_file() and not path.name.startswith(".")
        ]

        if not team_files:
            print(f"No team files found in {teams_dir}.")
            return

        team = RandomTeamFromPool(team_files)
        fmt = "gen9vgc2025regh"

        policy = PolicyNet()
        load_checkpoint(Path("./checkpoints/model_final.pt"), policy)

        bot_player = RLPlayer(
            policy=policy,
            account_configuration=AccountConfiguration("Bot", None),
            battle_format=fmt,
            server_configuration=LocalhostServerConfiguration,
            team=team,
            accept_open_team_sheet=True,
            max_concurrent_battles=1,
        )

        print("Bot is listening for challenges on localhost...")
        await bot_player.accept_challenges(None, 1000)

    asyncio.run(main())
