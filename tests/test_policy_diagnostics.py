import asyncio
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
from poke_env import AccountConfiguration, LocalhostServerConfiguration
from torch.distributions import Categorical

# Add src to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from policy import PolicyNet
from ppo_utils import initial_state, load_checkpoint
from rl_player import RLPlayer
from teams import RandomTeamFromPool


class DiagnosticRLPlayer(RLPlayer):
    def _top_p(self, obs, action_mask, is_tp: bool):
        if self.state is None:
            self.state = initial_state(self.policy, 1, self.policy.device)

        policy_logits, _, _, _, self.state = self.policy(
            obs, self.state, action_mask, sample_actions=False
        )
        logits = self.policy._apply_masks(policy_logits, action_mask)

        # Diagnostics for Pokemon 1
        p1_logits_all = logits[0, 0]
        p1_valid_mask = p1_logits_all > float("-inf")
        p1_valid_indices = torch.where(p1_valid_mask)[0]

        # We apply top-p as RLPlayer does
        p1_logits_tp = self._apply_top_p(logits[:, 0])
        p1_probs_tp = F.softmax(p1_logits_tp[0], dim=-1)

        print(
            f"\n[Selection] Player: {self.username} | Phase: {'Team Preview' if is_tp else 'Battle'}"
        )
        print("  Pokemon 1:")
        print(f"    Valid Indices: {p1_valid_indices.tolist()}")
        print(f"    Top-p Probs:   {[f'{p:.4f}' for p in p1_probs_tp[p1_valid_indices].tolist()]}")

        # Sample action 1
        cat1 = Categorical(logits=p1_logits_tp)
        action1 = cat1.sample()

        print(f"    Selected Action: {action1.item()}")

        # Diagnostics for Pokemon 2
        is_tp_t = torch.tensor([is_tp], device=self.policy.device, dtype=torch.bool)
        logits = self.policy._apply_sequential_masks(logits, action1, action_mask, is_tp_t)

        p2_logits_all = logits[0, 1]
        p2_valid_mask = p2_logits_all > float("-inf")
        p2_valid_indices = torch.where(p2_valid_mask)[0]

        p2_logits_tp = self._apply_top_p(logits[:, 1])
        p2_probs_tp = F.softmax(p2_logits_tp[0], dim=-1)

        print("  Pokemon 2:")
        print(f"    Valid Indices: {p2_valid_indices.tolist()}")
        print(f"    Top-p Probs:   {[f'{p:.4f}' for p in p2_probs_tp[p2_valid_indices].tolist()]}")

        cat2 = Categorical(logits=p2_logits_tp)
        action2 = cat2.sample()
        print(f"    Selected Action: {action2.item()}")

        return torch.stack([action1, action2], dim=-1)


async def run_diagnostic_battle():
    root_dir = Path(__file__).resolve().parent.parent
    teams_dir = root_dir / "teams"
    checkpoint_path = root_dir / "checkpoints" / "ppo_checkpoint.pt"

    team_files = [
        path.read_text(encoding="utf-8")
        for path in teams_dir.iterdir()
        if path.is_file() and not path.name.startswith(".")
    ]

    if not team_files:
        print(f"No team files found in {teams_dir}")
        return

    team = RandomTeamFromPool(team_files)

    device = torch.device("cpu")
    policy = PolicyNet().to(device)
    policy.device = device

    if checkpoint_path.exists():
        print(f"Loading checkpoint from: {checkpoint_path}")
        load_checkpoint(checkpoint_path, policy)
    else:
        print(
            f"Warning: Checkpoint not found at {checkpoint_path}. Using randomly initialized policy."
        )

    p1 = DiagnosticRLPlayer(
        policy=policy,
        account_configuration=AccountConfiguration("Player 1", None),
        server_configuration=LocalhostServerConfiguration,
        battle_format="gen9vgc2025regh",
        team=team,
        accept_open_team_sheet=True,
        p=1.0,
    )
    p2 = DiagnosticRLPlayer(
        policy=policy,
        account_configuration=AccountConfiguration("Player 2", None),
        server_configuration=LocalhostServerConfiguration,
        battle_format="gen9vgc2025regh",
        team=team,
        accept_open_team_sheet=True,
        p=1.0,
    )

    try:
        await p1.battle_against(p2, n_battles=1)
    except Exception as e:
        print(f"Error during battle: {e}")
    finally:
        await p1.ps_client.stop_listening()
        await p2.ps_client.stop_listening()
    print("Battle finished.")


if __name__ == "__main__":
    asyncio.run(run_diagnostic_battle())
