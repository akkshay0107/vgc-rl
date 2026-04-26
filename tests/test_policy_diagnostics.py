import asyncio
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from poke_env import AccountConfiguration, LocalhostServerConfiguration
from poke_env.battle import Pokemon
from torch.distributions import Categorical

# Add src to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import observation_builder
from env import Gen9VGCEnv
from policy import PolicyNet
from ppo_utils import initial_state, load_checkpoint
from rl_player import RLPlayer
from teams import RandomTeamFromPool


class DiagnosticRLPlayer(RLPlayer):
    def choose_move(self, battle):
        print(f"\n{'='*80}")
        print(f"  TURN {battle.turn:2} | Player: {self.username}")
        print(f"{'='*80}")
        return super().choose_move(battle)

    def _get_action(self, battle, is_tp: bool):
        obs = self.get_observation(battle)
        action_mask = observation_builder.get_action_mask(battle)
        actions = self._top_p(
            obs.unsqueeze(0).to(self.policy.device),
            action_mask.unsqueeze(0).to(self.policy.device),
            is_tp,
            battle=battle,
        )
        return actions[0].cpu().numpy()

    def _get_action_description(self, action_idx, battle, is_tp, pos):
        if is_tp:
            p1_idx = action_idx // 6
            p2_idx = action_idx % 6
            team_list = list(battle.team.values())
            p1_name = team_list[p1_idx].species if p1_idx < len(team_list) else "None"
            p2_name = team_list[p2_idx].species if p2_idx < len(team_list) else "None"
            return f"Lead: {p1_name}, {p2_name}" if pos == 0 else f"Back: {p1_name}, {p2_name}"
        else:
            try:
                # Gen9VGCEnv._action_to_order_individual expects np.int64 because it calls .item()
                action_idx_np = np.int64(action_idx)
                order = Gen9VGCEnv._action_to_order_individual(action_idx_np, battle, fake=True, pos=pos)
                if isinstance(order.order, str):
                    return order.order
                if isinstance(order.order, Pokemon):
                    return f"Switch to {order.order.species}"
                if hasattr(order.order, "id"):  # Move
                    target = ""
                    if order.move_target is not None and order.move_target != 0:
                        target = f" (target {order.move_target})"
                    return f"{'Tera ' if order.terastallize else ''}{order.order.id}{target}"
                return str(order)
            except Exception as e:
                return f"Error: {e} (idx={action_idx})"

    def _top_p(self, obs, action_mask, is_tp: bool, battle=None):
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
            f"\n[Selection] Phase: {'Team Preview' if is_tp else 'Battle'}"
        )
        print("  Slot 1:")
        valid_indices_list = p1_valid_indices.tolist()
        probs_list = p1_probs_tp[p1_valid_indices].tolist()
        
        # Sort by probability for better readability
        sorted_p1 = sorted(zip(valid_indices_list, probs_list), key=lambda x: x[1], reverse=True)
        
        for idx, prob in sorted_p1:
            if prob < 0.0001: continue
            desc = self._get_action_description(idx, battle, is_tp, 0)
            print(f"    {prob*100:6.2f}% | Index {idx:2}: {desc}")

        # Sample action 1
        cat1 = Categorical(logits=p1_logits_tp)
        action1 = cat1.sample()

        print(f"    -> SELECTED: {action1.item()} ({self._get_action_description(action1.item(), battle, is_tp, 0)})")

        # Diagnostics for Pokemon 2
        is_tp_t = torch.tensor([is_tp], device=self.policy.device, dtype=torch.bool)
        logits = self.policy._apply_sequential_masks(logits, action1, action_mask, is_tp_t)

        p2_logits_all = logits[0, 1]
        p2_valid_mask = p2_logits_all > float("-inf")
        p2_valid_indices = torch.where(p2_valid_mask)[0]

        p2_logits_tp = self._apply_top_p(logits[:, 1])
        p2_probs_tp = F.softmax(p2_logits_tp[0], dim=-1)

        print("\n  Slot 2:")
        valid_indices_list_2 = p2_valid_indices.tolist()
        probs_list_2 = p2_probs_tp[p2_valid_indices].tolist()
        
        # Sort by probability
        sorted_p2 = sorted(zip(valid_indices_list_2, probs_list_2), key=lambda x: x[1], reverse=True)
        
        for idx, prob in sorted_p2:
            if prob < 0.0001: continue
            desc = self._get_action_description(idx, battle, is_tp, 1)
            print(f"    {prob*100:6.2f}% | Index {idx:2}: {desc}")

        cat2 = Categorical(logits=p2_logits_tp)
        action2 = cat2.sample()
        print(f"    -> SELECTED: {action2.item()} ({self._get_action_description(action2.item(), battle, is_tp, 1)})")

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
