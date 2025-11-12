from poke_env.battle import AbstractBattle, DoubleBattle
from poke_env.player import Player, DefaultBattleOrder
from encoder import Encoder, BATTLE_STATE_DIMS
from gen9vgcenv import Gen9VGCEnv
from teams import RandomTeamFromPool
import torch
from pseudo_policy import ACT_SIZE, PseudoPolicy


class RLPlayer(Player):
    def __init__(self, policy: PseudoPolicy, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.policy = policy
        self.team = RandomTeamFromPool

    def choose_move(self, battle: AbstractBattle):
        assert isinstance(battle, DoubleBattle)
        if battle._wait:
            return DefaultBattleOrder()
        obs = self.get_observation(battle)
        action_mask = self.get_action_mask(battle)
        with torch.no_grad():
            obs_tensor = torch.as_tensor(obs, device=self.policy.device).unsqueeze(0)
            action_pair_np, _, _ = self.policy.forward(obs_tensor, action_mask)
        return Gen9VGCEnv.action_to_order(action_pair_np, battle)

    def get_observation(self, battle: DoubleBattle):
        obs = torch.Tensor(BATTLE_STATE_DIMS)
        Encoder.encode_battle_state(battle, obs)
        return obs

    def get_action_mask(self, battle: DoubleBattle):
        """
        Returns a [2, ACT_SIZE] action mask for both active Pokémon.
        Each row is a mask for the legal actions of that Pokémon.
        """

        def single_action_mask(battle: DoubleBattle, pos: int) -> list[int]:
            act_len = ACT_SIZE  # For your action index size
            switch_space = [
                i + 1
                for i, pokemon in enumerate(battle.team.values())
                if not battle.trapped[pos]
                and pokemon.base_species in [p.base_species for p in battle.available_switches[pos]]
            ]
            active_mon = battle.active_pokemon[pos]
            # "Pass" or forced switch handling
            if battle._wait or (any(battle.force_switch) and not battle.force_switch[pos]):
                actions = [0]
            elif all(battle.force_switch) and len(battle.available_switches[0]) == 1:
                actions = switch_space + [0]
            elif battle.teampreview or active_mon is None:
                actions = switch_space
            else:
                # Build move action indices
                move_spaces = [
                    [
                        7 + 5 * i + j + 2
                        for j in battle.get_possible_showdown_targets(move, active_mon)
                    ]
                    for i, move in enumerate(active_mon.moves.values())
                    if move.id in [m.id for m in battle.available_moves[pos]]
                ]
                move_space = [i for s in move_spaces for i in s]
                # Terastallization
                tera_space = [
                    i + 20 for i in move_space if getattr(battle, "can_tera", [False, False])[pos]
                ]
                # Struggle/recharge
                if (
                    not move_space
                    and len(battle.available_moves[pos]) == 1
                    and battle.available_moves[pos][0].id in ["struggle", "recharge"]
                ):
                    move_space = [9]
                actions = switch_space + move_space + tera_space
            actions = actions or [0]
            action_mask = [int(i in actions) for i in range(act_len)]
            return action_mask

        # Stack for both active Pokémon (positions 0 and 1)
        mask0 = single_action_mask(battle, 0)
        mask1 = single_action_mask(battle, 1)
        # Return as a torch tensor of shape (2, ACT_SIZE)
        return torch.tensor([mask0, mask1], dtype=torch.uint8)

    def teampreview(self, battle: AbstractBattle) -> str:
        # defaults to random team preview
        return super().teampreview(battle)
