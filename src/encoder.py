import torch
from poke_env.battle import AbstractBattle, DoubleBattle

OBS_DIM = (2, 5, 30)
# Define action space parameters (from gen9vgcenv.py)
NUM_SWITCHES = 6
NUM_MOVES = 4
NUM_TARGETS = 5
NUM_GIMMICKS = 1
ACT_SIZE = 1 + NUM_SWITCHES + NUM_MOVES * NUM_TARGETS * (NUM_GIMMICKS + 1)


class Encoder:
    """
    Static library class containing methods to
    - Get the set of all valid actions from the current battle state
    - Encode the battle state into an observation for the policy network
    """

    @staticmethod
    def get_action_mask(battle: AbstractBattle):
        """
        Returns a [2, ACT_SIZE] action mask for both active Pokémon.
        Each row is a mask for the legal actions of that Pokémon.
        """
        assert isinstance(battle, DoubleBattle)

        # direct copy of vgc-bench action mask
        def single_action_mask(battle: DoubleBattle, pos: int) -> list[int]:
            switch_space = [
                i + 1
                for i, pokemon in enumerate(battle.team.values())
                if not battle.trapped[pos]
                and pokemon.base_species in [p.base_species for p in battle.available_switches[pos]]
            ]
            active_mon = battle.active_pokemon[pos]
            if battle._wait or (any(battle.force_switch) and not battle.force_switch[pos]):
                actions = [0]
            elif all(battle.force_switch) and len(battle.available_switches[0]) == 1:
                actions = switch_space + [0]
            elif battle.teampreview or active_mon is None:
                actions = switch_space
            else:
                move_spaces = [
                    [
                        7 + 5 * i + j + 2
                        for j in battle.get_possible_showdown_targets(move, active_mon)
                    ]
                    for i, move in enumerate(active_mon.moves.values())
                    if move.id in [m.id for m in battle.available_moves[pos]]
                ]
                move_space = [i for s in move_spaces for i in s]
                tera_space = [i + 20 for i in move_space if battle.can_tera[pos]]
                if (
                    not move_space
                    and len(battle.available_moves[pos]) == 1
                    and battle.available_moves[pos][0].id in ["struggle", "recharge"]
                ):
                    move_space = [9]
                actions = switch_space + move_space + tera_space
            actions = actions or [0]
            action_mask = [int(i in actions) for i in range(ACT_SIZE)]
            return action_mask

        # Stack for both active Pokémon (positions 0 and 1)
        mask0 = single_action_mask(battle, 0)
        mask1 = single_action_mask(battle, 1)
        # Return as a torch tensor of shape (2, ACT_SIZE)
        return torch.tensor([mask0, mask1], dtype=torch.uint8)
