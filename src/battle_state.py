import torch
import poke_env.data.gen_data as gen_data


# TODO: redesign the state to have two channels each for one player's state
# mold the global field state into local field states in each players side
class BattleState:
    DEX = gen_data.GenData(9)
    ABILITIES = {0:0, 1:1, 3:'H', 4:'S'}
    VOLATILE_STATUS = ['none', 'encore', 'taunt', 'confusion']
    NON_VOLATILE_STATUS = ['none', 'par', 'brn', 'slp', 'frz', 'psn']
    STAT_STAGES = [0.25, 0.285, 0.33, 0.4, 0.5, 0.66, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0]

    def __init__(self, p1_state, p2_state, global_field_state=torch.zeros(5)):
        """
        global_field_state: Tensor for global conditions, all stored as [turns passed] with 0 marking inactive
            0: Sun
            1: Rain
            2: Grassy Terrain
            3: Psychic Terrain
            4: Trick Room
        p1_state: Tensor for player 1's state
            0: Self Field Conditions (same as global)
                0: Veil
                1: Tailwind
                2: Friend Guard (always active with 2 pokemon)
            1: Pokemon 1
                0: Name
                1: Stats
                    0: HP
                    1: Atk
                    2: Def
                    3: Spa
                    4: Spd
                    5: Spe
                2: Ability Number (refer to ABILITIES)
                3: Volatile Statuses (refer to VOLATILE_STATUS)
                4: Non-volatile Statuses (refer to NON_VOLATILE_STATUS)
                5: Stat Stages
                    0: Atk
                    1: Def
                    2: Spa
                    3: Spd
                    4: Spe
                6: Moves
                7: Tera [type, active=1 else 0]
                8: Current HP%
            - Below all same as Pokemon 1
            2: Pokemon 2
            Backup Space:
            3: Pokemon 3
            4: Pokemon 4
        p2_state: Same as p1_state
        """
        self.state = torch.tensor([
            global_field_state,
            p1_state,
            p2_state
        ])
