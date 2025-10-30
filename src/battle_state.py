import numpy as np
import torch
import poke_env.data.gen_data as gen_data

class battle_state:
    DEX = gen_data.GenData(9)

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
            1: Pokemon 1
                0: Name
                1: EVs
                2: IVs
                3: Nature
                4: Volatile Statuses
                5: Non-volatile Statuses
                6: Stat Stages
                7: Moves
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

    
    def calculate_stats(self, base: dict, iv: dict, ev: dict, nature: dict) -> dict:
        stats = {}
        for k in base.keys():
            if k == 'hp':
                stats[k] = np.floor(((2 * base[k] + iv[k] + (ev[k] // 4)) * 50) // 100 + 50 + 10)
            else:
                stats[k] = np.floor((((2 * base[k] + iv[k] + (ev[k] // 4)) * 50) // 100 + 5) * nature[k])
        return stats