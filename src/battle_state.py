import torch


class BattleState:
    """
    BattleState is a wrapper over a 2x5x22 tensor storing the state of the pokemons
    of the two players.

    2 channels - one for each player

    5 rows - 1st row for field conditions on respective players side of the field
                Next 4 rows for the state of each pokemon selected to play

    22 cols for pokemons
        Col 0 - pokemonID
        Col 1 - primary typing
        Col 2 - secondary typing
        Col 3 - tera burnt or not
        Col 4 - item held / consumed or knocked off
        Col 5 - non volatile status condition
        Col [6-8] - one hot encoding of taunt, encore, confusion status respectively (1 if active, 0 otherwise)
        Col 9 - current HP stat
        Col [10-14] - base stats (excluding HP)
        Col [15-21] - stat stages (all 6 base stars excluding HP + accuracy and evasion)

    cols for field effects (first 5 are global, last 2 are local, value of 0 means inactive)
        Col 0 - trick room turns remaining
        Col 1 - grassy terrain turns remaining
        Col 2 - psy terrain turns remaining
        Col 3 - sun turns remaining
        Col 4 - rain turns remaining
        Col 5 - tailwind turns remaining
        Col 6 - aurora veil turns remaining
        Col [7-21] - padding using 0 (future space to expand ??)

    Additional considerations for the future
        Store history of moves for moves that are not independent (Protect, Stomping Tantrum)
        Store Rage Fist stacks for annihilape ??
    """

    def __init__(self):
        # defaults to wrapping over a bunch of zeros
        return torch.zeros(2, 5, 22)
