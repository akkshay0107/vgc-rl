import torch


class BattleState:
    """
    BattleState is a wrapper over a 2x5x22 tensor storing the state of the pokemons
    of the two players.

    2 channels - one for each player

    5 rows - 1st row for field conditions on respective players side of the field
                Next 4 rows for the state of each pokemon selected to play

    24 cols for pokemons
        Col 0 - pokemonID
        Col 1 - primary typing
        Col 2 - secondary typing
        Col 3 - tera type (0 if tera not used else tera type)
        Col 4 - item held / consumed or knocked off
        Col 5 - non volatile status condition
        Col [6-8] - one hot encoding of taunt, encore, confusion status respectively (1 if active, 0 otherwise)
        Col 9 - current HP stat
        Col [10-15] - base stats
        Col [16-22] - stat stages (all 6 base stars excluding HP + accuracy and evasion)
        Col [23-26] - pp for each of the 4 moves
        Col 27 - protect counter
        Col 28 - boolean that denotes whether the last turn missed or not (for stomping tantrum)

    cols for field effects (first 5 are global, 6 and 7th are local, value of 0 means inactive)
        also stores some team level counters
        Col 0 - trick room turns remaining
        Col 1 - grassy terrain turns remaining
        Col 2 - psy terrain turns remaining
        Col 3 - sun turns remaining
        Col 4 - rain turns remaining
        Col 5 - tailwind turns remaining
        Col 6 - aurora veil turns remaining
        Col 7 - number of fainted pokemon in the team
        Col 8 - rage fist stacks (0 if no annihilape in the team)
        Col [9-28] - padding using 0 (future space to expand ??)
    """

    def __init__(self):
        self.state = torch.zeros((2, 5, 29))
        self.prob = 1.0

    def check_game_end(self) -> bool:
        player_loss = self.state[0, 0, 7].item() == 4
        opponent_loss = self.state[1, 0, 7].item() == 4
        return player_loss or opponent_loss
