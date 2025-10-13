class RPSState:
    # class is a wrapper over a state that looks like
    # [[reqd_rock1, reqd_paper1, reqd_scissor1, score1], [reqd_rock2, reqd_paper2, reqd_scissor2, score2]]
    # Assumed that MCTS always plays as player 1 (shouldn't matter since game is symmetric)

    ACTIONS = [0,1,2] # rock = 0, paper = 1, scissor = 2
    MAX_TURNS = 10

    def __init__(self, state = None, turn = 0) -> None:
        if state is None:
            self.state = [[2,2,2,0], [2,2,2,0]]
        else:
            # deep copy of state
            self.state = [list(player_state) for player_state in state]

        self.turn = turn

    def moves(self):
        # Generator that yields all legal moves from current state
        remaining_turns = self.MAX_TURNS - self.turn

        required_p1 = self.state[0][:3]
        total_required_p1 = sum(required_p1)

        required_p2 = self.state[1][:3]
        total_required_p2 = sum(required_p2)

        legal_moves_p1 = []
        for m in self.ACTIONS:
            if required_p1[m] > 0 or (required_p1[m] == 0 and total_required_p1 < remaining_turns):
                legal_moves_p1.append(m)

        legal_moves_p2 = []
        for m in self.ACTIONS:
            if required_p2[m] > 0 or (required_p2[m] == 0 and total_required_p2 < remaining_turns):
                legal_moves_p2.append(m)

        for move_p1 in legal_moves_p1:
            for move_p2 in legal_moves_p2:
                yield (move_p1, move_p2)

    @staticmethod
    def _turn_result(action):
        # action is a tuple (player_1_move, player_2_move)
        # returns who won the turn
        if action[0] == action[1]:
            return 0

        if action[0] == 0 and action[1] == 2:
            return 1

        if action[0] == 2 and action[1] == 0:
            return -1

        if action[1] - action[0] == 1:
            return -1

        return 1

    def next_state(self, action):
         # action is a tuple (player_1_move, player_2_move)
        res = self._turn_result(action)
        new_state = [list(p) for p in self.state]  # Deep copy

        if res == 1:
            new_state[0][3] += 1
        elif res == -1:
            new_state[1][3] += 1

        if new_state[0][action[0]] != 0:
            new_state[0][action[0]] -= 1

        if new_state[1][action[1]] != 0:
            new_state[1][action[1]] -= 1

        return RPSState(new_state, self.turn + 1)

    def is_end_state(self):
        return self.turn == self.MAX_TURNS

    def get_result(self):
            score1 = self.state[0][3]
            score2 = self.state[1][3]
            if score1 > score2:
                return 1
            elif score2 > score1:
                return -1
            return 0

    def __str__(self):
        # for debugging
        return (f"Turn: {self.turn}\n"
                f"Player 1 [Rock left, Paper left, Scissors left, Score]: {self.state[0]}\n"
                f"Player 2 [Rock left, Paper left, Scissors left, Score]: {self.state[1]}")
