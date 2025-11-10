import math
import random
from battle_state import BattleState
from eval import pseudo_eval
from sandbox import Sandbox

EXPLORATION_CONSTANT = math.sqrt(2)


class TreeNode:
    def __init__(self, parent=None, battle_state=BattleState()) -> None:
        self.parent = parent
        self.childs = []
        self.visits = 0
        self.value = 0.0
        self.battle_state = battle_state
        self.moves = Sandbox(self.battle_state).get_moves() # generator over moves
        self.action = -1
        self.exhausted = False  # Track if moves generator is exhausted

    def ucb1(self, N, c):
        if self.visits == 0:
            return float("inf")  # Encourage exploration of unvisited nodes
        return (self.value / self.visits) + c * math.sqrt(math.log(N) / self.visits)

    def expand(self):
        if self.exhausted:
            return None
        try:
            action = next(self.moves)
        except StopIteration:
            self.exhausted = True
            return None
        box = Sandbox(self.battle_state)
        next_state = box.simulate_turn(action)
        child = TreeNode(self, next_state)
        child.action = action
        self.childs.append(child)
        return child

    def rollout(self, max_depth=2):
        current_state = self.battle_state
        box = Sandbox(self.battle_state)
        for _ in range(max_depth):
            moves = list(box.get_moves())
            if not moves:
                break
            action = random.choice(moves)
            current_state = box.simulate_turn(action)

        return pseudo_eval(current_state)

    def back_propagate(self, result):
        self.visits += 1
        self.value += result
        if self.parent:
            self.parent.back_propagate(result)

    def best_child(self):
        total_visits = self.visits if not self.parent else self.parent.visits
        return max(self.childs, key=lambda c: c.ucb1(total_visits, EXPLORATION_CONSTANT))

    def is_terminal(self):
        return self.battle_state.check_game_end()


if __name__ == "__main_":
    # run tests here
    pass
