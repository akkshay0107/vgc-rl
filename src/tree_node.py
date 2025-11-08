import math
import random
from battle_state import BattleState

EXPLORATION_CONSTANT = math.sqrt(2)

class TreeNode:
    def __init__(
        self,
        parent=None,
    ) -> None:
        self.parent = parent
        self.childs = []
        self.visits = 0.0
        self.value = 0.0
        self.battle_state = BattleState()
        self.action = -1

    def ucb1(self, N, c):
        X_i = self.value / self.visits
        n_i = self.visits
        return X_i + c * math.sqrt(math.log(N) / n_i)

    # TODO: implement expand
    # Requires modification to BattleState to continue
    def expand(self):
        pass
        # action = self.moves.pop()
        # nextRPS = self.RPS.next_state(action)
        # child = MCTNode(self, nextRPS)
        # child.action = action
        # self.childs.append(child)
        # return child

    # TODO: implement rollout
    # Make sure to rollout to a fixed depth below current node
    # and then use eval from NN to choose the best node
    # Do not rollout entire games it is too slow
    def rollout(self):
        pass
        # if self.state:
        #     curRPS = self.RPS

        #     while True:
        #         if curRPS.is_end_state():
        #             return curRPS.get_result()

        #         action = random.choice(list(curRPS.moves()))
        #         curRPS = curRPS.next_state(action)

    def back_propagate(self, result):
        self.visits += 1
        self.value += result
        if self.parent:
            self.parent.back_propagate(result)

    def best_child(self):
        return max(self.childs, key=lambda c: c.ucb1(EXPLORATION_CONSTANT))

    # TODO: implement is_fully_expanded and is_terminal
    def is_fully_expanded(self):
        pass

    def is_terminal(self):
        pass


if __name__ == "__main_":
    # run tests here
    pass
