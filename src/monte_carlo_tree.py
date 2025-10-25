import rps_state
import math
import random

class MCTNode:
    def __init__(self, parent=None, rps=rps_state.RPSState()) -> None:
        self.parent = parent
        self.childs = []
        self.visits = 0.0
        self.value = 0.0
        self.state = rps.state
        self.RPS = rps
        self.moves = list(self.RPS.moves())
        self.action = -1

    def __iter__(self):
        # Iterator for the node, just iterates over all children
        return iter(self.childs)  

    def calcUCB1(self, N, c):
        X_i = self.value / self.visits
        n_i = self.visits
        return X_i + c * math.sqrt(math.log(N)/n_i)

    def expand(self):
        action = self.moves.pop()
        nextRPS = self.RPS.next_state(action)
        child = MCTNode(self, nextRPS)
        child.action = action
        self.childs.append(child)
        return child

    def rollout(self):
        if self.state:
            curRPS = self.RPS

            while True:
                if curRPS.is_end_state():
                    return curRPS.get_result()

                action = random.choice(list(curRPS.moves()))
                curRPS = curRPS.next_state(action)
    
    
    def backPropagrate(self, result):
        self.visits += 1
        self.value += result
        if self.parent:
            self.parent.backPropagrate(result)

    def best_child(self, c=math.sqrt(2)):
        N = self.visits
        bestChild = self.childs[0]
        bestUCB1 = -1e18
        for child in self.childs:
            if child.calcUCB1(N, c) > bestUCB1:
                bestChild = child
                bestUCB1 = child.calcUCB1(N, c)
        return bestChild

    def is_fully_expanded(self):
        return len(self.moves) == 0

    def is_terminal(self):
        return self.RPS.is_end_state()

# TODO: implement the UCB1 algorithm for the MCTNode class
# Complete the implementation of the MCTNode class
# Create a dummy state and benchmark performance for the MCTNode

'''
Upper Confidence Bound Formula:

UCB1(i) = X_i + c * sqrt(ln(N)/n_i)
Where Xi is the average reward of node i
c is the exploration parameter (usually sqrt(2))
N is the total number of visits
n_i is the number of visits to node i
'''

def MCTSearch(iters = 500, root=MCTNode()): 
    for _ in range(iters):
        curNode = root

        while not curNode.is_terminal() and curNode.is_fully_expanded():
            curNode = curNode.best_child()
        
        if not curNode.is_terminal():
            curNode = curNode.expand()
        
        result = curNode.rollout()

        curNode.backPropagrate(result)
    
    return root

def human_vs_bot_game():
    values = {0: "Rock", 1: "Paper", 2: "Scissors"}
    
    print("Play a game? (yes/no)")
    ans = input().strip().lower()
    
    if ans not in ['y', 'yes']:
        return
    
    initial_state = rps_state.RPSState()
    root = MCTNode(rps=initial_state)
    
    root = MCTSearch(2000, root)
    
    current_state = initial_state
    current_node = root
    
    for turn in range(10):
        print(f"\n=== Turn {turn + 1}/10 ===")
        print(f"Current scores - Bot: {current_state.state[0][3]}, Player: {current_state.state[1][3]}")
        print(f"Bot's remaining moves - Rock: {current_state.state[0][0]}, Paper: {current_state.state[0][1]}, Scissors: {current_state.state[0][2]}")
        print(f"Your remaining moves - Rock: {current_state.state[1][0]}, Paper: {current_state.state[1][1]}, Scissors: {current_state.state[1][2]}")
        
        bot_move = select_bot_move(current_node)
        
        player_move = get_human_move(current_state.state[1], turn)
        
        action = (bot_move, player_move)
        print(f"Bot's {values[bot_move]} vs Player's {values[player_move]}")
        
        next_node = find_child_node(current_node, action)
        
        if next_node is None:
            new_state = current_state.next_state(action)
            next_node = MCTNode(parent=current_node, rps=new_state)
            next_node.action = action
            current_node.childs.append(next_node)
        
        current_node = next_node
        current_state = next_node.RPS
        
        if turn < 9:  
            current_node = MCTSearch(500, current_node)
        
        if current_state.is_end_state():
            break

    final_score_bot = current_state.state[0][3]
    final_score_player = current_state.state[1][3]
    
    print(f"\n=== Game Over ===")
    print(f"Final Score - Bot: {final_score_bot}, Player: {final_score_player}")
    if final_score_bot > final_score_player:
        print("Bot wins!")
    elif final_score_player > final_score_bot:
        print("You win!")
    else:
        print("It's a tie!")

def select_bot_move(current_node):
    if not current_node.childs:
        return random.choice([0, 1, 2])
    
    bot_move_values = {}
    
    for child in current_node.childs:
        bot_move = child.action[0]
        
        if bot_move not in bot_move_values:
            bot_move_values[bot_move] = []
        
        if child.visits > 0:
            value = child.value / child.visits
            bot_move_values[bot_move].append(value)
    
    best_move = None
    best_min_value = -float('inf')
    
    for bot_move, values in bot_move_values.items():
        if values: 
            min_value = min(values) 
            if min_value > best_min_value:
                best_min_value = min_value
                best_move = bot_move
    
    return best_move if best_move is not None else random.choice([0, 1, 2])

def get_human_move(player_resources, turn):
    remaining_turns = 10 - turn
    total_required = sum(player_resources[:3])
    unforced_moves = remaining_turns - total_required
    
    while True:
        print("Your move? Rock(0), Paper(1), Scissors(2)")
        try:
            move = int(input().strip())
            if move not in [0, 1, 2]:
                print("Invalid move. Choose 0, 1, or 2.")
                continue
            
            if player_resources[move] > 0:
                return move
            elif unforced_moves > 0:
                return move
            else:
                print(f"You must use your required moves first!")
                
        except ValueError:
            print("Please enter a number (0, 1, or 2).")

def find_child_node(parent_node, action):
    for child in parent_node.childs:
        if child.action == action:
            return child
    return None

if __name__ == "__main__":
    human_vs_bot_game()