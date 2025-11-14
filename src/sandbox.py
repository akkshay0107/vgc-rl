from battle_state import BattleState
from lookups import MOVE_LISTS, MOVES
import copy
import torch


class Sandbox:
    def __init__(self, battle_state: BattleState):
        self.battle_state = battle_state

    def _get_pokemon_moves(self, pid, slice):
        move_list = []

        if slice[7] > 0:
            # encored into last move
            last_move_idx = int(slice[29].item())
            if slice[23 + last_move_idx - 1] > 0:
                move_names = MOVE_LISTS[pid].split(",")
                name = move_names[last_move_idx - 1]
                move = MOVES[name]
                if move["category"] == "status" and slice[6] > 0:
                    pass  # can't use status move if taunted
                else:
                    if move["target"] == "normal":
                        for to in range(1, 4):
                            modified_move = copy.deepcopy(move)
                            modified_move["to"] = to
                            move_list.append(modified_move)
                    else:
                        move_list.append(move)
        elif pid > 0 and slice[9] > 0:
            move_names = MOVE_LISTS[pid].split(",")
            for i, name in enumerate(move_names):
                if slice[23 + i] == 0:
                    continue  # pp out
                move = MOVES[name]
                if move["category"] == "status" and slice[6] > 0:
                    continue  # can't use status move if taunted
                if move["target"] == "normal":
                    for to in range(1, 4):
                        modified_move = copy.deepcopy(move)
                        modified_move["to"] = to
                        move_list.append(modified_move)
                else:
                    move_list.append(move)
        else:
            move_list.append(None)
        return move_list

    def _get_moves_for_player(self, player):
        """
        Returns all possible moves for a specific player as a generator
        Generator elements are tuples with 2 elements.
        Element is a dict if it represents a move
        Otherwise element is a switch to pokemon of specific row
        None if less than one active pokemon
        """
        slice = self.battle_state.state[player]
        switches = [i for i in range(3, 5) if slice[i, 9] > 0] # Benched pokemon are rows 3 & 4, right?
        tera_burnt = int(slice[0, 9].item())

        pid1 = int(slice[1, 0].item())
        pid2 = int(slice[2, 0].item())

        move_list1 = self._get_pokemon_moves(pid1, slice[1])
        move_list2 = self._get_pokemon_moves(pid2, slice[2])

        struggle = {
            "base_power": 50,
            "type": "normal",
            "category": "physical",
            "priority": 0,
            "accuracy": True,
            "target": "normal",
            "pp": None,
            "flags": {"contact": 1, "protect": 1, "mirror": 1},
            "recoil": "25% of max hp",
        }
        if len(move_list1) == 0:
            move_list1.append(struggle)

        if len(move_list2) == 0:
            move_list2.append(struggle)

        # regular moves with targeting
        for move1 in move_list1:
            for move2 in move_list2:
                yield (move1, move2)

        # regular one switch one move (no tera)
        for switch in switches:
            for move1 in move_list1:
                yield (move1, switch)

            for move2 in move_list2:
                yield (switch, move2)

        # double switch
        if len(switches) == 2:
            yield (switches[0], switches[1])
            yield (switches[1], switches[0])

        if tera_burnt == 0:
            # tera moves with targeting
            for move1 in move_list1:
                tmove1 = copy.deepcopy(move1)
                tmove1["tera"] = True
                for move2 in move_list2:
                    tmove2 = copy.deepcopy(move2)
                    tmove2["tera"] = True
                    yield (tmove1, move2)
                    yield (move1, tmove2)

            # one switch with tera
            for switch in switches:
                for move1 in move_list1:
                    tmove1 = copy.deepcopy(move1)
                    tmove1["tera"] = True
                    yield (tmove1, switch)

                for move2 in move_list2:
                    tmove2 = copy.deepcopy(move2)
                    tmove2["tera"] = True
                    yield (switch, tmove2)

    def get_moves(self):
        for move_pair1 in self._get_moves_for_player(player=0):
            for move_pair2 in self._get_moves_for_player(player=1):
                yield (move_pair1, move_pair2)

    def simulate_turn(self, action: tuple[tuple, tuple]):
        pass


if __name__ == "__main__":
    sample_battle_state = BattleState()
    sample_battle_state.state = torch.Tensor()
    box = Sandbox(sample_battle_state)
    print(list(box.get_moves()))
