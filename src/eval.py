from battle_state import BattleState
import torch


def pseudo_eval(state: BattleState) -> float:
    # returns E[sum of health of player's pokemon - sum of health of opponent's pokemon]
    player_health_sum = torch.sum(state.state[0, 1:5, 9])
    opponent_health_sum = torch.sum(state.state[1, 1:5, 9])
    return (player_health_sum - opponent_health_sum).item()
