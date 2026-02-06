import random
from dataclasses import dataclass

import torch
import torch.nn as nn
from poke_env.battle import AbstractBattle, DoubleBattle

from lookups import POKEMON


def moveset_key_from_pokemon(mon) -> str:

    move_ids = sorted(list(getattr(mon, "moves", {}).keys()))
    return ",".join(move_ids)


def opponent_team_ids(battle: DoubleBattle) -> torch.Tensor:

    ids: list[int] = []
    for mon in battle.opponent_team.values():
        key = moveset_key_from_pokemon(mon)
        ids.append(int(POKEMON.get(key, 0)))
    return torch.tensor(ids, dtype=torch.long)

def topk(logits: torch.Tensor, k: int) -> list[int]:

    k = min(int(k), int(logits.shape[0]))
    return torch.topk(logits, k=k, dim=-1).indices.tolist()


class TeamPreviewNet(nn.Module):

    def __init__(
        self,
        num_opponent_tokens: int | None = None,
        d_model: int = 128,
        mlp_hidden: int = 256,
    ) -> None:
        super().__init__()
        self.team_size = 6

        if num_opponent_tokens is None:
            # 0 is unknown if it somehow happens
            num_opponent_tokens = max(POKEMON.values()) + 1

        self.opp_embed = nn.Embedding(int(num_opponent_tokens), int(d_model))
        self.opp_encoder = nn.Sequential(
            nn.Linear(int(d_model), int(mlp_hidden)),
            nn.ReLU(),
            nn.Linear(int(mlp_hidden), int(d_model)),
            nn.ReLU(),
        )

        self.bring_head = nn.Linear(int(d_model), self.team_size)
        self.lead_head = nn.Linear(int(d_model), self.team_size)

    def forward(self, opp_ids: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if opp_ids.dim() != 2 or opp_ids.shape[1] != 6:
            raise ValueError(f"Expected opp_ids shape (B, 6), got {tuple(opp_ids.shape)}")

        embed = self.opp_embed(opp_ids)

        x = embed.mean(dim=1) 
        z = self.opp_encoder(x)  

        bring_logits = self.bring_head(z)  # (B, team_size)
        lead_logits = self.lead_head(z)  # (B, team_size)
        return bring_logits, lead_logits



class TeamPreviewHandler:
    """Handles team preview selection for the RL player."""

    def __init__(
        self,
        model: TeamPreviewNet | None = None,
        device: torch.device | None = None,
    ) -> None:
        self.model = model
        self.bring = 4
        self.lead = 2
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

        if self.model is not None:
            self.model.to(self.device)
            self.model.eval()

    def select_team(self, battle: AbstractBattle) -> str:
        assert isinstance(battle, DoubleBattle)

        if self.model is None:
            members = list(range(1, len(battle.team) + 1))
            random.shuffle(members)

            return "/team " + "".join([str(c) for c in members[:self.bring]])

        decision = self.decide(battle)
        # showdown expects 1-based party slots
        bring_1_based = [i + 1 for i in decision[0]]
        lead_1_based = [i + 1 for i in decision[1]]

        remaining = [i for i in bring_1_based if i not in set(lead_1_based)]
        ordered = lead_1_based + remaining
        return "/team " + "".join(str(i) for i in ordered)

    @torch.no_grad()
    def decide(self, battle: DoubleBattle) -> tuple[tuple[int, ...], tuple[int, ...]]:


        team_size = len(battle.team)
        if team_size <= 0:
            raise ValueError("team is empty")

        opp_ids = opponent_team_ids(battle).unsqueeze(0).to(self.device)
        bring_logits, lead_logits = self.model(opp_ids) # type: ignore
        bring_logits = bring_logits[0, :team_size].detach().cpu()
        lead_logits = lead_logits[0, :team_size].detach().cpu()

        bring_k = min(self.bring, team_size)
        lead_k = min(self.lead, bring_k)

        bring_idx = topk(bring_logits, bring_k)

        # Mask lead choice to only the brought mons
        mask = torch.full((team_size,), float("-inf"), dtype=lead_logits.dtype)
        for i in bring_idx:
            mask[i] = 0.0
        masked_lead_logits = lead_logits + mask

        lead_idx = topk(masked_lead_logits, lead_k)

        lead_set = [i for i in lead_idx if i in set(bring_idx)]
        if len(lead_set) < lead_k:
            for i in bring_idx:
                if i not in set(lead_set):
                    lead_set.append(i)
                if len(lead_set) >= lead_k:
                    break

        bring_tuple = tuple(int(i) for i in bring_idx[:bring_k])
        lead_tuple = tuple(int(i) for i in lead_set[:lead_k])
        return bring_tuple, lead_tuple
