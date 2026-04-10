import torch
import torch.nn as nn
import torch.nn.init as init

from lookups import OBS_DIM


class CLSReducer(nn.Module):
    CLS = 0
    FIELD = 1
    TEAM_SLOT_0 = 2
    TEAM_SLOT_1 = 3
    TEAM_BENCH = 4
    TEAM_DROPPED = 5
    OPP_SLOT_0 = 6
    OPP_SLOT_1 = 7
    OPP_BENCH = 8
    OPP_DROPPED = 9
    H1 = 10
    H2 = 11
    H3 = 12
    HG = 13
    UNKNOWN = 14
    TEXT_A, TEXT_B, NUM = 0, 1, 2

    def __init__(
        self,
        seq_len: int,
        feat_dim: int,
        d_model: int = 768,
        nhead: int = 8,
        nlayer: int = 2,
        dim_feedforward: int = 3072,
        dropout: float = 0.1,
    ):
        super().__init__()
        if seq_len != OBS_DIM[0]:
            raise ValueError(f"This CLSReducer assumes seq_len={OBS_DIM[0]}.")

        self.seq_len = seq_len
        self.feat_dim = feat_dim
        self.d_model = d_model

        self.in_proj = nn.Linear(feat_dim, d_model)

        self.cls_base = nn.Parameter(torch.rand(1, 1, d_model) * self.d_model**-0.5)
        self.cls_conditioner = nn.Linear(d_model, d_model)

        self.type_emb = nn.Embedding(15, d_model)
        self.part_emb = nn.Embedding(3, d_model)

        # GRU for history update: HG_curr = GRU([CLS_prev, H1_curr], HG_prev)
        self.history_gru = nn.GRUCell(d_model * 2, d_model)

        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
            norm_first=True,
            activation="gelu",
        )
        self.encoder = nn.TransformerEncoder(
            enc_layer,
            num_layers=nlayer,
            enable_nested_tensor=False,
        )

        # 0: CLS
        # 1: Field Text
        # 2: Info Text
        # 3: H1 Text
        # 4: H2 Text
        # 5: H3 Text
        # 6-17: P1 Text (12)
        # 18-29: Opp Text (12)
        # 30-35: P1 Num (6)
        # 36-41: Opp Num (6)
        # 42: Field Num (1)
        # 43: HG

        part_ids = torch.cat(
            [
                torch.tensor([self.TEXT_A], dtype=torch.long).repeat(3),  # H1, H2, H3
                torch.tensor([self.TEXT_A, self.TEXT_B], dtype=torch.long).repeat(
                    12
                ),  # P1 then Opp text
                torch.tensor([self.NUM], dtype=torch.long).repeat(13),  # P1+Opp+Field Nums
            ]
        )
        self.register_buffer("part_ids", part_ids, persistent=False)

        type_ids = torch.zeros(44, dtype=torch.long)
        type_ids[0] = self.CLS
        type_ids[1:3] = self.FIELD
        type_ids[3] = self.H1
        type_ids[4] = self.H2
        type_ids[5] = self.H3

        # P1 Texts (6-17)
        type_ids[6:8] = self.TEAM_SLOT_0
        type_ids[8:10] = self.TEAM_SLOT_1
        type_ids[10:14] = self.TEAM_BENCH
        type_ids[14:18] = self.TEAM_DROPPED

        # Opp Texts (18-29)
        type_ids[18:20] = self.OPP_SLOT_0
        type_ids[20:22] = self.OPP_SLOT_1
        type_ids[22:26] = self.OPP_BENCH
        type_ids[26:30] = self.OPP_DROPPED

        # P1 Nums (30-35)
        type_ids[30] = self.TEAM_SLOT_0
        type_ids[31] = self.TEAM_SLOT_1
        type_ids[32:34] = self.TEAM_BENCH
        type_ids[34:36] = self.TEAM_DROPPED

        # Opp Nums (36-41)
        type_ids[36] = self.OPP_SLOT_0
        type_ids[37] = self.OPP_SLOT_1
        type_ids[38:40] = self.OPP_BENCH
        type_ids[40:42] = self.OPP_DROPPED

        # Field Nums (42)
        type_ids[42] = self.FIELD
        # HG (43)
        type_ids[43] = self.HG
        self.register_buffer("type_ids", type_ids, persistent=False)

        self._init_weights()

    @torch.no_grad()
    def _init_weights(self):
        init.orthogonal_(self.in_proj.weight, gain=1.0)
        init.zeros_(self.in_proj.bias)

        init.orthogonal_(self.cls_conditioner.weight, gain=1.0)
        init.zeros_(self.cls_conditioner.bias)

        emb_gain = self.d_model**-0.5
        init.normal_(self.cls_base, std=emb_gain)
        init.normal_(self.type_emb.weight, std=emb_gain)
        init.normal_(self.part_emb.weight, std=emb_gain)

    def forward(
        self, obs: torch.Tensor, state: tuple[torch.Tensor, torch.Tensor] | None = None
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        if obs.dim() == 2:
            obs = obs.unsqueeze(0)
        B, S, F = obs.shape
        if S != self.seq_len or F != self.feat_dim:
            raise ValueError(f"Got shape ({S}, {F}). Expected ({self.seq_len}, {self.feat_dim})")

        x = self.in_proj(obs)

        if state is None:
            hg_prev = torch.zeros(B, self.d_model, device=obs.device)
            cls_prev = torch.zeros(B, self.d_model, device=obs.device)
        else:
            hg_prev, cls_prev = state

        # h1_curr is at index 2 of obs
        h1_curr = x[:, 2]
        hg_curr = self.history_gru(torch.cat([cls_prev, h1_curr], dim=-1), hg_prev)

        global_ctx = x.mean(dim=1, keepdim=True)
        cls = self.cls_base.expand(B, -1, -1) + self.cls_conditioner(global_ctx)

        # x is (B, 42, D)
        # Add HG at the end
        x = torch.cat([cls, x, hg_curr.unsqueeze(1)], dim=1)  # (B, 44, D)

        type_e = self.type_emb(self.type_ids).unsqueeze(0).expand(B, -1, -1)
        part_e = self.part_emb(self.part_ids).unsqueeze(0).expand(B, -1, -1)

        x += type_e
        # part_ids apply to index 3 to 42 (H1-H3, P1-Opp texts, P1-Opp-Field nums)
        # transformer indices:
        # 0: CLS
        # 1: Field Text (no part_id)
        # 2: Info Text (no part_id)
        # 3-42: part_ids apply
        # 43: HG (no part_id)
        x[:, 3:43, :] += part_e

        x = self.encoder(x)
        cls_out = x[:, 0]
        return cls_out, (hg_curr, cls_out)
