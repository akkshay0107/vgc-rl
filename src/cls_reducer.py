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
    UNKNOWN = 10
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
            raise ValueError(
                f"This CLSReducer assumes seq_len={OBS_DIM[0]} (field + info + 12*text + 12*num)."
            )

        self.seq_len = seq_len
        self.feat_dim = feat_dim
        self.d_model = d_model

        self.in_proj = nn.Linear(feat_dim, d_model)

        self.cls_base = nn.Parameter(torch.rand(1, 1, d_model) * self.d_model**-0.5)
        self.cls_conditioner = nn.Linear(d_model, d_model)

        self.type_emb = nn.Embedding(11, d_model)
        self.part_emb = nn.Embedding(3, d_model)

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
            enable_nested_tensor=False,  # annoying warning
        )

        # part_ids does not apply to cls and field text
        # Transformer indices (after cat cls at dim 0):
        # 0: CLS
        # 1: Field Text
        # 2: Info Text
        # 3-14: P1 Text (12)
        # 15-26: Opp Text (12)
        # 27-32: P1 Num (6)
        # 33-38: Opp Num (6)
        # 39: Field Num (1)

        part_ids = torch.cat(
            [
                torch.tensor([self.TEXT_A, self.TEXT_B], dtype=torch.long).repeat(
                    12
                ),  # P1 then Opp text
                torch.tensor([self.NUM], dtype=torch.long).repeat(13),  # P1+Opp+Field Nums
            ]
        )
        self.register_buffer("part_ids", part_ids, persistent=False)

        type_ids = torch.zeros(40, dtype=torch.long)
        type_ids[0] = self.CLS
        type_ids[1:3] = self.FIELD

        # P1 Texts (3-14)
        type_ids[3:5] = self.TEAM_SLOT_0
        type_ids[5:7] = self.TEAM_SLOT_1
        type_ids[7:11] = self.TEAM_BENCH
        type_ids[11:15] = self.TEAM_DROPPED

        # Opp Texts (15-26)
        type_ids[15:17] = self.OPP_SLOT_0
        type_ids[17:19] = self.OPP_SLOT_1
        type_ids[19:23] = self.OPP_BENCH
        type_ids[23:27] = self.OPP_DROPPED

        # P1 Nums (27-32)
        type_ids[27] = self.TEAM_SLOT_0
        type_ids[28] = self.TEAM_SLOT_1
        type_ids[29:31] = self.TEAM_BENCH
        type_ids[31:33] = self.TEAM_DROPPED

        # Opp Nums (33-38)
        type_ids[33] = self.OPP_SLOT_0
        type_ids[34] = self.OPP_SLOT_1
        type_ids[35:37] = self.OPP_BENCH
        type_ids[37:39] = self.OPP_DROPPED

        # Field Nums (39)
        type_ids[39] = self.FIELD
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

        for p in self.encoder.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
            else:
                nn.init.zeros_(p)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        if obs.dim() == 2:
            obs = obs.unsqueeze(0)
        B, S, F = obs.shape
        if S != self.seq_len or F != self.feat_dim:
            raise ValueError(f"Got shape ({S}, {F}). Expected ({self.seq_len}, {self.feat_dim})")

        x = self.in_proj(obs)

        global_ctx = x.mean(dim=1, keepdim=True)
        cls = self.cls_base.expand(B, -1, -1) + self.cls_conditioner(global_ctx)
        x = torch.cat([cls, x], dim=1)  # (B, 40, D)

        type_e = self.type_emb(self.type_ids).unsqueeze(0).expand(B, -1, -1)
        part_e = self.part_emb(self.part_ids).unsqueeze(0).expand(B, -1, -1)

        x += type_e
        x[:, 3:, :] += part_e

        x = self.encoder(x)
        return x[:, 0]
