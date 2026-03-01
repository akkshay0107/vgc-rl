import torch
import torch.nn as nn
import torch.nn.init as init

from observation_builder import OBS_DIM


class CLSReducer(nn.Module):
    CLS, FIELD, ALLY, FOE = 0, 1, 2, 3
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
                f"This CLSReducer assumes seq_len={OBS_DIM[0]} (field + extra info + 12*(textA,textB,num))."
            )

        self.seq_len = seq_len
        self.feat_dim = feat_dim
        self.d_model = d_model

        self.in_proj = nn.Linear(feat_dim, d_model)

        self.cls_base = nn.Parameter(torch.rand(1, 1, d_model) * self.d_model**-0.5)
        self.cls_conditioner = nn.Linear(d_model, d_model)

        self.type_emb = nn.Embedding(4, d_model)
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
            enc_layer, num_layers=nlayer
        )

        type_ids, part_ids = self._build_ids()
        self.register_buffer("type_ids", type_ids, persistent=False)
        self.register_buffer("part_ids", part_ids, persistent=False)

        self._init_weights()

    def _build_ids(self):
        type_ids = torch.empty(self.seq_len + 1, dtype=torch.long)
        type_ids[0] = self.CLS
        type_ids[1] = type_ids[2] = self.FIELD
        # Ally pokemon text (6 pokemon * 2 parts)
        type_ids[3:15] = self.ALLY
        # Foe pokemon text (6 pokemon * 2 parts)
        type_ids[15:27] = self.FOE
        # Ally pokemon numerical (6 pokemon * 1 part)
        type_ids[27:33] = self.ALLY
        # Foe pokemon numerical (6 pokemon * 1 part)
        type_ids[33:39] = self.FOE

        # first 3 tokens do not have a part embedding (they are all of one part each)
        part_ids = torch.cat(
            [
                torch.tensor([self.TEXT_A, self.TEXT_B], dtype=torch.long).repeat(12),
                torch.tensor([self.NUM], dtype=torch.long).repeat(12),
            ]
        )

        return type_ids, part_ids

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
        x = torch.cat([cls, x], dim=1)

        type_e = self.type_emb(self.type_ids).unsqueeze(0).expand(B, -1, -1)
        part_e = self.part_emb(self.part_ids).unsqueeze(0).expand(B, -1, -1)
        x += type_e
        x[:, 3:, :] += part_e

        x = self.encoder(x)
        return x[:, 0]
