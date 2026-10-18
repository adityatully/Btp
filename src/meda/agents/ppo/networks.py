"""Set encoder, candidate scorer and value head (SPEC 11.4).

The network only scores: every candidate row already carries the exact result of its action
(the afterstate features of `envs.encoding`), so nothing here computes a ratio (rule 1).

1. Each source row and each target row is embedded by its own MLP; a learned global token
   plus a projection of `globals` forms one more token.
2. A Transformer encoder runs over [global; sources; targets] with **no positional encoding**
   and key-padding masks, so row order carries no meaning and padded rows have no influence.
3. Each candidate gathers, through `cand_index`, the embedding of its first source and of its
   second operand: the second source (Mix), the target (Output) or a learned blank (Discard).
   These, the candidate's own features and the global token go through an MLP to one logit.
4. The value head reads the global token.

Input widths come from `EnvSizes`, so the network follows the observation layout instead of
hard-coding it.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor, nn

from meda.envs.encoding import OP_MIX, OP_OUTPUT, EnvSizes

TensorObs = dict[str, Tensor]


@dataclass(frozen=True, slots=True)
class NetConfig:
    d_model: int = 128
    n_layers: int = 3
    n_heads: int = 4


def _mlp(d_in: int, d_hidden: int, d_out: int) -> nn.Sequential:
    return nn.Sequential(nn.Linear(d_in, d_hidden), nn.ReLU(), nn.Linear(d_hidden, d_out))


class SetEncoderNet(nn.Module):
    def __init__(self, sizes: EnvSizes, cfg: NetConfig) -> None:
        super().__init__()
        d = cfg.d_model
        self.sizes = sizes
        self.cfg = cfg
        self.source_mlp = _mlp(sizes.d_src, d, d)
        self.target_mlp = _mlp(sizes.d_tgt, d, d)
        self.global_token = nn.Parameter(torch.zeros(d))
        self.global_proj = nn.Linear(EnvSizes.D_GLOB, d)
        layer = nn.TransformerEncoderLayer(
            d, cfg.n_heads, dim_feedforward=4 * d, dropout=0.0, batch_first=True, norm_first=True
        )
        self.encoder = nn.TransformerEncoder(
            layer, cfg.n_layers, norm=nn.LayerNorm(d), enable_nested_tensor=False
        )
        self.discard_blank = nn.Parameter(torch.zeros(d))
        self.scorer = nn.Sequential(
            nn.Linear(sizes.d_cand + 3 * d, d),
            nn.ReLU(),
            nn.Linear(d, d),
            nn.ReLU(),
            nn.Linear(d, 1),
        )
        self.value_head = _mlp(d, d, 1)

    def encode(self, obs: TensorObs) -> tuple[Tensor, Tensor, Tensor]:
        """Contextual embeddings: global token [B, d], sources [B, S, d], targets [B, T, d]."""
        src = self.source_mlp(obs["sources"])
        tgt = self.target_mlp(obs["targets"])
        glob = (self.global_token + self.global_proj(obs["globals"])).unsqueeze(1)
        tokens = torch.cat([glob, src, tgt], dim=1)
        batch = tokens.shape[0]
        pad = torch.cat(
            [
                torch.zeros(batch, 1, dtype=torch.bool, device=tokens.device),
                obs["source_mask"] == 0,
                obs["target_mask"] == 0,
            ],
            dim=1,
        )
        h = self.encoder(tokens, src_key_padding_mask=pad)
        s = src.shape[1]
        return h[:, 0], h[:, 1 : 1 + s], h[:, 1 + s :]

    def forward(self, obs: TensorObs) -> tuple[Tensor, Tensor]:
        """Raw candidate logits [B, L] (unmasked) and values [B]."""
        g, src, tgt = self.encode(obs)
        index = obs["cand_index"].long()
        op, ia, ib = index[..., 0], index[..., 1].clamp(min=0), index[..., 2].clamp(min=0)
        emb_a = _gather_rows(src, ia)
        emb_src_b = _gather_rows(src, ib.clamp(max=src.shape[1] - 1))
        emb_tgt = _gather_rows(tgt, ib.clamp(max=tgt.shape[1] - 1))
        blank = self.discard_blank.expand_as(emb_a)
        emb_b = torch.where(
            (op == OP_MIX).unsqueeze(-1),
            emb_src_b,
            torch.where((op == OP_OUTPUT).unsqueeze(-1), emb_tgt, blank),
        )
        g_rep = g.unsqueeze(1).expand_as(emb_a)
        logits = self.scorer(torch.cat([obs["candidates"], emb_a, emb_b, g_rep], dim=-1))
        return logits.squeeze(-1), self.value_head(g).squeeze(-1)


def _gather_rows(rows: Tensor, index: Tensor) -> Tensor:
    """rows [B, N, d], index [B, L] -> [B, L, d]."""
    return torch.gather(rows, 1, index.unsqueeze(-1).expand(-1, -1, rows.shape[-1]))
