"""Network v2: set encoder over fluid, droplet and target tokens, candidate scorer, value head
(SPEC 11.4).

The network only scores: every candidate already carries the exact result of its action (the
afterstate features of `envs.encoding`), so nothing here computes a ratio (rule 1).

**No parameter's shape depends on a count** (SPEC 1.4): not on the number of fluids F, targets
T, droplets N or candidates L, not on M or storage capacity, not on any padding size. Those
appear only in activations. Every aggregation over a set is a masked mean or a softmax, never a
sum over a count. So one set of weights reads any instance.

1. Fluid embeddings from how the targets use each fluid:
   `use_f = mean_t MLP_u([target_comp[t, f]; targets[t]])`, `x_f = MLP_f([fluids[f]; use_f])`.
2. Composition embedding `comp(C) = MLP_c(C @ X)`: a linear layer on composition columns whose
   per-fluid weight is fluid f's own embedding x_f, so it exists for any F. Built from the
   pre-Transformer x_f, so compositions get context only through `use_f` (a deliberate choice).
3. Tokens: fluids x_f; droplets `MLP_d([droplets; comp])`; targets `MLP_t([targets; comp])`;
   a global token, a learned vector + `MLP_g(globals)`.
4. A pre-norm Transformer (no positional encoding) over [global; fluids; droplets; targets].
   **Relation biases:** for each related pair the attention logit of every head gets a
   learned linear function of the exact pair fact, with no constant term, so a zero fact
   adds nothing: droplet<->fluid from `droplet_comp`, target<->fluid from `target_comp`
   (one weight per head and direction), droplet<->target from `droplet_target` (distance,
   satisfies; a 2 -> heads map per direction). Each layer has its own bias maps.
5. Candidate scorer, one logit per candidate (candidates do not attend to each other, so cost
   stays linear in L): [candidate scalars; comp(cand_comp); target summary r; operand a;
   operand b; global token] -> MLP. `r` is a softmax over the valid targets, scored by
   `MLP_r(cand_target)`, of the contextual target tokens. Operands come from `cand_index` by
   (type, row): a fluid, droplet or target token, or a learned blank (Discard has none).
6. Value head: MLP on the global token.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from meda.envs.encoding import (
    D_CAND,
    D_DROP,
    D_FLUID,
    D_GLOB,
    D_PAIR,
    D_TGT,
    TYPE_DROPLET,
    TYPE_FLUID,
    TYPE_TARGET,
)

TensorObs = dict[str, Tensor]


@dataclass(frozen=True, slots=True)
class NetConfig:
    d_model: int = 128
    n_layers: int = 3
    n_heads: int = 4


def _mlp(d_in: int, d_hidden: int, d_out: int) -> nn.Sequential:
    return nn.Sequential(nn.Linear(d_in, d_hidden), nn.ReLU(), nn.Linear(d_hidden, d_out))


def masked_mean(x: Tensor, mask: Tensor, dim: int) -> Tensor:
    """Mean of `x` over `dim` where `mask` (broadcastable, 1 = valid) is set."""
    m = mask.to(x.dtype)
    return (x * m).sum(dim) / m.sum(dim).clamp(min=1.0)


class PreNormLayer(nn.Module):
    """A pre-norm Transformer encoder layer (attention, then a 4d ReLU feed-forward, each with
    a residual) over `scaled_dot_product_attention` with an additive per-head mask.

    Written out rather than `nn.TransformerEncoderLayer`: in eval mode that class switches to
    a fused kernel that handles a per-head float mask differently from training mode (measured:
    logits differed by 0.09), which would make rollout log-probs disagree with the update's.
    This layer runs the same code in both modes. Dropout 0."""

    def __init__(self, d: int, heads: int) -> None:
        super().__init__()
        if d % heads:
            raise ValueError(f"d_model {d} is not divisible by n_heads {heads}")
        self.heads = heads
        self.norm1 = nn.LayerNorm(d)
        self.qkv = nn.Linear(d, 3 * d)
        self.out = nn.Linear(d, d)
        self.norm2 = nn.LayerNorm(d)
        self.ff = _mlp(d, 4 * d, d)

    def forward(self, h: Tensor, bias: Tensor) -> Tensor:
        """h [B, n, d]; bias [B, heads, n, n] added to the attention logits."""
        b, n, d = h.shape
        q, k, v = (
            t.view(b, n, self.heads, d // self.heads).transpose(1, 2)
            for t in self.qkv(self.norm1(h)).chunk(3, dim=-1)
        )
        att = F.scaled_dot_product_attention(q, k, v, attn_mask=bias)
        h = h + self.out(att.transpose(1, 2).reshape(b, n, d))
        return h + self.ff(self.norm2(h))


class RelationBias(nn.Module):
    """Per-head attention-logit biases from the pair facts, for one layer."""

    def __init__(self, heads: int) -> None:
        super().__init__()
        self.heads = heads
        self.drop_fluid = nn.Linear(1, heads, bias=False)
        self.fluid_drop = nn.Linear(1, heads, bias=False)
        self.tgt_fluid = nn.Linear(1, heads, bias=False)
        self.fluid_tgt = nn.Linear(1, heads, bias=False)
        self.drop_tgt = nn.Linear(D_PAIR, heads, bias=False)
        self.tgt_drop = nn.Linear(D_PAIR, heads, bias=False)

    def forward(self, obs: TensorObs, key_pad: Tensor) -> Tensor:
        """Additive attention mask [B, heads, n, n] (rows: queries, columns: keys), token
        order [global; fluids; droplets; targets]; padded keys get -inf."""
        dc, tc, dt = obs["droplet_comp"], obs["target_comp"], obs["droplet_target"]
        b, nd, nf = dc.shape
        nt = tc.shape[1]
        h = self.heads
        n = 1 + nf + nd + nt
        f0, d0, t0 = 1, 1 + nf, 1 + nf + nd
        bias = dc.new_zeros(b, h, n, n)

        def put(q0: int, k0: int, block: Tensor) -> None:  # block [B, nq, nk, H]
            bias[:, :, q0 : q0 + block.shape[1], k0 : k0 + block.shape[2]] = block.permute(
                0, 3, 1, 2
            )

        put(d0, f0, self.drop_fluid(dc.unsqueeze(-1)))
        put(f0, d0, self.fluid_drop(dc.transpose(1, 2).unsqueeze(-1)))
        put(t0, f0, self.tgt_fluid(tc.unsqueeze(-1)))
        put(f0, t0, self.fluid_tgt(tc.transpose(1, 2).unsqueeze(-1)))
        put(d0, t0, self.drop_tgt(dt))
        put(t0, d0, self.tgt_drop(dt.transpose(1, 2)))
        return bias.masked_fill(key_pad[:, None, None, :], float("-inf"))


class SetEncoderNet(nn.Module):
    def __init__(self, cfg: NetConfig) -> None:
        super().__init__()
        d = cfg.d_model
        self.cfg = cfg
        self.use_mlp = _mlp(1 + D_TGT, d, d)
        self.fluid_mlp = _mlp(D_FLUID + d, d, d)
        self.comp_mlp = _mlp(d, d, d)
        self.droplet_mlp = _mlp(D_DROP + d, d, d)
        self.target_mlp = _mlp(D_TGT + d, d, d)
        self.global_token = nn.Parameter(torch.zeros(d))
        self.global_mlp = _mlp(D_GLOB, d, d)
        self.layers = nn.ModuleList(PreNormLayer(d, cfg.n_heads) for _ in range(cfg.n_layers))
        self.biases = nn.ModuleList(RelationBias(cfg.n_heads) for _ in range(cfg.n_layers))
        self.final_norm = nn.LayerNorm(d)
        self.summary_score = _mlp(D_PAIR, d, 1)
        self.blank = nn.Parameter(torch.zeros(d))
        self.scorer = nn.Sequential(
            nn.Linear(D_CAND + 5 * d, d),
            nn.ReLU(),
            nn.Linear(d, d),
            nn.ReLU(),
            nn.Linear(d, 1),
        )
        self.value_head = _mlp(d, d, 1)

    # ------------------------------------------------------------------ tokens

    def fluid_embeddings(self, obs: TensorObs) -> Tensor:
        """x_f [B, F, d], informed by how the valid targets use each fluid."""
        tc, tg = obs["target_comp"], obs["targets"]
        nf = tc.shape[2]
        u_in = torch.cat([tc.unsqueeze(-1), tg.unsqueeze(2).expand(-1, -1, nf, -1)], dim=-1)
        use = masked_mean(self.use_mlp(u_in), obs["target_mask"][:, :, None, None], dim=1)
        return self.fluid_mlp(torch.cat([obs["fluids"], use], dim=-1))

    def comp(self, fractions: Tensor, x: Tensor) -> Tensor:
        """Composition embedding: fractions [B, R, F] over fluid embeddings x [B, F, d]."""
        return self.comp_mlp(fractions @ x)

    def encode(self, obs: TensorObs) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
        """Fluid embeddings x and contextual tokens: global [B, d], fluids [B, F, d],
        droplets [B, N, d], targets [B, T, d]."""
        x = self.fluid_embeddings(obs)
        drops = self.droplet_mlp(
            torch.cat([obs["droplets"], self.comp(obs["droplet_comp"], x)], dim=-1)
        )
        tgts = self.target_mlp(
            torch.cat([obs["targets"], self.comp(obs["target_comp"], x)], dim=-1)
        )
        glob = (self.global_token + self.global_mlp(obs["globals"])).unsqueeze(1)
        h = torch.cat([glob, x, drops, tgts], dim=1)
        key_pad = torch.cat(
            [
                torch.zeros(h.shape[0], 1, dtype=torch.bool, device=h.device),
                obs["fluid_mask"] == 0,
                obs["droplet_mask"] == 0,
                obs["target_mask"] == 0,
            ],
            dim=1,
        )
        for layer, bias in zip(self.layers, self.biases, strict=True):
            h = layer(h, bias(obs, key_pad))
        h = self.final_norm(h)
        nf, nd = x.shape[1], drops.shape[1]
        return x, h[:, 0], h[:, 1 : 1 + nf], h[:, 1 + nf : 1 + nf + nd], h[:, 1 + nf + nd :]

    # ------------------------------------------------------------------ scoring

    def forward(self, obs: TensorObs) -> tuple[Tensor, Tensor]:
        """Raw candidate logits [B, L] (unmasked) and values [B]."""
        x, g, hf, hd, ht = self.encode(obs)
        b, nf, d = hf.shape
        nd, nt = hd.shape[1], ht.shape[1]

        c_res = self.comp(obs["cand_comp"], x)
        score = self.summary_score(obs["cand_target"]).squeeze(-1)  # [B, L, T]
        valid_t = obs["target_mask"][:, None, :] == 1
        score = score.masked_fill(~valid_t, torch.finfo(score.dtype).min)
        r = torch.softmax(score, dim=-1) @ ht

        pool = torch.cat([hf, hd, ht, self.blank.expand(b, 1, d)], dim=1)
        blank_row = nf + nd + nt
        index = obs["cand_index"].long()

        def operand(kind: Tensor, row: Tensor) -> Tensor:
            offset = torch.where(
                kind == TYPE_FLUID,
                row,
                torch.where(
                    kind == TYPE_DROPLET,
                    nf + row,
                    torch.where(kind == TYPE_TARGET, nf + nd + row, blank_row),
                ),
            )
            offset = torch.where(kind < 0, blank_row, offset)
            return _gather_rows(pool, offset)

        h_a = operand(index[..., 1], index[..., 2])
        h_b = operand(index[..., 3], index[..., 4])
        g_rep = g.unsqueeze(1).expand_as(h_a)
        z = torch.cat([obs["candidates"], c_res, r, h_a, h_b, g_rep], dim=-1)
        logits = self.scorer(z).squeeze(-1)
        return logits, self.value_head(g).squeeze(-1)


def _gather_rows(rows: Tensor, index: Tensor) -> Tensor:
    """rows [B, N, d], index [B, L] -> [B, L, d]."""
    return torch.gather(rows, 1, index.unsqueeze(-1).expand(-1, -1, rows.shape[-1]))
