"""State -> padded numpy observation v2: tokens plus pair matrices (SPEC 9.2-9.3).

The network never does arithmetic: every result composition, distance and "satisfies target"
flag is computed here from the exact integer compositions of `meda.core`. Results are formed as
exact integer numerators over an integer denominator (numpy int64) and only then divided, so each
float is the correctly rounded value of the exact rational. "Satisfies" flags are compared
exactly in integers.

**Every count is a set of rows, never a feature column** (SPEC 1.4). Fluids, droplets, targets
and candidates are rows; compositions are fraction matrices over the fluid rows; per-target
detail is in pair matrices. No feature column belongs to a particular fluid or target, so the
padding sizes `F_pad`, `T_pad`, `N_pad`, `L_pad` are memory settings, not limits of a network.

**Scales.** Every volume appears twice: relative (`/ M`, how much of a mixer it fills) and
absolute (`/ volume_ref`, the same on every chip). Absolute counts are divided by `count_ref`.
These are scales, not caps: such features may exceed 1.

Row order: fluids by fluid id (the fluid rows are the reservoirs); droplets in the state's
canonical order; targets by target id; candidates in `rules.legal_actions` order.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
import numpy.typing as npt
from gymnasium import spaces

from meda.core.actions import Action, Discard, Mix, Output, SourceKind
from meda.core.fluids import FluidKind
from meda.core.problem import Instance
from meda.core.state import PrepState

# Bound on denominators (compositions, tolerances) so exact int64 products cannot overflow.
EXACT_LIMIT = 2**20
KINDS = (FluidKind.SAMPLE, FluidKind.REAGENT, FluidKind.BUFFER)
OP_MIX, OP_OUTPUT, OP_DISCARD = 0, 1, 2
TYPE_FLUID, TYPE_DROPLET, TYPE_TARGET = 0, 1, 2

D_FLUID = 3  # kind one-hot
D_DROP = 6  # vol/M, vol/volume_ref, denominator, age, nearest open distance, satisfies any
D_TGT = 7  # dvol/M, dvol/volume_ref, tolerance, rem/demand, rem/count_ref, met, denominator
D_CAND = 17  # op (3), k, l, k+l on both scales (6), den, nearest, satisfies, reagent (3),
#              waste, droplets after / storage capacity
D_GLOB = 7  # step/limit, steps left, on chip/capacity, rem/total, rem/count_ref, M, capacity
D_PAIR = 2  # L-inf distance, satisfies flag

Obs = dict[str, npt.NDArray[Any]]


@dataclass(frozen=True, slots=True)
class EnvSizes:
    """Padding sizes and feature scales (from `configs/env`). Exceeding a padding size raises;
    nothing is truncated. Raising one never invalidates trained weights (SPEC 9.2)."""

    f_pad: int
    t_pad: int
    n_pad: int
    l_pad: int
    volume_ref: int
    count_ref: int

    def __post_init__(self) -> None:
        for name in ("f_pad", "t_pad", "n_pad", "l_pad", "volume_ref", "count_ref"):
            if getattr(self, name) < 1:
                raise ValueError(f"env: {name} must be >= 1, got {getattr(self, name)}")


def den_feature(den: npt.NDArray[np.int64]) -> npt.NDArray[np.float64]:
    """1 - 1/den: 0 for pure fluids, rising toward 1 for finer compositions."""
    return 1.0 - 1.0 / den.astype(np.float64)


class Encoder:
    def __init__(self, sizes: EnvSizes) -> None:
        self.sizes = sizes

    def observation_space(self) -> spaces.Dict:
        z = self.sizes
        inf = np.inf

        def box(shape: tuple[int, ...], high: float = 1.0) -> spaces.Box:
            return spaces.Box(0.0, high, shape=shape, dtype=np.float32)

        return spaces.Dict(
            {
                "fluids": box((z.f_pad, D_FLUID)),
                "fluid_mask": spaces.MultiBinary(z.f_pad),
                "droplets": box((z.n_pad, D_DROP), inf),
                "droplet_mask": spaces.MultiBinary(z.n_pad),
                "droplet_comp": box((z.n_pad, z.f_pad)),
                "targets": box((z.t_pad, D_TGT), inf),
                "target_mask": spaces.MultiBinary(z.t_pad),
                "target_comp": box((z.t_pad, z.f_pad)),
                "droplet_target": box((z.n_pad, z.t_pad, D_PAIR)),
                "candidates": box((z.l_pad, D_CAND), inf),
                "cand_comp": box((z.l_pad, z.f_pad)),
                "cand_target": box((z.l_pad, z.t_pad, D_PAIR)),
                "cand_index": spaces.Box(
                    low=-1,
                    high=max(z.f_pad, z.n_pad, z.t_pad, 3),
                    shape=(z.l_pad, 5),
                    dtype=np.int32,
                ),
                "action_mask": spaces.MultiBinary(z.l_pad),
                "globals": box((D_GLOB,), inf),
            }
        )

    def check_instance(self, inst: Instance) -> None:
        """Raise if the instance does not fit the padding sizes or the exact-int64 bounds."""
        z = self.sizes

        def fail(msg: str) -> None:
            raise ValueError(f"instance '{inst.instance_id}': {msg}")

        if inst.num_fluids > z.f_pad:
            fail(f"{inst.num_fluids} fluids > env F_pad {z.f_pad}")
        if len(inst.targets) > z.t_pad:
            fail(f"{len(inst.targets)} targets > env T_pad {z.t_pad}")
        if inst.chip.storage_capacity > z.n_pad:
            fail(f"storage_capacity {inst.chip.storage_capacity} > env N_pad {z.n_pad}")
        if inst.chip.max_denominator > EXACT_LIMIT:
            fail(f"max_denominator {inst.chip.max_denominator} > encoder limit {EXACT_LIMIT}")
        for t in inst.targets:
            if t.composition.den > EXACT_LIMIT or t.tolerance.den > EXACT_LIMIT:
                fail(f"target {t.id} denominators exceed encoder limit {EXACT_LIMIT}")
            if t.tolerance > 1:
                fail(f"target {t.id} tolerance {t.tolerance} > 1 (L-inf distances are <= 1)")
            if t.droplet_volume > inst.chip.max_mix_volume:
                fail(
                    f"target {t.id} droplet_volume {t.droplet_volume} > max_mix_volume "
                    f"{inst.chip.max_mix_volume}: no droplet can ever be large enough"
                )

    def encode(
        self,
        state: PrepState,
        inst: Instance,
        actions: Sequence[Action],
        born: Mapping[int, int],
    ) -> Obs:
        z = self.sizes
        if len(actions) > z.l_pad:
            raise ValueError(
                f"instance '{inst.instance_id}' step {state.step}: {len(actions)} legal actions "
                f"> env L_pad {z.l_pad}"
            )
        chip = inst.chip
        m = float(chip.max_mix_volume)
        vref, cref = float(z.volume_ref), float(z.count_ref)
        nf = inst.num_fluids
        n = len(state.droplets)
        nt = len(inst.targets)

        # ---- sources (internal only): rows 0..F-1 are the reservoirs, F..F+n-1 the droplets
        s = nf + n
        snum = np.zeros((s, nf), dtype=np.int64)
        sden = np.ones(s, dtype=np.int64)
        svol = np.zeros(s, dtype=np.int64)  # 0 marks a reservoir (unlimited)
        is_res = np.zeros(s, dtype=bool)
        kind = np.zeros((s, len(KINDS)), dtype=np.int64)
        for f in inst.fluids:
            snum[f.id, f.id] = 1
            is_res[f.id] = True
            kind[f.id, KINDS.index(f.kind)] = 1
        row_of: dict[int, int] = {}
        age = np.zeros(n, dtype=np.float64)
        for i, d in enumerate(state.droplets):
            r = nf + i
            row_of[d.uid] = r
            snum[r] = d.composition.num
            sden[r] = d.composition.den
            svol[r] = d.volume
            age[i] = (state.step - born.get(d.uid, state.step)) / inst.step_limit

        # ---- targets
        tnum = np.zeros((nt, nf), dtype=np.int64)
        tden = np.ones(nt, dtype=np.int64)
        tol_num = np.zeros(nt, dtype=np.int64)
        tol_den = np.ones(nt, dtype=np.int64)
        open_t = np.zeros(nt, dtype=bool)
        for t in inst.targets:
            tnum[t.id] = t.composition.num
            tden[t.id] = t.composition.den
            tol_num[t.id], tol_den[t.id] = t.tolerance.num, t.tolerance.den
            open_t[t.id] = state.remaining_demand[t.id] > 0

        def pairs(
            num: npt.NDArray[np.int64], den: npt.NDArray[np.int64]
        ) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.bool_]]:
            """Per (row, target): L-inf distance and exact `distance <= tolerance`, with the
            met-target convention of SPEC 9.2 (distance 1, satisfies 0)."""
            diff = np.abs(
                num[:, None, :] * tden[None, :, None] - tnum[None, :, :] * den[:, None, None]
            ).max(axis=2, initial=0)
            scale = den[:, None] * tden[None, :]
            dist = np.where(open_t[None, :], diff / scale, 1.0)
            sat = (diff * tol_den[None, :] <= tol_num[None, :] * scale) & open_t[None, :]
            return dist, sat

        def nearest(dist: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
            return dist.min(axis=1, initial=1.0)

        # ---- fluid rows
        fluids = np.zeros((z.f_pad, D_FLUID), dtype=np.float32)
        fluids[:nf] = kind[:nf]

        # ---- droplet rows and their pairs
        dnum, dden, dvol = snum[nf:], sden[nf:], svol[nf:]
        d_dist, d_sat = pairs(dnum, dden)
        droplets = np.zeros((z.n_pad, D_DROP), dtype=np.float32)
        droplets[:n, 0] = dvol / m
        droplets[:n, 1] = dvol / vref
        droplets[:n, 2] = den_feature(dden)
        droplets[:n, 3] = age
        droplets[:n, 4] = nearest(d_dist)
        droplets[:n, 5] = d_sat.any(axis=1)
        droplet_comp = np.zeros((z.n_pad, z.f_pad), dtype=np.float32)
        droplet_comp[:n, :nf] = dnum / dden[:, None]
        droplet_target = np.zeros((z.n_pad, z.t_pad, D_PAIR), dtype=np.float32)
        droplet_target[:n, :nt, 0] = d_dist
        droplet_target[:n, :nt, 1] = d_sat

        # ---- target rows
        targets = np.zeros((z.t_pad, D_TGT), dtype=np.float32)
        target_comp = np.zeros((z.t_pad, z.f_pad), dtype=np.float32)
        for t in inst.targets:
            rem = state.remaining_demand[t.id]
            targets[t.id] = (
                t.droplet_volume / m,
                t.droplet_volume / vref,
                float(t.tolerance),
                rem / t.demand,
                rem / cref,
                not open_t[t.id],
                den_feature(tden[t.id : t.id + 1])[0],
            )
            target_comp[t.id, :nf] = tnum[t.id] / tden[t.id]

        # ---- candidates: exact afterstates
        nl = len(actions)
        # One (op, row of a, row of b or target id, k, l) per action, built as plain tuples and
        # converted once: per-element numpy writes dominated the encoder at a few thousand rows.
        out_vol = [t.droplet_volume for t in inst.targets]
        res = SourceKind.RESERVOIR
        rows: list[tuple[int, int, int, int, int]] = []
        for act in actions:
            if type(act) is Mix:
                a, b = act.a, act.b
                ra = a.index if a.kind is res else row_of[a.index]
                rb = b.index if b.kind is res else row_of[b.index]
                rows.append((OP_MIX, ra, rb, act.k, act.l))
            elif type(act) is Output:
                s = act.src
                rs = s.index if s.kind is res else row_of[s.index]
                rows.append((OP_OUTPUT, rs, act.target, out_vol[act.target], 0))
            elif type(act) is Discard:
                r = row_of[act.uid]
                rows.append((OP_DISCARD, r, -1, int(svol[r]), 0))
            else:
                raise TypeError(f"unknown action type {type(act).__name__}")
        cols = np.array(rows, dtype=np.int64).reshape(nl, 5)
        op, ia, ib, kk, ll = (np.ascontiguousarray(cols[:, c]) for c in range(5))

        is_mix = op == OP_MIX
        ib_src = np.where(is_mix, ib, ia)  # second source row (only meaningful for Mix)
        lm = np.where(is_mix, ll, 0)
        da, db = sden[ia], sden[ib_src]
        rnum = np.where(
            is_mix[:, None],
            kk[:, None] * snum[ia] * db[:, None] + lm[:, None] * snum[ib_src] * da[:, None],
            snum[ia],
        )
        rden = np.where(is_mix, (kk + lm) * da * db, da)
        g = rden.copy()
        for col in range(nf):
            g = np.gcd(g, rnum[:, col])
        rnum, rden = rnum // g[:, None], rden // g
        r_dist, r_sat = pairs(rnum, rden)

        a_used = ~is_res[ia] & (svol[ia] == kk)
        b_used = is_mix & ~is_res[ib_src] & (svol[ib_src] == ll)
        storage_after = np.where(
            is_mix,
            n + 1 - a_used - b_used,
            np.where(op == OP_OUTPUT, n - a_used, n - 1),
        )
        total = np.where(is_mix, kk + lm, kk)
        reagent = kk[:, None] * kind[ia] + lm[:, None] * np.where(is_mix[:, None], kind[ib_src], 0)
        reagent = np.where(op[:, None] == OP_DISCARD, 0, reagent)

        cand = np.zeros((z.l_pad, D_CAND), dtype=np.float32)
        cand[np.arange(nl), op] = 1.0
        cand[:nl, 3], cand[:nl, 4], cand[:nl, 5] = kk / m, lm / m, total / m
        cand[:nl, 6], cand[:nl, 7], cand[:nl, 8] = kk / vref, lm / vref, total / vref
        cand[:nl, 9] = den_feature(rden)
        cand[:nl, 10] = nearest(r_dist)
        cand[:nl, 11] = r_sat.any(axis=1)
        cand[:nl, 12:15] = reagent / vref
        cand[:nl, 15] = np.where(op == OP_DISCARD, kk, 0) / vref
        cand[:nl, 16] = storage_after / chip.storage_capacity
        cand_comp = np.zeros((z.l_pad, z.f_pad), dtype=np.float32)
        cand_comp[:nl, :nf] = rnum / rden[:, None]
        cand_target = np.zeros((z.l_pad, z.t_pad, D_PAIR), dtype=np.float32)
        cand_target[:nl, :nt, 0] = r_dist
        cand_target[:nl, :nt, 1] = r_sat

        # Operands by (type, row): a source row below F is a fluid, else droplet row - F.
        def typed(src_row: npt.NDArray[np.int64]) -> tuple[npt.NDArray[np.int64], ...]:
            fluid = src_row < nf
            return np.where(fluid, TYPE_FLUID, TYPE_DROPLET), np.where(fluid, src_row, src_row - nf)

        a_type, a_row = typed(ia)
        b_type_src, b_row_src = typed(np.maximum(ib, 0))
        b_type = np.where(is_mix, b_type_src, np.where(op == OP_OUTPUT, TYPE_TARGET, -1))
        b_row = np.where(is_mix, b_row_src, np.where(op == OP_OUTPUT, ib, -1))
        cand_index = np.full((z.l_pad, 5), -1, dtype=np.int32)
        cand_index[:nl] = np.stack([op, a_type, a_row, b_type, b_row], axis=1)

        total_demand = sum(t.demand for t in inst.targets)
        remaining = sum(state.remaining_demand)
        glob = np.array(
            [
                state.step / inst.step_limit,
                (inst.step_limit - state.step) / cref,
                n / chip.storage_capacity,
                remaining / total_demand if total_demand else 0.0,
                remaining / cref,
                chip.max_mix_volume / vref,
                chip.storage_capacity / cref,
            ],
            dtype=np.float32,
        )
        return {
            "fluids": fluids,
            "fluid_mask": _mask(nf, z.f_pad),
            "droplets": droplets,
            "droplet_mask": _mask(n, z.n_pad),
            "droplet_comp": droplet_comp,
            "targets": targets,
            "target_mask": _mask(nt, z.t_pad),
            "target_comp": target_comp,
            "droplet_target": droplet_target,
            "candidates": cand,
            "cand_comp": cand_comp,
            "cand_target": cand_target,
            "cand_index": cand_index,
            "action_mask": _mask(nl, z.l_pad),
            "globals": glob,
        }


def _mask(n: int, size: int) -> npt.NDArray[np.int8]:
    out = np.zeros(size, dtype=np.int8)
    out[:n] = 1
    return out
