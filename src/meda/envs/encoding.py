"""State -> padded numpy observation, including per-candidate afterstate features (SPEC 9.2-9.3).
# this file is used to encode the state into a padded numpy observation which will
#  be used to train the policy network and is the input to the agent when making a decision
The network never does arithmetic: every result composition, distance and "satisfies target"
flag is computed here from the exact integer compositions of `meda.core`. Results are formed as
exact integer numerators over an integer denominator (numpy int64) and only then divided, so each
float is the correctly rounded value of the exact rational. "Satisfies" flags are compared
exactly in integers.

All float features lie in [0, 1]. Volumes are scaled by the chip's max mix volume M.

Rows of `sources`: reservoirs first (rows 0..F-1, by fluid id), then droplets on chip in the
state's canonical order (rows F..F+n-1).
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
import numpy.typing as npt
from gymnasium import spaces

from meda.core.actions import Action, Discard, Mix, Output, Source
from meda.core.fluids import FluidKind
from meda.core.problem import Instance
from meda.core.state import PrepState

# Bound on denominators (compositions, tolerances) so exact int64 products cannot overflow.
EXACT_LIMIT = 2**20
KINDS = (FluidKind.SAMPLE, FluidKind.REAGENT, FluidKind.BUFFER)
OP_MIX, OP_OUTPUT, OP_DISCARD = 0, 1, 2

Obs = dict[str, npt.NDArray[Any]]


@dataclass(frozen=True, slots=True)
class EnvSizes:
    """Padding sizes (from `configs/env`). Exceeding any of them raises; nothing is truncated."""

    f_max: int
    t_max: int
    storage_max: int
    l_max: int

    @property
    def s_max(self) -> int:
        return self.f_max + self.storage_max

    @property
    def d_src(self) -> int:
        # composition, volume, is_reservoir, kind one-hot, denominator, L-inf and satisfies per
        # target, age
        return self.f_max + 1 + 1 + len(KINDS) + 1 + 2 * self.t_max + 1

    @property
    def d_tgt(self) -> int:
        # composition, tolerance, droplet volume, remaining demand fraction, met flag
        return self.f_max + 4

    @property
    def d_cand(self) -> int:
        # op one-hot, k, l, k+l, result composition, result denominator, L-inf per target,
        # satisfies any open target, reagent by kind, waste, storage after
        return 3 + 3 + self.f_max + 1 + self.t_max + 1 + len(KINDS) + 1 + 1

    D_GLOB = 5  # step fraction, storage fraction, remaining demand fraction, F/f_max, T/t_max


def den_feature(den: npt.NDArray[np.int64]) -> npt.NDArray[np.float64]:
    """1 - 1/den: 0 for pure fluids, rising toward 1 for finer compositions."""
    return 1.0 - 1.0 / den.astype(np.float64)


class Encoder:
    def __init__(self, sizes: EnvSizes) -> None:
        self.sizes = sizes

    def observation_space(self) -> spaces.Dict:
        z = self.sizes
        return spaces.Dict(
            {
                "sources": spaces.Box(0.0, 1.0, shape=(z.s_max, z.d_src), dtype=np.float32),
                "source_mask": spaces.MultiBinary(z.s_max),
                "targets": spaces.Box(0.0, 1.0, shape=(z.t_max, z.d_tgt), dtype=np.float32),
                "target_mask": spaces.MultiBinary(z.t_max),
                "candidates": spaces.Box(0.0, 1.0, shape=(z.l_max, z.d_cand), dtype=np.float32),
                "cand_index": spaces.Box(
                    low=-1, high=max(z.s_max, z.t_max, 3), shape=(z.l_max, 3), dtype=np.int32
                ),
                "action_mask": spaces.MultiBinary(z.l_max),
                "globals": spaces.Box(0.0, 1.0, shape=(EnvSizes.D_GLOB,), dtype=np.float32),
            }
        )

    def check_instance(self, inst: Instance) -> None:
        """Raise if the instance does not fit the padding sizes or the exact-int64 bounds."""
        z = self.sizes

        def fail(msg: str) -> None:
            raise ValueError(f"instance '{inst.instance_id}': {msg}")

        if inst.num_fluids > z.f_max:
            fail(f"{inst.num_fluids} fluids > env F_max {z.f_max}")
        if len(inst.targets) > z.t_max:
            fail(f"{len(inst.targets)} targets > env T_max {z.t_max}")
        if inst.chip.storage_capacity > z.storage_max:
            fail(
                f"storage_capacity {inst.chip.storage_capacity} > env storage_capacity_max "
                f"{z.storage_max}"
            )
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
        if len(actions) > z.l_max:
            raise ValueError(
                f"instance '{inst.instance_id}' step {state.step}: {len(actions)} legal actions "
                f"> env L_max {z.l_max}"
            )
        chip = inst.chip
        m = float(chip.max_mix_volume)
        num_f = inst.num_fluids
        n = len(state.droplets)
        s = num_f + n

        # ---- sources: exact integer compositions
        snum = np.zeros((s, z.f_max), dtype=np.int64)
        sden = np.ones(s, dtype=np.int64)
        svol = np.zeros(s, dtype=np.int64)  # 0 marks a reservoir (unlimited)
        is_res = np.zeros(s, dtype=bool)
        kind = np.zeros((s, len(KINDS)), dtype=np.int64)
        age = np.zeros(s, dtype=np.float64)
        for f in inst.fluids:
            snum[f.id, f.id] = 1
            is_res[f.id] = True
            kind[f.id, KINDS.index(f.kind)] = 1
        row_of: dict[int, int] = {}
        for i, d in enumerate(state.droplets):
            r = num_f + i
            row_of[d.uid] = r
            snum[r, :num_f] = d.composition.num
            sden[r] = d.composition.den
            svol[r] = d.volume
            age[r] = (state.step - born.get(d.uid, state.step)) / inst.step_limit

        # ---- targets
        nt = len(inst.targets)
        tnum = np.zeros((nt, z.f_max), dtype=np.int64)
        tden = np.ones(nt, dtype=np.int64)
        tol_num = np.zeros(nt, dtype=np.int64)
        tol_den = np.ones(nt, dtype=np.int64)
        open_t = np.zeros(nt, dtype=bool)
        for t in inst.targets:
            tnum[t.id, :num_f] = t.composition.num
            tden[t.id] = t.composition.den
            tol_num[t.id], tol_den[t.id] = t.tolerance.num, t.tolerance.den
            open_t[t.id] = state.remaining_demand[t.id] > 0

        def dist_and_sat(
            num: npt.NDArray[np.int64], den: npt.NDArray[np.int64]
        ) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.bool_]]:
            """L-inf distance to each target (float) and exact `distance <= tolerance`."""
            diff = np.abs(
                num[:, None, :] * tden[None, :, None] - tnum[None, :, :] * den[:, None, None]
            ).max(axis=2)
            scale = den[:, None] * tden[None, :]
            dist = diff / scale
            sat = diff * tol_den[None, :] <= tol_num[None, :] * scale
            return dist, sat

        s_dist, s_sat = dist_and_sat(snum, sden)
        sources = np.zeros((z.s_max, z.d_src), dtype=np.float32)
        c = 0
        sources[:s, c : c + z.f_max] = snum / sden[:, None]
        c += z.f_max
        sources[:s, c] = np.where(is_res, 1.0, svol / m)
        c += 1
        sources[:s, c] = is_res
        c += 1
        sources[:s, c : c + len(KINDS)] = kind
        c += len(KINDS)
        sources[:s, c] = den_feature(sden)
        c += 1
        sources[:s, c : c + z.t_max] = 1.0
        sources[:s, c : c + nt] = np.where(open_t[None, :], s_dist, 1.0)
        c += z.t_max
        sources[:s, c : c + nt] = s_sat & open_t[None, :]
        c += z.t_max
        sources[:s, c] = age

        targets = np.zeros((z.t_max, z.d_tgt), dtype=np.float32)
        for t in inst.targets:
            targets[t.id, : z.f_max] = tnum[t.id] / tden[t.id]
            targets[t.id, z.f_max] = float(t.tolerance)
            targets[t.id, z.f_max + 1] = t.droplet_volume / m
            targets[t.id, z.f_max + 2] = state.remaining_demand[t.id] / t.demand
            targets[t.id, z.f_max + 3] = not open_t[t.id]

        # ---- candidates
        nl = len(actions)
        op = np.zeros(nl, dtype=np.int64)
        ia = np.zeros(nl, dtype=np.int64)
        ib = np.full(nl, -1, dtype=np.int64)
        kk = np.zeros(nl, dtype=np.int64)
        ll = np.zeros(nl, dtype=np.int64)

        def row(src: Source) -> int:
            return src.index if src.is_reservoir else row_of[src.index]

        for j, act in enumerate(actions):
            if isinstance(act, Mix):
                op[j], ia[j], ib[j], kk[j], ll[j] = OP_MIX, row(act.a), row(act.b), act.k, act.l
            elif isinstance(act, Output):
                op[j], ia[j], ib[j] = OP_OUTPUT, row(act.src), act.target
                kk[j] = inst.targets[act.target].droplet_volume
            elif isinstance(act, Discard):
                op[j], ia[j] = OP_DISCARD, row_of[act.uid]
                kk[j] = svol[ia[j]]

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
        for col in range(z.f_max):
            g = np.gcd(g, rnum[:, col])
        rnum, rden = rnum // g[:, None], rden // g
        r_dist, r_sat = dist_and_sat(rnum, rden)

        res_vol = np.where(is_res[ia], -1, svol[ia])
        a_used = (res_vol == kk) & ~is_res[ia]
        b_used = is_mix & ~is_res[ib_src] & (svol[ib_src] == ll)
        storage_after = np.where(
            is_mix,
            n + 1 - a_used - b_used,
            np.where(op == OP_OUTPUT, n - a_used, n - 1),
        )

        cand = np.zeros((z.l_max, z.d_cand), dtype=np.float32)
        c = 0
        cand[np.arange(nl), op] = 1.0
        c += 3
        cand[:nl, c] = kk / m
        cand[:nl, c + 1] = lm / m
        cand[:nl, c + 2] = np.where(is_mix, kk + lm, kk) / m
        c += 3
        cand[:nl, c : c + z.f_max] = rnum / rden[:, None]
        c += z.f_max
        cand[:nl, c] = den_feature(rden)
        c += 1
        cand[:nl, c : c + z.t_max] = 1.0
        cand[:nl, c : c + nt] = np.where(open_t[None, :], r_dist, 1.0)
        c += z.t_max
        cand[:nl, c] = (r_sat & open_t[None, :]).any(axis=1)
        c += 1
        reagent = kk[:, None] * kind[ia] + lm[:, None] * np.where(is_mix[:, None], kind[ib_src], 0)
        cand[:nl, c : c + len(KINDS)] = np.where(op[:, None] == OP_DISCARD, 0, reagent) / m
        c += len(KINDS)
        cand[:nl, c] = np.where(op == OP_DISCARD, kk, 0) / m
        c += 1
        cand[:nl, c] = storage_after / chip.storage_capacity

        cand_index = np.full((z.l_max, 3), -1, dtype=np.int32)
        cand_index[:nl, 0] = op
        cand_index[:nl, 1] = ia
        cand_index[:nl, 2] = ib

        total_demand = sum(t.demand for t in inst.targets)
        glob = np.array(
            [
                state.step / inst.step_limit,
                n / chip.storage_capacity,
                sum(state.remaining_demand) / total_demand if total_demand else 0.0,
                num_f / z.f_max,
                nt / z.t_max,
            ],
            dtype=np.float32,
        )
        return {
            "sources": sources,
            "source_mask": _mask(s, z.s_max),
            "targets": targets,
            "target_mask": _mask(nt, z.t_max),
            "candidates": cand,
            "cand_index": cand_index,
            "action_mask": _mask(nl, z.l_max),
            "globals": glob,
        }


def _mask(n: int, size: int) -> npt.NDArray[np.int8]:
    out = np.zeros(size, dtype=np.int8)
    out[:n] = 1
    return out
