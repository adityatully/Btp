"""Exact check of the list scheduler on small graphs (SPEC 14.4): a time-indexed ILP.

Variables: x[o, t] = 1 if op o starts at tick t (every op but DISPENSE; a dispense runs in the
`dispense_ticks` before its consumer, so its port use is written on the consumer's start),
stored[d, t] >= 0 per MIX droplet and tick, and the makespan C. Constraints:

- each op starts once, no earlier than its dispense lead;
- precedence: S_j >= S_i + dur_i (after_end), S_j >= S_i (after_start), with S = sum t x;
- capacity per tick: running mixes <= mixers; dispenses of fluid f <= ports_per_fluid;
  running detections <= detectors (no changeover: the check needs `changeover_ticks == 0`);
- storage per tick: stored[d, t] >= started(producer, t - dur) - started(consumer, t) for each
  consumer, and the sum over droplets <= capacity (the same [end, last start) rule as 14.3);
- C >= S_o + dur_o. Minimise C.

The horizon is a known feasible makespan (the list schedule's), so the ILP never misses the
optimum. Uses scipy's HiGHS (SPEC 3); never imported by `meda.core`.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from scipy.optimize import Bounds, LinearConstraint, milp
from scipy.sparse import coo_array

from meda.pipeline.opgraph import OpGraph, OpKind
from meda.pipeline.platform import Platform


@dataclass(frozen=True, slots=True)
class IlpResult:
    """`makespan` is the ILP's best (None if it found no schedule within the time limit);
    `optimal` is False if it hit the limit; `lower_bound` is the solver's proved bound."""

    makespan: int | None
    optimal: bool
    lower_bound: int


def exact_makespan(
    graph: OpGraph, platform: Platform, horizon: int, time_limit: float = 60.0
) -> IlpResult:
    """Smallest makespan of any schedule of `graph` within `horizon` ticks."""
    if platform.changeover_ticks:
        raise ValueError(
            f"{graph.instance_id}: the exact check has no detector changeover; "
            f"run it with changeover_ticks = 0 (got {platform.changeover_ticks})"
        )
    ops = [o for o in graph.ops if o.kind is not OpKind.DISPENSE]
    lead = {o.id: platform.dispense_ticks if graph.dispenses(o.id) else 0 for o in ops}
    dur = {o.id: o.duration for o in ops}
    var: dict[tuple[int, int], int] = {}
    for o in ops:
        for t in range(lead[o.id], horizon - dur[o.id] + 1):
            var[(o.id, t)] = len(var)
        if not any(k[0] == o.id for k in var):
            raise ValueError(f"{graph.instance_id}: horizon {horizon} too short for op {o.id}")
    mix_droplets = [d for d in graph.droplets if graph.ops[d.producer].kind is OpKind.MIX]
    stored: dict[tuple[int, int], int] = {}
    for i, _ in enumerate(mix_droplets):
        for t in range(horizon):
            stored[(i, t)] = len(var) + len(stored)
    c_var = len(var) + len(stored)
    n = c_var + 1

    rows: list[int] = []
    cols: list[int] = []
    vals: list[float] = []
    lo: list[float] = []
    hi: list[float] = []

    def row(entries: dict[int, float], low: float, high: float) -> None:
        r = len(lo)
        for c, v in entries.items():
            if v:
                rows.append(r)
                cols.append(c)
                vals.append(v)
        lo.append(low)
        hi.append(high)

    def starts(oid: int) -> dict[int, int]:
        return {t: j for (o, t), j in var.items() if o == oid}

    def start_sum(oid: int, sign: float = 1.0) -> dict[int, float]:
        return {j: sign * t for t, j in starts(oid).items()}

    def started_by(oid: int, tick: int, sign: float = 1.0) -> dict[int, float]:
        return {j: sign for t, j in starts(oid).items() if t <= tick}

    for o in ops:
        row({j: 1.0 for j in starts(o.id).values()}, 1, 1)
        for p in o.after_end:
            e = start_sum(o.id)
            for j, v in start_sum(p, -1.0).items():
                e[j] = e.get(j, 0.0) + v
            row(e, dur[p], np.inf)
        for p in o.after_start:
            e = start_sum(o.id)
            for j, v in start_sum(p, -1.0).items():
                e[j] = e.get(j, 0.0) + v
            row(e, 0, np.inf)
        e = {c_var: 1.0}
        for j, v in start_sum(o.id, -1.0).items():
            e[j] = v
        row(e, dur[o.id], np.inf)

    fluids: dict[int, list[int]] = {}
    for o in ops:
        for f in graph.dispenses(o.id):
            assert f.fluid is not None
            fluids.setdefault(f.fluid, []).append(o.id)
    for tick in range(horizon):
        for kind, cap in ((OpKind.MIX, platform.mixers), (OpKind.DETECT, platform.detectors)):
            e = {
                j: 1.0
                for o in ops
                if o.kind is kind
                for t, j in starts(o.id).items()
                if t <= tick < t + dur[o.id]
            }
            if e:
                row(e, 0, cap)
        if platform.dispense_ticks:
            for users in fluids.values():
                e = {
                    j: 1.0
                    for oid in users
                    for t, j in starts(oid).items()
                    if t - platform.dispense_ticks <= tick < t
                }
                if e:
                    row(e, 0, platform.ports_per_fluid)
        total = {}
        for i, d in enumerate(mix_droplets):
            total[stored[(i, tick)]] = 1.0
            made = started_by(d.producer, tick - dur[d.producer])
            for c in d.consumers:
                e = {stored[(i, tick)]: 1.0}
                for j, v in made.items():
                    e[j] = e.get(j, 0.0) - v
                for j, v in started_by(c, tick).items():
                    e[j] = e.get(j, 0.0) + v
                row(e, 0, np.inf)
        if total:
            row(total, 0, graph.storage_capacity)

    a = coo_array((vals, (rows, cols)), shape=(len(lo), n)).tocsr()
    cost = np.zeros(n)
    cost[c_var] = 1.0
    integrality = np.zeros(n)
    integrality[: len(var)] = 1
    upper = np.ones(n)
    upper[len(var) : c_var] = np.inf
    upper[c_var] = horizon
    res = milp(
        cost,
        constraints=LinearConstraint(a, np.array(lo), np.array(hi)),
        integrality=integrality,
        bounds=Bounds(np.zeros(n), upper),
        options={"time_limit": time_limit, "mip_rel_gap": 0.0},
    )
    bound = getattr(res, "mip_dual_bound", None)
    lower = math.ceil(bound - 1e-6) if bound is not None and math.isfinite(bound) else 0
    if res.x is None:
        if res.status != 1:  # 1 = time limit; anything else is a modelling error
            raise RuntimeError(f"{graph.instance_id}: exact check failed ({res.message})")
        return IlpResult(None, False, lower)
    best: int = round(float(res.fun))
    return IlpResult(best, bool(res.status == 0), best if res.status == 0 else lower)
