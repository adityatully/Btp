"""Run validator and metrics of the grid harness (SPEC 14.10, M11).

`validate_run(run, graph, chip, gp)` replays a run tick by tick, independently of how the
executor and router made it, and raises `RunError` naming the instance and every broken rule:

1. CONTINUITY: a droplet moves at most one cell (8 neighbours) per tick, stays on the chip.
2. SPACING: droplets keep `spacing` empty cells between them, in space and time (each one's
   footprint at t against the other's at u = t - 1, t, t + 1, exemptions judged at min(t, u)),
   keep out of active regions grown by
   `spacing` (except their own op's), never cover a dead cell; only the exemptions of
   `router.exempt`.
3. OPERATIONS: precedence; durations; a mix's inputs are inside its module at its start; a
   detection's droplet is on its site for the hold; storage count <= capacity; one dispense per
   port at a time; one droplet per detection site, changeover between different compositions.
4. VOLUME: each droplet's takes sum to its volume; every droplet that does not merge into a mix
   leaves the chip at a waste cell.
5. COMPLETENESS: every op has a record.

`run_metrics(run, graph, chip, gp)` gives exact integers (14.10).
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from enum import Enum
from itertools import combinations, pairwise

import numpy as np

from meda.pipeline.executor import Run
from meda.pipeline.geometry import Chip, GridPlatform, at
from meda.pipeline.opgraph import OpGraph, OpKind
from meda.pipeline.router import INF, Body, exempt
from meda.pipeline.schedule import peak, stored_intervals


class RunRule(Enum):
    CONTINUITY = "continuity"
    SPACING = "spacing"
    OPERATIONS = "operations"
    VOLUME = "volume"
    COMPLETENESS = "completeness"


@dataclass(frozen=True, slots=True)
class RunIssue:
    rule: RunRule
    message: str


class RunError(ValueError):
    def __init__(self, instance_id: str, issues: list[RunIssue]) -> None:
        self.issues = issues
        lines = "; ".join(f"[{i.rule.value}] {i.message}" for i in issues[:10])
        more = f" (+{len(issues) - 10} more)" if len(issues) > 10 else ""
        super().__init__(f"{instance_id}: invalid run: {lines}{more}")

    @property
    def rules(self) -> set[RunRule]:
        return {i.rule for i in self.issues}


def take_times(run: Run, graph: OpGraph) -> tuple[dict[int, int], dict[int, int]]:
    """Start and end per op in 14.4's sense, for the storage count: an op "starts" when it
    takes its inputs (its first input droplet leaves its holder), and a dispense ends when its
    consumer takes the droplet, so a dispensed droplet is never stored."""
    rec = run.records
    first: dict[int, int] = {}
    for b in run.bodies:
        if b.dest is not None:
            first[b.dest] = min(first.get(b.dest, b.start), b.start)
    start = {o.id: first.get(o.id, rec[o.id].start) for o in graph.ops}
    end = {o.id: rec[o.id].end for o in graph.ops}
    for o in graph.ops:
        if o.kind is OpKind.DISPENSE and o.feeds is not None:
            start[o.id] = end[o.id] = start[o.feeds]
    return start, end


def _last_tick(b: Body, horizon: int) -> int:
    return int(b.end) - 1 if b.end != INF else horizon


def check_run(run: Run, graph: OpGraph, chip: Chip, gp: GridPlatform) -> list[RunIssue]:
    issues: list[RunIssue] = []

    def bad(rule: RunRule, msg: str) -> None:
        issues.append(RunIssue(rule, msg))

    sp = gp.spacing
    bodies = list(run.bodies)
    horizon = run.makespan + 1
    # ---- 5. completeness
    if len(run.records) != len(graph.ops) or any(r.op != i for i, r in enumerate(run.records)):
        bad(RunRule.COMPLETENESS, "records do not match the operations one to one")
        return issues
    rec = run.records
    # ---- 1. continuity and bounds
    for b in bodies:
        for p, q in pairwise(b.path):
            if max(abs(p[0] - q[0]), abs(p[1] - q[1])) > 1:
                bad(RunRule.CONTINUITY, f"body {b.id} jumps from {p} to {q}")
        for p in set(b.path):
            rect = at(p, b.shape)
            if not chip.inside(rect):
                bad(RunRule.CONTINUITY, f"body {b.id} leaves the chip at {p}")
            elif any(cell in chip.dead for cell in rect.cells()):
                bad(RunRule.SPACING, f"body {b.id} covers a dead cell at {p}")
        if b.end != INF and b.end <= b.start:
            bad(RunRule.CONTINUITY, f"body {b.id} ends before it starts")
    # ---- 2. spacing between droplets: a pair is checked at the ticks where either moves,
    # appears, or the other changes (a still pair stays as it was)
    alive_at: dict[int, list[Body]] = defaultdict(list)
    for b in bodies:
        for t in range(b.start, _last_tick(b, horizon) + 1):
            alive_at[t].append(b)
    for a, b in combinations(bodies, 2):
        lo = max(a.start, b.start) - 1
        hi = min(_last_tick(a, horizon), _last_tick(b, horizon)) + 1
        if lo > hi:
            continue
        for t in range(max(lo, 0), hi + 1):
            ra = a.rect(t)
            if ra is None:
                continue
            moving = any(x.pos(u) != x.pos(u - 1) for x in (a, b) for u in (t - 1, t, t + 1, t + 2))
            if not moving and t not in (lo + 1, hi - 1):
                continue
            for u in (t - 1, t, t + 1):
                if exempt(a, b, min(t, u), sp):
                    continue
                rb = b.rect(u)
                if rb is not None and ra.gap(rb) < sp:
                    bad(
                        RunRule.SPACING,
                        f"bodies {a.id} and {b.id} too close at tick {t} (other at {u})",
                    )
                    break
            else:
                continue
            break
    # ---- 2b. regions
    for g in run.regions:
        grown = g.rect.inflate(sp)
        for b in bodies:
            if b.dest == g.owner:
                continue
            t0, t1 = max(b.start, g.t0), min(_last_tick(b, horizon), int(min(g.t1, horizon)) - 1)
            for t in range(t0, t1 + 1):
                rb = b.rect(t)
                if rb is not None and rb.gap(grown) < 0:
                    bad(
                        RunRule.SPACING,
                        f"body {b.id} inside the {g.kind} of op {g.owner} at tick {t}",
                    )
                    break
    # ---- 3. operations
    take, _ = take_times(run, graph)
    for o in graph.ops:
        r = rec[o.id]
        if r.end - r.start != o.duration:
            bad(RunRule.OPERATIONS, f"op {o.id} lasts {r.end - r.start}, not {o.duration}")
        for pred in o.after_end:
            if take[o.id] < rec[pred].end:
                bad(
                    RunRule.OPERATIONS,
                    f"op {o.id} takes at {take[o.id]} before op {pred} ends at {rec[pred].end}",
                )
        for pred in o.after_start:
            if take[o.id] < take[pred]:
                bad(RunRule.OPERATIONS, f"op {o.id} takes before op {pred}")
    inputs_of: dict[int, list[Body]] = defaultdict(list)
    for b in bodies:
        if b.dest is not None:
            inputs_of[b.dest].append(b)
    for o in graph.ops:
        r = rec[o.id]
        if o.kind is OpKind.MIX:
            assert r.module is not None
            ins = inputs_of[o.id]
            need = sum(1 for d in graph.droplets if o.id in d.consumers)
            if len(ins) != need:
                bad(RunRule.OPERATIONS, f"mix {o.id} has {len(ins)} input droplets, needs {need}")
            for b in ins:
                last = b.rect(r.start)
                if b.end != r.start + 1 or last is None or not r.module.contains(last):
                    bad(
                        RunRule.OPERATIONS,
                        f"input body {b.id} of mix {o.id} is not in its module at {r.start}",
                    )
        if o.kind is OpKind.DETECT:
            assert r.module is not None and r.resource is not None
            ins = [b for b in inputs_of[o.id] if b.role == "detect"]
            if len(ins) != 1:
                bad(RunRule.OPERATIONS, f"detection {o.id} has {len(ins)} droplets")
                continue
            b = ins[0]
            for t in (r.start, r.end - 1):
                rb = b.rect(t)
                site = chip.detectors[r.resource[1]]
                if rb is None or rb.gap(site) >= 0 or not r.module.contains(rb):
                    bad(RunRule.OPERATIONS, f"detection {o.id}: droplet not on its site at {t}")
    # storage count (14.4's rule on the run's take times)
    stored = peak(stored_intervals(graph, *take_times(run, graph)))
    if stored > graph.storage_capacity:
        bad(
            RunRule.OPERATIONS,
            f"{stored} droplets stored at once > capacity {graph.storage_capacity}",
        )
    by_res: dict[tuple[object, ...], list[tuple[int, int, int]]] = defaultdict(list)
    for o in graph.ops:
        r = rec[o.id]
        if r.resource is not None:
            by_res[r.resource].append((r.start, r.end, o.id))
    for res, slots in by_res.items():
        slots.sort()
        for (_, e1, op1), (s2, _, b2) in pairwise(slots):
            gap = 0
            if res[0] == "detector":
                ca, cb = graph.ops[op1].composition, graph.ops[b2].composition
                gap = gp.changeover_ticks if ca != cb else 0
            if s2 < e1 + gap:
                bad(
                    RunRule.OPERATIONS,
                    f"{res}: op {b2} at {s2} overlaps op {op1} ending {e1} (+{gap})",
                )
    # ---- 4. volume and where droplets end
    for i, d in enumerate(graph.droplets):
        if d.takes and sum(d.takes) != d.volume:
            bad(RunRule.VOLUME, f"droplet {i}: takes {d.takes} do not sum to {d.volume}")
    for b in bodies:
        if b.role in ("mix",) or (b.end == INF and b.role == "stored"):
            if b.role == "stored":
                bad(RunRule.VOLUME, f"body {b.id} is still on the chip at the end")
            continue
        successors = [
            x
            for x in bodies
            if x.start == b.end
            and x.path[0] == b.path[-1]
            and (x.parent == b.id or x.droplet == b.droplet)
        ]
        if b.to_waste:
            if not any(at(b.path[-1], b.shape).covers(w) for w in chip.waste):
                bad(RunRule.VOLUME, f"body {b.id} leaves the chip away from a waste cell")
        elif b.end == INF:
            bad(RunRule.VOLUME, f"body {b.id} never leaves the chip")
        elif not successors:
            bad(RunRule.VOLUME, f"body {b.id} vanishes at tick {b.end}")
    return issues


def validate_run(run: Run, graph: OpGraph, chip: Chip, gp: GridPlatform) -> None:
    issues = check_run(run, graph, chip, gp)
    if issues:
        raise RunError(run.instance_id, issues)


@dataclass(frozen=True, slots=True)
class RunMetrics:
    """14.10, exact integers. `travel_cells`: cells moved by all droplets; `transit_ticks`:
    ticks droplets spent between release and arrival (moves and waits); `storage_ticks` and
    `storage_peak`: stored droplets (14.4's rule) summed over ticks and at most;
    `storage_cell_ticks`: cells of stored droplets with their spacing ring, summed over ticks;
    `peak_cells`: cells covered by droplets and active regions at once."""

    makespan: int
    schedule_makespan: int
    travel_cells: int
    transit_ticks: int
    storage_ticks: int
    storage_peak: int
    storage_cell_ticks: int
    peak_cells: int
    space_wait: int
    route_wait: int
    port_wait: int
    detector_wait: int
    storage_wait: int
    route_calls: int
    route_failures: int

    def as_dict(self) -> dict[str, int]:
        return {k: getattr(self, k) for k in self.__slots__}


def run_metrics(run: Run, graph: OpGraph, chip: Chip, gp: GridPlatform) -> RunMetrics:
    sp = gp.spacing
    travel = transit = cell_ticks = 0
    for b in run.bodies:
        if b.role in ("move", "mix", "detect") and len(b.path) > 1:
            travel += sum(p != q for p, q in pairwise(b.path))
            transit += len(b.path) - 1
        if b.role == "stored" and b.end != INF:
            g = at(b.path[0], b.shape).inflate(sp)
            r0, c0 = max(g.r, 0), max(g.c, 0)
            r1, c1 = min(g.r2, chip.height - 1), min(g.c2, chip.width - 1)
            cell_ticks += (r1 - r0 + 1) * (c1 - c0 + 1) * (int(b.end) - b.start)
    iv = stored_intervals(graph, *take_times(run, graph))
    storage_ticks = int(sum(min(e, run.makespan) - s for s, e in iv))
    peak_cells = 0
    for t in range(run.makespan + 1):
        used = np.zeros((chip.height, chip.width), dtype=bool)
        for b in run.bodies:
            rect = b.rect(t)
            if rect is not None:
                used[rect.r : rect.r2 + 1, rect.c : rect.c2 + 1] = True
        for reg in run.regions:
            if reg.active(t):
                rr = reg.rect
                used[rr.r : rr.r2 + 1, rr.c : rr.c2 + 1] = True
        peak_cells = max(peak_cells, int(used.sum()))
    w = run.waits
    return RunMetrics(
        run.makespan,
        run.schedule_makespan,
        travel,
        transit,
        storage_ticks,
        peak(iv),
        cell_ticks,
        peak_cells,
        w["space"],
        w["route"],
        w["port"],
        w["detector"],
        w["storage"],
        run.route_calls,
        run.route_failures,
    )


__all__ = [
    "RunError",
    "RunIssue",
    "RunMetrics",
    "RunRule",
    "check_run",
    "run_metrics",
    "validate_run",
]
