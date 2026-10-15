"""Proxy and structural (tier-1) metrics of a `MixingDAG` (SPEC 12.1).

Everything is computed from the DAG alone, in exact integers, so every producer (solver or
agent) is measured on the same columns.

- **Proxy metrics** are what the sample-preparation literature reports: mixes, reagent by fluid
  kind, waste, distinct intermediates, and the sharing count (mix nodes whose fluid reaches two
  or more distinct targets).
- **Structural metrics** come from an abstract list schedule: no grid and no routing, only a
  duration per operation and a cap on concurrent mixes (`ScheduleConfig`, from the `eval`
  config, since these are physical assumptions). Each MIX or OUTPUT starts as soon as its inputs
  exist and the mix cap allows, ties broken by the plan's own (step, id) order. WASTE is
  instantaneous and happens once every other use of the droplet has started, so it ends a
  droplet's life but never adds to the makespan.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Mapping
from dataclasses import asdict, dataclass
from typing import Any

from meda.core.dag import DagNode, MixingDAG, NodeType
from meda.core.fluids import FluidKind
from meda.core.problem import Instance


@dataclass(frozen=True, slots=True)
class ScheduleConfig:
    """Durations are in abstract time slots (>= 1). `max_concurrent_mixes` None = unlimited."""

    mix_duration: int
    output_duration: int
    max_concurrent_mixes: int | None

    def __post_init__(self) -> None:
        if self.mix_duration < 1 or self.output_duration < 1:
            raise ValueError(f"schedule durations must be >= 1 slot: {self}")
        if self.max_concurrent_mixes is not None and self.max_concurrent_mixes < 1:
            raise ValueError(f"max_concurrent_mixes must be >= 1 or null: {self}")

    @staticmethod
    def from_dict(data: Mapping[str, Any]) -> ScheduleConfig:
        keys = {"mix_duration", "output_duration", "max_concurrent_mixes"}
        if set(data) != keys:
            raise ValueError(f"eval.schedule must have keys {sorted(keys)}; got {sorted(data)}")
        cap = data["max_concurrent_mixes"]
        return ScheduleConfig(
            int(data["mix_duration"]),
            int(data["output_duration"]),
            None if cap is None else int(cap),
        )


@dataclass(frozen=True, slots=True)
class DagMetrics:
    """All volumes in units, all times in slots."""

    # proxy
    mix_ops: int
    outputs: int
    total_reagent: int
    sample_volume: int
    reagent_volume: int
    buffer_volume: int
    waste_volume: int
    distinct_intermediates: int
    sharing_count: int
    # structural (list schedule)
    makespan: int
    critical_path: int
    peak_droplets: int
    storage_slots: int

    def as_dict(self) -> dict[str, int]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class Schedule:
    """`start` / `finish` per MIX, OUTPUT and WASTE node id (WASTE has start == finish)."""

    start: Mapping[int, int]
    finish: Mapping[int, int]


def _links(dag: MixingDAG) -> tuple[dict[int, list[int]], dict[int, list[int]]]:
    parents: dict[int, list[int]] = defaultdict(list)
    children: dict[int, list[int]] = defaultdict(list)
    for e in dag.edges:
        parents[e.dst].append(e.src)
        children[e.src].append(e.dst)
    return parents, children


def _duration(node: DagNode, cfg: ScheduleConfig) -> int:
    return cfg.mix_duration if node.type is NodeType.MIX else cfg.output_duration


def list_schedule(dag: MixingDAG, cfg: ScheduleConfig) -> Schedule:
    """Greedy ASAP schedule under the mix cap (see module docstring)."""
    nodes = {n.id: n for n in dag.nodes}
    parents, children = _links(dag)
    finish: dict[int, int] = {n.id: 0 for n in dag.nodes if n.type is NodeType.RESERVOIR}
    start: dict[int, int] = {}
    pending = sorted(
        (n for n in dag.nodes if n.type in (NodeType.MIX, NodeType.OUTPUT)),
        key=lambda n: (n.step, n.id),
    )
    mixes_started: list[int] = []
    t = 0
    while pending:
        running = sum(1 for nid in mixes_started if start[nid] <= t < finish[nid])
        waiting: list[DagNode] = []
        for n in pending:
            ps = parents[n.id]
            ready = max((finish[p] for p in ps if p in finish), default=0)
            if any(p not in finish for p in ps) or ready > t:
                waiting.append(n)
                continue
            if n.type is NodeType.MIX and cfg.max_concurrent_mixes is not None:
                if running >= cfg.max_concurrent_mixes:
                    waiting.append(n)
                    continue
                running += 1
            start[n.id] = t
            finish[n.id] = t + _duration(n, cfg)
            if n.type is NodeType.MIX:
                mixes_started.append(n.id)
        pending = waiting
        if pending:
            later = [f for f in finish.values() if f > t]
            if not later:
                raise ValueError(
                    f"{dag.instance_id}: list schedule stuck at slot {t}; "
                    f"{len(pending)} operations wait on inputs that never finish"
                )
            t = min(later)
    for n in dag.nodes:
        if n.type is NodeType.WASTE:
            (p,) = parents[n.id]
            siblings = [start[c] for c in children[p] if nodes[c].type is not NodeType.WASTE]
            start[n.id] = finish[n.id] = max([finish[p], *siblings])
    return Schedule(start, {nid: f for nid, f in finish.items() if nid in start})


def critical_path(dag: MixingDAG, cfg: ScheduleConfig) -> int:
    """Longest chain of MIX/OUTPUT durations: the makespan with unlimited concurrent mixes."""
    parents, _ = _links(dag)
    done: dict[int, int] = {}
    for n in sorted(dag.nodes, key=lambda x: (x.step, x.id)):
        if n.type is NodeType.RESERVOIR:
            done[n.id] = 0
        elif n.type is NodeType.WASTE:
            continue
        else:
            done[n.id] = max((done[p] for p in parents[n.id]), default=0) + _duration(n, cfg)
    return max(done.values(), default=0)


def _sharing_count(dag: MixingDAG) -> int:
    """MIX nodes whose fluid reaches OUTPUT nodes of two or more distinct targets."""
    _, children = _links(dag)
    reach: dict[int, frozenset[int]] = {}
    for n in sorted(dag.nodes, key=lambda x: (x.step, x.id), reverse=True):
        if n.type is NodeType.OUTPUT:
            assert n.target_id is not None
            reach[n.id] = frozenset({n.target_id})
        else:
            reach[n.id] = frozenset().union(*(reach.get(c, frozenset()) for c in children[n.id]))
    return sum(1 for n in dag.nodes if n.type is NodeType.MIX and len(reach[n.id]) >= 2)


def dag_metrics(dag: MixingDAG, instance: Instance, cfg: ScheduleConfig) -> DagMetrics:
    """Metrics of `dag` (complete or not). Totals are recomputed from the nodes, not read from
    the producer's reported totals (the validator checks those separately)."""
    drawn = {kind: 0 for kind in FluidKind}
    mixes = outputs = waste = 0
    for n in dag.nodes:
        if n.type is NodeType.RESERVOIR:
            assert n.fluid_id is not None
            drawn[instance.fluids[n.fluid_id].kind] += n.volume
        elif n.type is NodeType.MIX:
            mixes += 1
        elif n.type is NodeType.OUTPUT:
            outputs += 1
        else:
            waste += n.volume

    sched = list_schedule(dag, cfg)
    nodes = {n.id: n for n in dag.nodes}
    _, children = _links(dag)
    makespan = max(
        (f for nid, f in sched.finish.items() if nodes[nid].type is not NodeType.WASTE),
        default=0,
    )
    events: list[tuple[int, int]] = []  # (slot, +1 / -1); a release sorts before a start
    storage = 0
    for n in dag.nodes:
        if n.type is not NodeType.MIX:
            continue
        uses = [sched.start[c] for c in children[n.id]]
        end = max(uses) if uses else makespan  # a failed episode can leave it on chip
        events += [(sched.start[n.id], 1), (end, -1)]
        storage += end - sched.finish[n.id]
    alive = peak = 0
    for _, delta in sorted(events):
        alive += delta
        peak = max(peak, alive)

    return DagMetrics(
        mix_ops=mixes,
        outputs=outputs,
        total_reagent=sum(drawn.values()),
        sample_volume=drawn[FluidKind.SAMPLE],
        reagent_volume=drawn[FluidKind.REAGENT],
        buffer_volume=drawn[FluidKind.BUFFER],
        waste_volume=waste,
        distinct_intermediates=len({n.composition for n in dag.nodes if n.type is NodeType.MIX}),
        sharing_count=_sharing_count(dag),
        makespan=makespan,
        critical_path=critical_path(dag, cfg),
        peak_droplets=peak,
        storage_slots=storage,
    )
