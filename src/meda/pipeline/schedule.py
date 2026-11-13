"""List scheduling of an operation graph, the schedule validator and its metrics (SPEC 14.4).

`ListScheduler(priority).schedule(graph, platform)` is the classical method: time moves
forward; at each decision tick the ready operations start in priority order while a resource
instance is free and storage allows.

- An op is *ready* once every op it waits on is scheduled. An op fed by dispenses has a lead:
  its dispenses run on ports in the `dispense_ticks` before it starts (just in time), so the
  decision tick is its start minus the lead.
- Resource instances are used in time order (each has a `free_at` tick). Mixers and ports take
  the earliest free instance, lowest index first; a detector also needs `changeover_ticks`
  after a droplet of a different composition, so the one that can start soonest is taken, one
  needing no changeover first.
- Storage: a droplet is stored from its producer's end until its last consumer starts. An op
  starts only if, with what is already scheduled, the stored droplets stay within
  `graph.storage_capacity` at every tick (droplets whose consumers are not all scheduled count
  as stored for ever, so the check never has to be undone), and only if the ops left can still
  run one at a time in plan order within capacity afterwards (the plan's own order is
  storage-feasible, 7.3), so list scheduling never boxes itself in.
- DISCARD ops have no resource and start as soon as they can.
- Priority `critical_path` (default): longest duration path to a sink, then fewer stored
  droplets after the start, then the DAG's (step, id). `plan_order`: (step, id) alone.
- If nothing can ever start (storage full), `ScheduleDeadlock` names the instance, the tick
  and the stored droplets. No silent stretch (rule 9).
"""

from __future__ import annotations

import math
from collections.abc import Iterable
from dataclasses import dataclass
from enum import Enum
from itertools import pairwise

from meda.pipeline.opgraph import Op, OpGraph, OpKind
from meda.pipeline.platform import Platform

INF = math.inf
Resource = tuple[str, int] | tuple[str, int, int]


@dataclass(frozen=True, slots=True)
class Slot:
    """When an op runs and on which resource instance (None for DELIVER and DISCARD, and for
    a zero-tick DISPENSE)."""

    start: int
    end: int
    resource: Resource | None


@dataclass(frozen=True, slots=True)
class Schedule:
    """`slots[i]` is op i's slot. `scheduler` names the method that made it."""

    instance_id: str
    scheduler: str
    slots: tuple[Slot, ...]


class ScheduleDeadlock(RuntimeError):
    """No operation can start: storage is full and nothing running will free it."""


class ScheduleRule(Enum):
    COMPLETE = "COMPLETE"
    PRECEDENCE = "PRECEDENCE"
    CAPACITY = "CAPACITY"
    STORAGE = "STORAGE"
    CHANGEOVER = "CHANGEOVER"


@dataclass(frozen=True, slots=True)
class ScheduleIssue:
    rule: ScheduleRule
    message: str


class ScheduleError(ValueError):
    """A schedule broke one of the rules of 14.4; `issues` lists every one found."""

    def __init__(self, instance_id: str, issues: list[ScheduleIssue]) -> None:
        self.issues = issues
        lines = "; ".join(f"{i.rule.value}: {i.message}" for i in issues[:5])
        more = f" (+{len(issues) - 5} more)" if len(issues) > 5 else ""
        super().__init__(f"{instance_id}: invalid schedule: {lines}{more}")

    @property
    def rules(self) -> set[ScheduleRule]:
        return {i.rule for i in self.issues}


# ---------------------------------------------------------------- storage


def stored_intervals(
    graph: OpGraph, start: dict[int, int], end: dict[int, int]
) -> list[tuple[int, float]]:
    """[from, to) per droplet whose producer is scheduled; `to` is INF while a consumer is
    unscheduled. Empty intervals are dropped."""
    out: list[tuple[int, float]] = []
    for d in graph.droplets:
        if d.producer not in end:
            continue
        begin = end[d.producer]
        if not d.consumers or any(c not in start for c in d.consumers):
            stop: float = INF if d.consumers else begin
        else:
            stop = max(start[c] for c in d.consumers)
        if stop > begin:
            out.append((begin, stop))
    return out


def peak(intervals: Iterable[tuple[int, float]]) -> int:
    """Most intervals covering one tick."""
    events: list[tuple[float, int]] = []
    for a, b in intervals:
        events.append((a, 1))
        events.append((b, -1))
    events.sort()  # at equal ticks the -1 comes first: [a, b) intervals
    best = cur = 0
    for _, delta in events:
        cur += delta
        best = max(best, cur)
    return best


# ---------------------------------------------------------------- scheduler


def critical_path(graph: OpGraph) -> dict[int, int]:
    """Longest duration path from each op to a sink (own duration included); start-start
    successors add nothing after the op."""
    succ_end: dict[int, list[int]] = {o.id: [] for o in graph.ops}
    succ_start: dict[int, list[int]] = {o.id: [] for o in graph.ops}
    for o in graph.ops:
        for p in o.after_end:
            succ_end[p].append(o.id)
        for p in o.after_start:
            succ_start[p].append(o.id)
    cp: dict[int, int] = {}
    for o in sorted(graph.ops, key=lambda o: -o.id):  # consumers have larger ids
        tail = max((cp[s] for s in succ_end[o.id]), default=0)
        cp[o.id] = o.duration + tail
        for s in succ_start[o.id]:
            cp[o.id] = max(cp[o.id], cp[s])
    return cp


class ListScheduler:
    """Classical list scheduling (SPEC 14.4)."""

    PRIORITIES = ("critical_path", "plan_order")

    def __init__(self, priority: str = "critical_path") -> None:
        if priority not in self.PRIORITIES:
            raise ValueError(f"priority must be one of {self.PRIORITIES}, got {priority!r}")
        self.priority = priority
        self.name = f"list_{priority}"

    def schedule(self, graph: OpGraph, platform: Platform) -> Schedule:
        return _ListRun(graph, platform, self.priority).run(self.name)


class _ListRun:
    def __init__(self, graph: OpGraph, platform: Platform, priority: str) -> None:
        self.g = graph
        self.p = platform
        self.priority = priority
        self.start: dict[int, int] = {}
        self.end: dict[int, int] = {}
        self.res: dict[int, Resource | None] = {}
        self.mixer_free = [0] * platform.mixers
        self.port_free: dict[int, list[int]] = {}
        self.det_free = [0] * platform.detectors
        self.det_last: list[object] = [None] * platform.detectors
        self.feeders = {o.id: graph.dispenses(o.id) for o in graph.ops}
        self.cp = critical_path(graph)
        self.consumers_of: dict[int, list[int]] = {}  # op -> droplets (index) it consumes
        for i, d in enumerate(graph.droplets):
            for c in d.consumers:
                self.consumers_of.setdefault(c, []).append(i)
        self.makes = {
            d.producer for d in graph.droplets if graph.ops[d.producer].kind is OpKind.MIX
        }
        self.droplet_of = {d.producer: i for i, d in enumerate(graph.droplets)}

    # ---- helpers
    def lead(self, o: Op) -> int:
        return self.p.dispense_ticks if self.feeders[o.id] else 0

    def ports(self, fluid: int) -> list[int]:
        return self.port_free.setdefault(fluid, [0] * self.p.ports_per_fluid)

    def ready(self, o: Op) -> bool:
        return all(q in self.end for q in o.after_end) and all(
            q in self.start for q in o.after_start
        )

    def earliest(self, o: Op, t: int) -> tuple[int, int | None]:
        """Earliest start of `o` with decision tick >= t, and the detector it would use."""
        lead = self.lead(o)
        s = t + lead
        s = max([s, *(self.end[q] for q in o.after_end), *(self.start[q] for q in o.after_start)])
        if self.p.dispense_ticks:
            for f in self.feeders[o.id]:
                assert f.fluid is not None
                s = max(s, min(self.ports(f.fluid)) + lead)
        det: int | None = None
        if o.kind is OpKind.MIX:
            s = max(s, min(self.mixer_free))
        elif o.kind is OpKind.DETECT:
            options = []
            for i in range(self.p.detectors):
                needs = self.det_last[i] is not None and self.det_last[i] != o.composition
                ready_at = self.det_free[i] + (self.p.changeover_ticks if needs else 0)
                options.append((max(s, ready_at), needs, i))
            s, _, det = min(options)
        return s, det

    def net_storage(self, o: Op) -> int:
        """Droplets made minus droplets this op is the last unscheduled user of."""
        freed = sum(
            1
            for i in self.consumers_of.get(o.id, [])
            if all(c == o.id or c in self.start for c in self.g.droplets[i].consumers)
        )
        return (1 if o.id in self.makes else 0) - freed

    def key(self, o: Op) -> tuple[int, ...]:
        if self.priority == "plan_order":
            return (o.step, o.node_id, o.id)
        return (-self.cp[o.id], self.net_storage(o), o.step, o.node_id, o.id)

    def storage_ok(self, o: Op, s: int) -> bool:
        return self.fits(o, s) and self.safe_after(o)

    def safe_after(self, o: Op) -> bool:
        """With `o` started, the ops left can still run one at a time in plan order within
        capacity (the DAG validator's replay guarantees that order at the start), so the
        scheduler never boxes itself in."""
        done = set(self.start) | {o.id}
        live = {
            i
            for i, d in enumerate(self.g.droplets)
            if d.producer in done and any(c not in done for c in d.consumers)
        }
        count = len(live)
        if count > self.g.storage_capacity:
            return False
        rest = sorted(
            (x for x in self.g.ops if x.id not in done and x.kind is not OpKind.DISPENSE),
            key=lambda x: (x.step, x.node_id, x.id),
        )
        for x in rest:
            done.add(x.id)
            for i in self.consumers_of.get(x.id, []):
                if i in live and all(c in done for c in self.g.droplets[i].consumers):
                    live.discard(i)
            if x.id in self.makes and self.g.droplets[self.droplet_of[x.id]].consumers:
                live.add(self.droplet_of[x.id])
            if len(live) > self.g.storage_capacity:
                return False
        return True

    def fits(self, o: Op, s: int) -> bool:
        start = dict(self.start)
        end = dict(self.end)
        start[o.id] = s
        end[o.id] = s + o.duration
        for f in self.feeders[o.id]:
            start[f.id] = s - f.duration
            end[f.id] = s
        return peak(stored_intervals(self.g, start, end)) <= self.g.storage_capacity

    def commit(self, o: Op, s: int, det: int | None) -> None:
        self.start[o.id] = s
        self.end[o.id] = s + o.duration
        if o.kind is OpKind.MIX:
            i = min(range(self.p.mixers), key=lambda k: (self.mixer_free[k] > s, k))
            assert self.mixer_free[i] <= s
            self.mixer_free[i] = s + o.duration
            self.res[o.id] = ("mixer", i)
        elif o.kind is OpKind.DETECT:
            assert det is not None
            self.det_free[det] = s + o.duration
            self.det_last[det] = o.composition
            self.res[o.id] = ("detector", det)
        else:
            self.res[o.id] = None
        for f in self.feeders[o.id]:
            self.start[f.id] = s - f.duration
            self.end[f.id] = s
            if f.duration == 0:
                self.res[f.id] = None
                continue
            assert f.fluid is not None
            ports = self.ports(f.fluid)
            k = min(range(len(ports)), key=lambda j: (ports[j] > s - f.duration, j))
            assert ports[k] <= s - f.duration
            ports[k] = s
            self.res[f.id] = ("port", f.fluid, k)

    def run(self, name: str) -> Schedule:
        todo = {o.id for o in self.g.ops if o.kind is not OpKind.DISPENSE}
        ops = self.g.ops
        t = 0
        blocked: set[int] = set()
        while todo:
            # resource-free ops (DELIVER, DISCARD) start as soon as they can
            moved = True
            while moved:
                moved = False
                for oid in sorted(todo):
                    o = ops[oid]
                    if o.kind in (OpKind.DISCARD, OpKind.DELIVER) and self.ready(o):
                        s, _ = self.earliest(o, t)
                        if self.storage_ok(o, s):
                            self.commit(o, s, None)
                            todo.discard(oid)
                            moved = True
            if not todo:
                break
            cands = [ops[i] for i in todo if self.ready(ops[i]) and i not in blocked]
            if not cands and blocked:
                # storage may free later without a new start: a stored droplet's last user
                # is already scheduled. Retry from the next such tick.
                frees = [b for _, b in stored_intervals(self.g, self.start, self.end)]
                later = [int(b) for b in frees if b != INF and b > t]
                later += [e for e in self.end.values() if e > t]
                if later:
                    t = min(later)
                    blocked.clear()
                    continue
            if not cands:
                stored = peak(stored_intervals(self.g, self.start, self.end))
                raise ScheduleDeadlock(
                    f"{self.g.instance_id}: no operation can start at tick {t}: "
                    f"{len(todo)} left, {stored} droplets stored, "
                    f"capacity {self.g.storage_capacity}"
                )
            t = max(t, min(self.earliest(o, t)[0] - self.lead(o) for o in cands))
            committed = False
            for o in sorted(cands, key=self.key):
                s, det = self.earliest(o, t)
                if s - self.lead(o) != t:
                    continue
                if not self.storage_ok(o, s):
                    blocked.add(o.id)
                    continue
                self.commit(o, s, det)
                todo.discard(o.id)
                committed = True
            if committed:
                blocked.clear()
        slots = tuple(Slot(self.start[o.id], self.end[o.id], self.res[o.id]) for o in ops)
        return Schedule(self.g.instance_id, name, slots)


# ---------------------------------------------------------------- validator


def check_schedule(schedule: Schedule, graph: OpGraph, platform: Platform) -> list[ScheduleIssue]:
    """Every broken rule of SPEC 14.4 (empty list = valid)."""
    issues: list[ScheduleIssue] = []

    def bad(rule: ScheduleRule, msg: str) -> None:
        issues.append(ScheduleIssue(rule, msg))

    if len(schedule.slots) != len(graph.ops):
        bad(ScheduleRule.COMPLETE, f"{len(schedule.slots)} slots for {len(graph.ops)} ops")
        return issues
    sl = schedule.slots
    for o in graph.ops:
        if sl[o.id].start < 0 or sl[o.id].end - sl[o.id].start != o.duration:
            bad(ScheduleRule.COMPLETE, f"op {o.id} runs {sl[o.id]} but lasts {o.duration}")
    # precedence
    for o in graph.ops:
        s = sl[o.id].start
        for q in o.after_end:
            if s < sl[q].end:
                bad(ScheduleRule.PRECEDENCE, f"op {o.id} starts {s} before op {q} ends")
        for q in o.after_start:
            if s < sl[q].start:
                bad(ScheduleRule.PRECEDENCE, f"op {o.id} starts {s} before op {q} starts")
        if o.feeds is not None and sl[o.id].end != sl[o.feeds].start:
            bad(ScheduleRule.PRECEDENCE, f"dispense {o.id} does not end when op {o.feeds} starts")
    # capacity: right kind of resource, in range, no overlap
    expect = {OpKind.MIX: "mixer", OpKind.DETECT: "detector", OpKind.DISPENSE: "port"}
    count = {"mixer": platform.mixers, "detector": platform.detectors}
    busy: dict[Resource, list[tuple[int, int, int]]] = {}
    for o in graph.ops:
        r = sl[o.id].resource
        want = expect.get(o.kind) if o.duration > 0 else None
        if want is None:
            if r is not None:
                bad(ScheduleRule.CAPACITY, f"op {o.id} ({o.kind.value}) holds {r}")
            continue
        if r is None or r[0] != want:
            bad(ScheduleRule.CAPACITY, f"op {o.id} ({o.kind.value}) on {r}, needs a {want}")
            continue
        if want == "port":
            if len(r) != 3 or r[1] != o.fluid or not 0 <= r[2] < platform.ports_per_fluid:
                bad(ScheduleRule.CAPACITY, f"dispense {o.id} of fluid {o.fluid} on {r}")
                continue
        elif len(r) != 2 or not 0 <= r[1] < count[want]:
            bad(ScheduleRule.CAPACITY, f"op {o.id} on {r}: no such {want}")
            continue
        busy.setdefault(r, []).append((sl[o.id].start, sl[o.id].end, o.id))
    for r, spans in busy.items():
        spans.sort()
        for (_, e1, a), (s2, _, b) in pairwise(spans):
            if s2 < e1:
                bad(ScheduleRule.CAPACITY, f"ops {a} and {b} overlap on {r}")
        if r[0] == "detector":
            for (_, e1, a), (s2, _, b) in pairwise(spans):
                differ = graph.ops[a].composition != graph.ops[b].composition
                if differ and s2 < e1 + platform.changeover_ticks:
                    bad(
                        ScheduleRule.CHANGEOVER,
                        f"ops {a} and {b} on {r}: {s2 - e1} ticks apart, "
                        f"changeover needs {platform.changeover_ticks}",
                    )
    start = {o.id: sl[o.id].start for o in graph.ops}
    end = {o.id: sl[o.id].end for o in graph.ops}
    top = peak(stored_intervals(graph, start, end))
    if top > graph.storage_capacity:
        bad(
            ScheduleRule.STORAGE,
            f"{top} droplets stored at once, capacity {graph.storage_capacity}",
        )
    return issues


def validate_schedule(schedule: Schedule, graph: OpGraph, platform: Platform) -> None:
    """Raise `ScheduleError` naming the instance and every broken rule."""
    issues = check_schedule(schedule, graph, platform)
    if issues:
        raise ScheduleError(graph.instance_id, issues)


# ---------------------------------------------------------------- metrics


@dataclass(frozen=True, slots=True)
class ScheduleMetrics:
    """All in ticks, exact integers. `storage_ticks`: stored droplets summed over ticks
    (occupancy x duration). `detector_wait`: ticks delivered droplets wait for a detector
    after their producer ends. `changeovers`: detector washes between different fluids."""

    makespan: int
    storage_peak: int
    storage_ticks: int
    detector_wait: int
    changeovers: int
    mixer_busy: int
    port_busy: int
    detector_busy: int

    def as_dict(self) -> dict[str, int]:
        return {
            "makespan": self.makespan,
            "storage_peak": self.storage_peak,
            "storage_ticks": self.storage_ticks,
            "detector_wait": self.detector_wait,
            "changeovers": self.changeovers,
            "mixer_busy": self.mixer_busy,
            "port_busy": self.port_busy,
            "detector_busy": self.detector_busy,
        }


def schedule_metrics(schedule: Schedule, graph: OpGraph) -> ScheduleMetrics:
    sl = schedule.slots
    start = {o.id: sl[o.id].start for o in graph.ops}
    end = {o.id: sl[o.id].end for o in graph.ops}
    spans = stored_intervals(graph, start, end)
    if any(b == INF for _, b in spans):
        raise ValueError(f"{graph.instance_id}: a droplet is never used up")
    busy = {"mixer": 0, "port": 0, "detector": 0}
    wait = 0
    by_det: dict[Resource, list[tuple[int, Op]]] = {}
    for o in graph.ops:
        r = sl[o.id].resource
        if r is not None:
            busy[r[0]] += o.duration
        if o.kind is OpKind.DETECT:
            ready = max([end[q] for q in o.after_end] or [start[o.id]])
            wait += start[o.id] - ready
            assert r is not None
            by_det.setdefault(r, []).append((start[o.id], o))
    changeovers = 0
    for runs in by_det.values():
        runs.sort(key=lambda x: x[0])
        changeovers += sum(1 for (_, a), (_, b) in pairwise(runs) if a.composition != b.composition)
    return ScheduleMetrics(
        makespan=max(end.values(), default=0),
        storage_peak=peak(spans),
        storage_ticks=int(sum(b - a for a, b in spans)),
        detector_wait=wait,
        changeovers=changeovers,
        mixer_busy=busy["mixer"],
        port_busy=busy["port"],
        detector_busy=busy["detector"],
    )
