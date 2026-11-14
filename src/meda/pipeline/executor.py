"""The grid executor: list scheduling on the chip with travel, space and routes (SPEC 14.9, M11).

`GridExecutor(gp).run(graph, chip)` plays one plan out on the grid. The M10 list schedule of the
same graph, computed once, gives the order (the better of its two priorities); the executor
finds the real times. At each event tick, ready operations are taken in that order and each is
committed all-or-nothing:

- MIX: a port per dispensed input (the dispense runs for `dispense_ticks`, then the droplet
  appears at the port), a module from the placer, and router paths for every input into it
  (dispensed droplets from their ports, takes from their stored parents after `split_ticks`).
  The mix starts when the last input arrives and lasts its module's time; its product then
  waits in the module (stored, 14.7).
- DETECT: the detection site that can start soonest (changeover as 14.4) and a path to it; the
  droplet is held for `detect_ticks`, then routed to a waste port (a removal task).
- DISCARD, DELIVER: a path to the nearest waste port that has one; the droplet leaves the chip.

An op that cannot get storage, a port, a detector, space or a route stays ready and retries at
the next event (the doc's loop 1); waits are counted by reason. If nothing can commit, nothing
is moving or running and nothing is due, `ExecutionDeadlock` names the instance, the tick and
why each ready op waits (rule 9: no silent stretch).

`grid_opgraph(dag, instance, gp)` is the M10 operation graph with grid durations: each mix the
time of its module (by volume), dispense and detect from the library units.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field, replace

from meda.core.composition import Composition
from meda.core.dag import MixingDAG
from meda.core.problem import Instance
from meda.pipeline.geometry import Cell, Chip, GridPlatform, Rect
from meda.pipeline.interfaces import Placer, Router
from meda.pipeline.opgraph import Op, OpGraph, OpKind, build_opgraph
from meda.pipeline.placement import NearestPlacer
from meda.pipeline.router import INF, AStarRouter, Body, ChipState, Region, RouteRequest
from meda.pipeline.schedule import ListScheduler, schedule_metrics

WAIT_REASONS = ("storage", "port", "detector", "space", "route")


class ExecutionDeadlock(RuntimeError):
    """No operation can ever commit (14.9)."""


def grid_opgraph(dag: MixingDAG, instance: Instance, gp: GridPlatform) -> OpGraph:
    g = build_opgraph(dag, instance, gp.platform(mixers=1))
    vol = {d.producer: d.volume for d in g.droplets}
    ops = tuple(
        replace(o, duration=gp.mixer(vol[o.id])[1]) if o.kind is OpKind.MIX else o for o in g.ops
    )
    return replace(g, ops=ops)


@dataclass(frozen=True, slots=True)
class OpRecord:
    """When an op ran and where. `start`/`end`: for a MIX the mix itself; for a DISPENSE the
    port's busy ticks; for DETECT the hold; for DISCARD/DELIVER the tick it left its parent.
    `committed`: the decision tick. `resource`: ("port", fluid, k) or ("detector", k)."""

    op: int
    committed: int
    start: int
    end: int
    module: Rect | None = None
    resource: tuple[str, int] | tuple[str, int, int] | None = None


@dataclass(frozen=True, slots=True)
class Run:
    """One plan played out on the grid (14.9)."""

    instance_id: str
    platform: str
    records: tuple[OpRecord, ...]
    bodies: tuple[Body, ...]
    regions: tuple[Region, ...]
    schedule_makespan: int
    makespan: int
    waits: dict[str, int]
    route_calls: int
    route_failures: int


@dataclass
class _Held:
    """Where an opgraph droplet is now: the body holding it and the volume left."""

    body: int
    left: int


@dataclass
class _Removal:
    body: int
    ready: int
    detector: int | None  # the site it leaves
    region: int | None  # index in state.regions


@dataclass
class _Exec:
    gp: GridPlatform
    graph: OpGraph
    chip: Chip
    state: ChipState
    router: Router
    placer: Placer
    order: list[int]
    records: dict[int, OpRecord] = field(default_factory=dict)
    held: dict[int, _Held] = field(default_factory=dict)  # droplet index -> holder
    ports: dict[int, list[int]] = field(default_factory=dict)
    det_free: list[float] = field(default_factory=list)
    det_last: list[Composition | None] = field(default_factory=list)
    removals: list[_Removal] = field(default_factory=list)
    events: set[int] = field(default_factory=set)
    waits: dict[str, int] = field(default_factory=lambda: dict.fromkeys(WAIT_REASONS, 0))
    why: dict[int, str] = field(default_factory=dict)
    route_calls: int = 0
    route_failures: int = 0
    fail_reasons: dict[str, int] = field(default_factory=lambda: defaultdict(int))
    next_id: int = 0

    def __post_init__(self) -> None:
        g = self.graph
        self.det_free = [0.0] * len(self.chip.detectors)
        self.det_last = [None] * len(self.chip.detectors)
        self.droplet_of = {d.producer: i for i, d in enumerate(g.droplets)}
        self.inputs: dict[int, list[int]] = defaultdict(list)  # op -> droplet indices
        for i, d in enumerate(g.droplets):
            for c in d.consumers:
                self.inputs[c].append(i)
        self.makes = {d.producer for d in g.droplets if g.ops[d.producer].kind is OpKind.MIX}

    # ---- books
    def new_id(self) -> int:
        self.next_id += 1
        return self.next_id

    def take(self, i: int, op: int) -> int:
        d = self.graph.droplets[i]
        return d.takes[d.consumers.index(op)]

    def ready(self, o: Op, t: int) -> bool:
        return all(q in self.records and self.records[q].end <= t for q in o.after_end) and all(
            q in self.records for q in o.after_start
        )

    def storage_ok(self, o: Op) -> bool:
        """M10's count rule and plan-order safety check (14.4), on committed ops."""
        g = self.graph
        done = set(self.records) | {o.id} | {f.id for f in g.dispenses(o.id)}
        live = {
            i
            for i, d in enumerate(g.droplets)
            if d.producer in self.makes
            and d.producer in done
            and any(c not in done for c in d.consumers)
        }
        if len(live) > g.storage_capacity:
            return False
        rest = sorted(
            (x for x in g.ops if x.id not in done and x.kind is not OpKind.DISPENSE),
            key=lambda x: (x.step, x.node_id, x.id),
        )
        for x in rest:
            done.add(x.id)
            for i in self.inputs.get(x.id, []):
                if i in live and all(c in done for c in g.droplets[i].consumers):
                    live.discard(i)
            if x.id in self.makes and g.droplets[self.droplet_of[x.id]].consumers:
                live.add(self.droplet_of[x.id])
            if len(live) > g.storage_capacity:
                return False
        return True

    # ---- sources
    def source_of(
        self, i: int, op: int, t: int, st: ChipState
    ) -> tuple[RouteRequest, list[Body]] | str:
        """A route request carrying droplet i's share for `op`, from its stored holder, and the
        state changes the take makes (holder ended, remainder added); or a wait reason."""
        d = self.graph.droplets[i]
        u = self.take(i, op)
        h = self.held[i]
        holder = st.bodies[h.body]
        release = max(t + self.gp.split_ticks, holder.start + 1)
        anchor = holder.path[-1]
        bid = self.new_id()
        changes = [replace(holder, end=release)]
        if u < h.left:
            left = h.left - u
            rem = Body(
                self.new_id(),
                self.gp.footprint(left),
                left,
                release,
                (anchor,),
                INF,
                "stored",
                parent=holder.id,
                droplet=i,
            )
            changes.append(rem)
        req = RouteRequest(
            bid,
            self.gp.footprint(u),
            u,
            anchor,
            anchor,
            release,
            dest=op,
            parent=holder.id,
            droplet=i,
        )
        del d
        return req, changes

    def dispense(
        self, f: Op, t: int, port_use: dict[int, list[int]]
    ) -> tuple[int, int, Cell] | str:
        """(port index, dispense start, port cell) for dispense op f, or a wait reason."""
        assert f.fluid is not None
        cells = self.chip.ports.get(f.fluid)
        if not cells:
            raise ValueError(f"{self.chip.instance_id}: no port for fluid {f.fluid}")
        free = port_use.setdefault(f.fluid, list(self.ports.get(f.fluid, [0] * len(cells))))
        k = min(range(len(cells)), key=lambda j: (free[j], j))
        if free[k] > t:
            return "port"
        free[k] = t + f.duration
        return k, t, cells[k]

    # ---- commits
    def try_op(self, o: Op, t: int) -> str | None:
        """Commit `o` at decision tick t; None on success, else the wait reason."""
        if not self.storage_ok(o):
            return "storage"
        if o.kind is OpKind.MIX:
            return self.try_mix(o, t)
        if o.kind is OpKind.DETECT:
            return self.try_detect(o, t)
        return self.try_leave(o, t)

    def gather(
        self, o: Op, t: int, st: ChipState
    ) -> tuple[list[RouteRequest], dict[int, OpRecord], dict[int, list[int]]] | str:
        """Route requests for every input of `o` (sources only; goals set by the caller)."""
        reqs: list[RouteRequest] = []
        recs: dict[int, OpRecord] = {}
        port_use: dict[int, list[int]] = {}
        for i in self.inputs[o.id]:
            d = self.graph.droplets[i]
            prod = self.graph.ops[d.producer]
            if prod.kind is OpKind.DISPENSE:
                got = self.dispense(prod, t, port_use)
                if isinstance(got, str):
                    return got
                k, s, cell = got
                shape = self.gp.footprint(d.volume)
                src = self.chip.port_anchor(cell, shape)
                rel = s + prod.duration
                assert prod.fluid is not None
                recs[prod.id] = OpRecord(prod.id, t, s, rel, None, ("port", prod.fluid, k))
                reqs.append(
                    RouteRequest(
                        self.new_id(), shape, d.volume, src, src, rel, dest=o.id, droplet=i
                    )
                )
            else:
                got2 = self.source_of(i, o.id, t, st)
                if isinstance(got2, str):
                    return got2
                req, changes = got2
                for b in changes:
                    st.put(b)
                reqs.append(req)
        return reqs, recs, port_use

    def call_router(self, reqs: list[RouteRequest], st: ChipState) -> bool:
        self.route_calls += 1
        res = self.router.route(reqs, st)
        if not res.ok:
            self.route_failures += 1
            self.fail_reasons[res.reason.split(" within")[0]] += 1
        return res.ok

    def try_mix(self, o: Op, t: int) -> str | None:
        st = self.state.copy()
        got = self.gather(o, t, st)
        if isinstance(got, str):
            return got
        reqs, recs, port_use = got
        vol = self.graph.droplets[self.droplet_of[o.id]].volume
        shape, ticks = self.gp.mixer(vol)
        module = self.placer.place(shape, [r.source for r in reqs], st, t)
        if module is None:
            return "space"
        goal = (module.r, module.c)
        reqs = sorted(
            (replace(r, goal=goal) for r in reqs),
            key=lambda r: (-max(abs(r.source[0] - goal[0]), abs(r.source[1] - goal[1])), r.body_id),
        )
        st.regions.append(Region(module, t, INF, o.id, "module"))
        if not self.call_router(reqs, st):
            return "route"
        start = max(st.bodies[r.body_id].arrival for r in reqs)
        end = start + ticks
        for r in reqs:  # inputs are in the module at the start tick, then merge
            st.put(replace(st.bodies[r.body_id], end=start + 1, role="mix"))
        st.regions[-1] = Region(module, t, end, o.id, "module")
        i = self.droplet_of[o.id]
        prod = Body(
            self.new_id(), self.gp.footprint(vol), vol, end, (goal,), INF, "stored", droplet=i
        )
        st.put(prod)
        self.adopt(st, port_use)
        self.records.update(recs)
        self.records[o.id] = OpRecord(o.id, t, start, end, module)
        for i2 in self.inputs[o.id]:
            self.use(i2, o.id)
        self.held[i] = _Held(prod.id, vol)
        self.events.update({start, end, *(r.release for r in reqs)})
        return None

    def use(self, i: int, op: int) -> None:
        """Book droplet i's take by `op` (dispensed droplets are used whole)."""
        if i not in self.held:
            return
        h = self.held[i]
        u = self.take(i, op)
        h.left -= u
        if h.left == 0:
            del self.held[i]
        else:
            # the remainder body made by `source_of` holds it now
            rem = max(
                (
                    b
                    for b in self.state.bodies.values()
                    if b.droplet == i and b.role == "stored" and b.end == INF
                ),
                key=lambda b: b.start,
            )
            h.body = rem.id

    def try_detect(self, o: Op, t: int) -> str | None:
        options = []
        for k, rect in enumerate(self.chip.detectors):
            if self.det_free[k] > t:
                continue
            needs = self.det_last[k] is not None and self.det_last[k] != o.composition
            ready_at = self.det_free[k] + (self.gp.changeover_ticks if needs else 0)
            options.append((ready_at, needs, k, rect))
        if not options:
            return "detector"
        last = "detector"
        for ready_at, _, k, rect in sorted(options, key=lambda x: (x[0], x[1], x[2])):
            st = self.state.copy()
            got = self.gather(o, t, st)
            if isinstance(got, str):
                return got
            reqs, recs, port_use = got
            (req,) = reqs
            # the droplet covers the site's top-left cells, shifted in to stay on the chip
            goal = (
                min(rect.r, self.chip.height - req.shape[0]),
                min(rect.c, self.chip.width - req.shape[1]),
            )
            spot = _union(rect, Rect(goal[0], goal[1], *req.shape))
            if self.placer_blocked(spot, st, t):
                last = "space"
                continue
            st.regions.append(Region(spot, t, INF, o.id, "detector"))
            region_index = len(st.regions) - 1
            if not self.call_router([replace(req, goal=goal, role="detect")], st):
                last = "route"
                continue
            body = st.bodies[req.body_id]
            start = max(body.arrival, int(ready_at))
            end = start + o.duration
            self.adopt(st, port_use)
            self.records.update(recs)
            self.records[o.id] = OpRecord(o.id, t, start, end, spot, ("detector", k))
            for i in self.inputs[o.id]:
                self.use(i, o.id)
            self.det_free[k] = INF
            self.det_last[k] = o.composition
            self.removals.append(_Removal(body.id, end, k, region_index))
            self.events.update({start, end, req.release})
            return None
        return last

    def placer_blocked(self, rect: Rect, st: ChipState, t: int) -> bool:
        """Whether `rect` (grown by spacing) meets a region or a droplet planned at or after t."""
        sp = self.gp.spacing
        for reg in st.regions:
            if reg.t1 > t and reg.rect.gap(rect) < sp:
                return True
        for b in st.bodies.values():
            if b.end <= t:
                continue
            first = max(t, b.start) - b.start
            for p in set(b.path[min(first, len(b.path) - 1) :]):
                if Rect(p[0], p[1], *b.shape).gap(rect) < sp:
                    return True
        return False

    def try_leave(self, o: Op, t: int) -> str | None:
        st = self.state.copy()
        got = self.gather(o, t, st)
        if isinstance(got, str):
            return got
        reqs, recs, port_use = got
        (req,) = reqs
        st2 = self.to_waste(req, st)
        if st2 is None:
            return "route"
        body = st2.bodies[req.body_id]
        self.adopt(st2, port_use)
        self.records.update(recs)
        self.records[o.id] = OpRecord(o.id, t, req.release, req.release)
        for i in self.inputs[o.id]:
            self.use(i, o.id)
        self.events.update({req.release, body.arrival + 1})
        return None

    def to_waste(self, req: RouteRequest, st: ChipState) -> ChipState | None:
        """`st` with `req` routed to the nearest waste port that has a path, or None."""
        cells = sorted(
            self.chip.waste,
            key=lambda w: (
                max(
                    abs(self.chip.waste_anchor(w, req.shape)[0] - req.source[0]),
                    abs(self.chip.waste_anchor(w, req.shape)[1] - req.source[1]),
                ),
                w,
            ),
        )
        for cell in cells:
            trial = st.copy()
            goal = self.chip.waste_anchor(cell, req.shape)
            if self.call_router([replace(req, goal=goal, to_waste=True, dest=None)], trial):
                return trial
        return None

    def try_removal(self, rm: _Removal, t: int) -> bool:
        st = self.state.copy()
        body = st.bodies[rm.body]
        release = max(t, rm.ready)
        st.put(replace(body, end=release))
        if rm.region is not None:
            st.regions[rm.region] = replace(st.regions[rm.region], t1=release)
        req = RouteRequest(
            self.new_id(),
            body.shape,
            body.volume,
            body.path[-1],
            body.path[-1],
            release,
            parent=body.id,
            droplet=body.droplet,
        )
        st2 = self.to_waste(req, st)
        if st2 is None:
            return False
        self.state = st2
        if rm.detector is not None:
            self.det_free[rm.detector] = release
        self.events.update({release, st2.bodies[req.body_id].arrival + 1})
        return True

    def adopt(self, st: ChipState, port_use: dict[int, list[int]]) -> None:
        self.state = st
        for f, free in port_use.items():
            self.ports[f] = free

    # ---- loop
    def run(self) -> None:
        g = self.graph
        todo = [i for i in self.order if g.ops[i].kind is not OpKind.DISPENSE]
        t = 0
        while todo or self.removals:
            tried: dict[int, str] = {}
            progress = True
            while progress:
                progress = False
                for rm in list(self.removals):
                    if rm.ready <= t and self.try_removal(rm, t):
                        self.removals.remove(rm)
                        progress = True
                for oid in list(todo):
                    o = g.ops[oid]
                    if not self.ready(o, t):
                        continue
                    why = self.try_op(o, t)
                    if why is None:
                        todo.remove(oid)
                        tried.pop(oid, None)
                        progress = True
                    else:
                        tried[oid] = why
            if not todo and not self.removals:
                break
            later = [e for e in self.events if e > t]
            later += [rm.ready for rm in self.removals if rm.ready > t]
            if not later:
                waiting = {oid: why for oid, why in tried.items()}
                raise ExecutionDeadlock(
                    f"{g.instance_id}: nothing can move at tick {t}: {len(todo)} ops left, "
                    f"{len(self.held)} droplets stored, {len(self.removals)} droplets waiting to "
                    f"leave; ready ops wait for {waiting}"
                )
            nxt = min(later)
            for why in tried.values():
                self.waits[why] += nxt - t
            t = nxt


def _union(a: Rect, b: Rect) -> Rect:
    r, c = min(a.r, b.r), min(a.c, b.c)
    return Rect(r, c, max(a.r2, b.r2) - r + 1, max(a.c2, b.c2) - c + 1)


def priority_order(graph: OpGraph, gp: GridPlatform) -> tuple[list[int], int]:
    """The start order of the better M10 list schedule (14.9) and its makespan."""
    n_mix = sum(o.kind is OpKind.MIX for o in graph.ops)
    platform = gp.platform(mixers=max(n_mix, 1))
    best = None
    for priority in ListScheduler.PRIORITIES:
        s = ListScheduler(priority).schedule(graph, platform)
        m = schedule_metrics(s, graph).makespan
        if best is None or m < best[0]:
            best = (m, s)
    assert best is not None
    m, s = best
    order = sorted(range(len(graph.ops)), key=lambda i: (s.slots[i].start, i))
    return order, m


class GridExecutor:
    """List scheduling on the grid (14.9)."""

    def __init__(
        self, gp: GridPlatform, router: Router | None = None, placer: Placer | None = None
    ) -> None:
        self.gp = gp
        self.router = router or AStarRouter()
        self.placer = placer or NearestPlacer()

    def run(self, graph: OpGraph, chip: Chip) -> Run:
        order, sched = priority_order(graph, self.gp)
        ex = _Exec(self.gp, graph, chip, ChipState(chip, self.gp), self.router, self.placer, order)
        ex.run()
        recs = tuple(ex.records[o.id] for o in graph.ops)
        bodies = tuple(sorted(ex.state.bodies.values(), key=lambda b: b.id))
        ends = [r.end for r in recs] + [int(b.end) for b in bodies if b.end != INF]
        return Run(
            graph.instance_id,
            self.gp.name,
            recs,
            bodies,
            tuple(ex.state.regions),
            sched,
            max(ends, default=0),
            dict(ex.waits),
            ex.route_calls,
            ex.route_failures,
        )
