"""Droplets, reservations and the classical router of the grid harness (SPEC 14.8, M11).

The router contract is the doc's section 8, fixed once so a learned router or the supervisor's
Dijkstra router can replace the classical one: `route(requests, state) -> RouteResult`. A call
either reserves a path for every request (adds a `Body` per request to `state`) or reserves
nothing.

`ChipState` is everything already planned: every droplet's trajectory (`Body`, one anchor per
tick from `start`, then held at the last anchor until `end`), the module and detector
reservations (`Region`), the dead cells. Times are integer ticks; `INF` means "until changed".

Spacing (14.6), the same rule as `run_check`: a droplet at tick t keeps `spacing` empty cells
from every other droplet's footprint at ticks u = t - 1, t and t + 1 (exemptions judged at
min(t, u)), and stays out of every active
region grown by `spacing`. Exempt: droplets bound for the same operation (they merge there), a
droplet and its own region, and two pieces of one droplet during the first `split_window` ticks
of the later one.

`AStarRouter` is prioritised space-time A* (cooperative A*): requests in the given order, each
with every earlier path reserved; moves to the 8 neighbours or wait; a move takes
`GridPlatform.move_ticks` (by shape and direction, L-21), holding the old anchor until it lands;
the heuristic is the static shortest distance in moves (dead cells and droplets parked for
good) times the cheapest move, which is admissible. A request fails if its goal is not reached
within `GridPlatform.horizon_for(shape)` ticks of its release.

`FreeRouter` is a measuring stick: shortest paths on the empty chip, every other droplet and
module ignored, never fails. Running the executor with it gives the makespan a perfect router
could reach with the same placement and order (the routing gap, D-89).
"""

from __future__ import annotations

import heapq
import math
from dataclasses import dataclass, field, replace
from itertools import pairwise

import numpy as np

from meda.pipeline.geometry import Cell, Chip, GridPlatform, Rect, Shape, at, window_any

INF = math.inf
MOVES = [(dr, dc) for dr in (-1, 0, 1) for dc in (-1, 0, 1)]


@dataclass(frozen=True, slots=True)
class Body:
    """One physical droplet. `path[i]` is its anchor at tick `start + i`; after the last entry
    it holds there until `end` (exclusive; `INF` while it stays). `dest`: the operation it is
    travelling into (exempt from that op's region and from the op's other inputs). `parent`:
    the body that ended where this one starts (a take, a continuation). `droplet`: the
    opgraph droplet it carries (index), if any. `role`: "move", "stored", "detect" or "mix"
    (inputs merged in a module)."""

    id: int
    shape: Shape
    volume: int
    start: int
    path: tuple[Cell, ...]
    end: float
    role: str
    dest: int | None = None
    parent: int | None = None
    droplet: int | None = None
    to_waste: bool = False

    def pos(self, t: int) -> Cell | None:
        if t < self.start or t >= self.end:
            return None
        return self.path[min(t - self.start, len(self.path) - 1)]

    def rect(self, t: int) -> Rect | None:
        p = self.pos(t)
        return None if p is None else at(p, self.shape)

    @property
    def arrival(self) -> int:
        return self.start + len(self.path) - 1


def split_window(shape: Shape, gp: GridPlatform) -> int:
    """Ticks a part is exempt from the droplet it was split from: the time to move
    `max(shape) + spacing + 1` cells at its slowest move (0.49: big droplets separate slower)."""
    slowest = max(gp.move_ticks(shape, dr, dc) for dr, dc in MOVES)
    return (max(shape) + gp.spacing + 1) * slowest


def exempt_until(a: Body, b: Body, gp: GridPlatform) -> float:
    """The pair is exempt from spacing at ticks before this (14.6): for ever if bound for the
    same op (they merge there); for two pieces of one droplet (a stored droplet, its remainder,
    its parts, a droplet leaving a detector) during the first `split_window` ticks of the later
    one, while they separate; else never."""
    if a.dest is not None and a.dest == b.dest:
        return INF
    same = (
        (a.droplet is not None and a.droplet == b.droplet) or a.parent == b.id or b.parent == a.id
    )
    if not same:
        return -INF
    window = max(split_window(a.shape, gp), split_window(b.shape, gp))
    return max(a.start, b.start) + window


def exempt(a: Body, b: Body, t: int, gp: GridPlatform) -> bool:
    """Whether the pair is exempt from spacing at tick `t` (see `exempt_until`)."""
    return t < exempt_until(a, b, gp)


@dataclass(frozen=True, slots=True)
class Region:
    """A reserved rectangle (a module or a detection site) from `t0` until `t1` (exclusive).
    Droplets other than those bound for `owner` keep `spacing` cells away from it."""

    rect: Rect
    t0: int
    t1: float
    owner: int
    kind: str

    def active(self, t: int) -> bool:
        return self.t0 <= t < self.t1


@dataclass(slots=True)
class ChipState:
    """Everything planned so far. `copy()` is cheap: bodies and regions are immutable, so a
    scratch copy can be changed and dropped (all-or-nothing commits)."""

    chip: Chip
    gp: GridPlatform
    bodies: dict[int, Body] = field(default_factory=dict)
    regions: list[Region] = field(default_factory=list)

    def copy(self) -> ChipState:
        return ChipState(self.chip, self.gp, dict(self.bodies), list(self.regions))

    def put(self, body: Body) -> None:
        self.bodies[body.id] = body

    def end_body(self, body_id: int, t: int) -> None:
        self.bodies[body_id] = replace(self.bodies[body_id], end=t)

    def static_grid(self) -> np.ndarray:
        g = np.zeros((self.chip.height, self.chip.width), dtype=bool)
        for r, c in self.chip.dead:
            g[r, c] = True
        return g

    def paint(self, g: np.ndarray, rect: Rect) -> None:
        r0, c0 = max(rect.r, 0), max(rect.c, 0)
        g[r0 : rect.r2 + 1, c0 : rect.c2 + 1] = True


@dataclass(frozen=True, slots=True)
class RouteRequest:
    """Move a new droplet of `shape` from anchor `source` (on chip from tick `release`) to
    anchor `goal`. `dest`: the op it feeds (exemptions); `parent`: the body it is split from;
    `to_waste`: it leaves the chip on arrival (else it holds at the goal until changed)."""

    body_id: int
    shape: Shape
    volume: int
    source: Cell
    goal: Cell
    release: int
    role: str = "move"
    dest: int | None = None
    parent: int | None = None
    droplet: int | None = None
    to_waste: bool = False


@dataclass(frozen=True, slots=True)
class RouteResult:
    """`paths[i]` for request i (empty if the call failed); `failed`: index of the first
    request without a path; `ticks`: the latest arrival minus the earliest release."""

    ok: bool
    paths: tuple[tuple[Cell, ...], ...]
    failed: int | None
    ticks: int
    reason: str = ""


class AStarRouter:
    """Prioritised space-time A* (14.8)."""

    name = "astar"

    def route(self, requests: list[RouteRequest], state: ChipState) -> RouteResult:
        scratch = state.copy()
        paths: list[tuple[Cell, ...]] = []
        for i, req in enumerate(requests):
            path, why = self._one(req, scratch)
            if path is None:
                return RouteResult(False, (), i, 0, why)
            paths.append(path)
            _reserve(req, path, scratch)
        state.bodies = scratch.bodies
        state.regions = scratch.regions
        return _done(requests, paths)

    # ---- one request
    def _one(self, req: RouteRequest, state: ChipState) -> tuple[tuple[Cell, ...] | None, str]:
        chip, gp = state.chip, state.gp
        h, w = req.shape
        rows, cols = chip.height - h + 1, chip.width - w + 1
        for name, (r, c) in (("source", req.source), ("goal", req.goal)):
            if not (0 <= r < rows and 0 <= c < cols):
                raise ValueError(
                    f"{chip.instance_id}: {name} {(r, c)} of a {h}x{w} droplet is off the chip"
                )
        probe = Body(
            req.body_id,
            req.shape,
            req.volume,
            req.release,
            (req.source,),
            INF,
            req.role,
            req.dest,
            req.parent,
            req.droplet,
        )
        # droplets still on the chip from the tick before the release, with their exemptions
        others = [b for b in state.bodies.values() if b.id != req.body_id and b.end >= req.release]
        until = {b.id: exempt_until(probe, b, gp) for b in others}
        regions = [g for g in state.regions if g.owner != req.dest]
        static = state.static_grid()
        # obstacles that stay for good from the release on: the heuristic and a fast check
        parked = static.copy()
        for b in others:
            if b.end == INF and b.arrival <= req.release and req.release >= until[b.id]:
                state.paint(parked, at(b.path[-1], b.shape).inflate(gp.spacing))
        for g in regions:
            if g.t0 <= req.release and g.t1 == INF:
                state.paint(parked, g.rect.inflate(gp.spacing))
        parked_bad = window_any(parked, req.shape, 0)
        dist = _distances(parked_bad, req.goal)
        if dist[req.source] < 0:
            return None, "no path past parked droplets"
        cache: dict[int, np.ndarray] = {}
        fixed = window_any(static, req.shape, 0)

        def region_bad(t: int) -> np.ndarray:
            g = np.zeros_like(static)
            for reg in regions:
                if reg.active(t):
                    state.paint(g, reg.rect.inflate(gp.spacing))
            return window_any(g, req.shape, 0) | fixed

        def bad(t: int) -> np.ndarray:
            if t not in cache:
                occ = np.zeros_like(static)
                for b in others:
                    if b.end < t - 1 or b.start > t + 1:
                        continue
                    for u in (t - 1, t, t + 1):
                        if min(t, u) < until[b.id]:
                            continue
                        rect = b.rect(u)
                        if rect is not None:
                            state.paint(occ, rect)
                cache[t] = window_any(occ, req.shape, gp.spacing) | region_bad(t)
            return cache[t]

        t0 = req.release
        if bad(t0)[req.source]:
            return None, "source blocked at release"
        horizon = gp.horizon_for(req.shape)
        limit = t0 + horizon
        cost = {m: gp.move_ticks(req.shape, *m) for m in MOVES}
        step = min(k for m, k in cost.items() if m != (0, 0))  # admissible: moves x cheapest
        start = (req.source[0], req.source[1], t0)
        heap: list[tuple[int, int, int, int, int]] = [
            (int(dist[req.source]) * step, int(dist[req.source]), t0, *req.source)
        ]
        came: dict[tuple[int, int, int], tuple[int, int, int] | None] = {start: None}
        while heap:
            _, _, t, r, c = heapq.heappop(heap)
            if (r, c) == req.goal:
                return _unroll((r, c, t), came), ""
            if t >= limit:
                continue
            for dr, dc in MOVES:
                nr, nc = r + dr, c + dc
                if not (0 <= nr < rows and 0 <= nc < cols):
                    continue
                if dr and dc and (fixed[r + dr, c] or fixed[r, c + dc]):
                    continue  # no diagonal past a dead cell
                k = cost[(dr, dc)]
                key = (nr, nc, t + k)
                if key in came or dist[nr, nc] < 0 or bad(t + k)[nr, nc]:
                    continue
                # a k-tick move holds the droplet where it is until it lands
                if any(bad(u)[r, c] for u in range(t + 1, t + k)):
                    continue
                came[key] = (r, c, t)
                hv = int(dist[nr, nc])
                heapq.heappush(heap, (t + k - t0 + hv * step, hv, t + k, nr, nc))
        return None, f"no path within {horizon} ticks"


def _unroll(
    node: tuple[int, int, int], came: dict[tuple[int, int, int], tuple[int, int, int] | None]
) -> tuple[Cell, ...]:
    """One anchor per tick from the search tree: a k-tick move holds the old anchor k - 1
    ticks, then lands."""
    nodes: list[tuple[int, int, int]] = []
    cur: tuple[int, int, int] | None = node
    while cur is not None:
        nodes.append(cur)
        cur = came[cur]
    nodes.reverse()
    path: list[Cell] = [(nodes[0][0], nodes[0][1])]
    for a, b in pairwise(nodes):
        path += [(a[0], a[1])] * (b[2] - a[2] - 1) + [(b[0], b[1])]
    return tuple(path)


def _reserve(req: RouteRequest, path: tuple[Cell, ...], state: ChipState) -> None:
    """Add the body that carries `req` along `path` to `state`."""
    end: float = req.release + len(path) if req.to_waste else INF
    state.put(
        Body(
            req.body_id,
            req.shape,
            req.volume,
            req.release,
            path,
            end,
            req.role,
            req.dest,
            req.parent,
            req.droplet,
            req.to_waste,
        )
    )


def _done(requests: list[RouteRequest], paths: list[tuple[Cell, ...]]) -> RouteResult:
    ticks = (
        max(r.release + len(p) - 1 for r, p in zip(requests, paths, strict=True))
        - min(r.release for r in requests)
        if requests
        else 0
    )
    return RouteResult(True, tuple(paths), None, ticks)


def free_distances(chip: Chip, gp: GridPlatform, shape: Shape, goal: Cell) -> np.ndarray:
    """Ticks from every anchor of a `shape` droplet to `goal` on the empty chip, moves costing
    `gp.move_ticks`: dead cells are the only obstacles (no droplets, no modules); -1 if
    unreachable. A lower bound on any real trip between the two anchors."""
    static = np.zeros((chip.height, chip.width), dtype=bool)
    for r, c in chip.dead:
        static[r, c] = True
    blocked = window_any(static, shape, 0)
    cost = {m: gp.move_ticks(shape, *m) for m in MOVES if m != (0, 0)}
    if len(set(cost.values())) == 1:  # every move the same: breadth-first, scaled
        d = _distances(blocked, goal)
        return np.where(d < 0, -1, d * next(iter(cost.values())))
    return _weighted_distances(blocked, goal, cost)


def _weighted_distances(
    blocked: np.ndarray, goal: Cell, cost: dict[tuple[int, int], int]
) -> np.ndarray:
    """Dijkstra from `goal` over the 8 neighbours with move costs `cost` (symmetric in
    direction); the goal is never blocked; -1 if unreachable."""
    rows, cols = blocked.shape
    dist = np.full((rows, cols), -1, dtype=np.int64)
    best = {goal: 0}
    heap = [(0, *goal)]
    while heap:
        d, r, c = heapq.heappop(heap)
        if dist[r, c] >= 0:
            continue
        dist[r, c] = d
        for (dr, dc), k in cost.items():
            nr, nc = r + dr, c + dc
            if (
                0 <= nr < rows
                and 0 <= nc < cols
                and not blocked[nr, nc]
                and dist[nr, nc] < 0
                and d + k < best.get((nr, nc), 1 << 62)
            ):
                best[(nr, nc)] = d + k
                heapq.heappush(heap, (d + k, nr, nc))
    return dist


class FreeRouter:
    """A measuring stick, not a router a chip could use (14.8): every droplet takes a fastest
    path on the empty chip (`free_distances`), ignoring every other droplet, module and the
    spacing rule, and never fails. A run with it is the run a perfect router would give with the
    same placement and order, minus the space a path needs; its paths overlap, so such a run
    does not pass `validate_run`. Ties between moves: the first of `MOVES`.

    D-91, the headroom of shape actions (still free of conflicts, so an optimistic bound):
    `turn` lets a droplet rotate on the way (`turn_distance`) and arrive in its own shape;
    `split` lets it split at the source into 1 x 1 pieces that travel at one tick a cell and
    merge at the goal, `ceil(log2(volume))` rounds of `split_ticks` each way (binary splits in
    parallel). Each trip takes the fastest allowed option; when that is not the plain path, the
    body holds at the source and then steps straight to the goal so that it arrives at that
    tick (only the arrival time is meaningful)."""

    def __init__(self, turn: bool = False, split: bool = False) -> None:
        self.turn, self.split = turn, split
        self.name = "free" + ("_turn" if turn else "") + ("_split" if split else "")
        self._cache: dict[tuple[str, str, Shape, Cell], np.ndarray] = {}

    def _dist(self, kind: str, state: ChipState, shape: Shape, goal: Cell) -> np.ndarray:
        key = (kind, state.chip.instance_id, shape, goal)
        if key not in self._cache:
            if kind == "turn":
                self._cache[key] = turn_distances(state.chip, state.gp, shape, goal)
            else:
                self._cache[key] = free_distances(state.chip, state.gp, shape, goal)
        return self._cache[key]

    def route(self, requests: list[RouteRequest], state: ChipState) -> RouteResult:
        gp = state.gp
        paths: list[tuple[Cell, ...]] = []
        for req in requests:
            dist = self._dist("move", state, req.shape, req.goal)
            if dist[req.source] < 0:
                raise ValueError(
                    f"{state.chip.instance_id}: goal {req.goal} unreachable from {req.source} "
                    "on the empty chip"
                )
            best = int(dist[req.source])
            if self.turn and req.shape[0] != req.shape[1]:
                td = int(self._dist("turn", state, req.shape, req.goal)[req.source])
                if 0 <= td < best:
                    best = td
            if self.split and req.volume > 1:
                d1 = int(self._dist("move", state, (1, 1), req.goal)[req.source])
                rounds = math.ceil(math.log2(req.volume))
                if d1 >= 0 and d1 + 2 * rounds * gp.split_ticks < best:
                    best = d1 + 2 * rounds * gp.split_ticks
            if best == int(dist[req.source]):
                path = self._descend(req, dist, gp)
            else:
                path = _straight(req.source, req.goal, best)
            paths.append(path)
            _reserve(req, path, state)
        return _done(requests, paths)

    @staticmethod
    def _descend(req: RouteRequest, dist: np.ndarray, gp: GridPlatform) -> tuple[Cell, ...]:
        path = [req.source]
        cur = req.source
        while cur != req.goal:
            r, c = cur
            d = dist[r, c]
            k, nxt = next(
                (gp.move_ticks(req.shape, dr, dc), (r + dr, c + dc))
                for dr, dc in MOVES
                if (dr, dc) != (0, 0)
                and 0 <= r + dr < dist.shape[0]
                and 0 <= c + dc < dist.shape[1]
                and dist[r + dr, c + dc] >= 0
                and dist[r + dr, c + dc] == d - gp.move_ticks(req.shape, dr, dc)
            )
            path += [cur] * (k - 1) + [nxt]
            cur = nxt
        return tuple(path)


def _straight(source: Cell, goal: Cell, ticks: int) -> tuple[Cell, ...]:
    """Hold at `source`, then one cell a tick straight (Chebyshev) to `goal`, arriving after
    `ticks` ticks."""
    steps: list[Cell] = []
    r, c = source
    while (r, c) != goal:
        r += (goal[0] > r) - (goal[0] < r)
        c += (goal[1] > c) - (goal[1] < c)
        steps.append((r, c))
    if ticks < len(steps):
        raise ValueError(f"{ticks} ticks are fewer than the {len(steps)} cells to the goal")
    return (source,) * (ticks - len(steps) + 1) + tuple(steps)


def turn_distances(chip: Chip, gp: GridPlatform, shape: Shape, goal: Cell) -> np.ndarray:
    """Ticks from every anchor of a `shape` droplet to `goal` on the empty chip when it may
    rotate on the way (`morph_ticks`, keeping its anchor) and must arrive as `shape`: Dijkstra
    over (anchor, orientation); -1 if unreachable. Shape of the result: the anchors of
    `shape`."""
    h, w = shape
    shapes = (shape, (w, h))
    static = np.zeros((chip.height, chip.width), dtype=bool)
    for r, c in chip.dead:
        static[r, c] = True
    blocked = []
    for shp in shapes:
        b = np.ones((chip.height, chip.width), dtype=bool)
        win = window_any(static, shp, 0)
        b[: win.shape[0], : win.shape[1]] = win
        blocked.append(b)
    cost = [{m: gp.move_ticks(shp, *m) for m in MOVES if m != (0, 0)} for shp in shapes]
    morph = gp.morph_ticks(shape)
    dist = np.full((2, chip.height, chip.width), -1, dtype=np.int64)
    best = {(0, *goal): 0}
    heap = [(0, 0, *goal)]
    while heap:
        d, o, r, c = heapq.heappop(heap)
        if dist[o, r, c] >= 0:
            continue
        dist[o, r, c] = d
        nexts = [(1 - o, r, c, morph)] if not blocked[1 - o][r, c] else []
        nexts += [(o, r + dr, c + dc, k) for (dr, dc), k in cost[o].items()]
        for o2, nr, nc, k in nexts:
            if (
                0 <= nr < chip.height
                and 0 <= nc < chip.width
                and not blocked[o2][nr, nc]
                and dist[o2, nr, nc] < 0
                and d + k < best.get((o2, nr, nc), 1 << 62)
            ):
                best[(o2, nr, nc)] = d + k
                heapq.heappush(heap, (d + k, o2, nr, nc))
    rows, cols = chip.height - h + 1, chip.width - w + 1
    return dist[0, :rows, :cols]


def _distances(blocked: np.ndarray, goal: Cell) -> np.ndarray:
    """Moves (8 directions) from every anchor to `goal` avoiding `blocked`; -1 if unreachable.
    The goal itself is never blocked here (a droplet may always finish there). Breadth-first,
    one whole-grid step per distance."""
    rows, cols = blocked.shape
    dist = np.full((rows, cols), -1, dtype=np.int32)
    free = ~blocked
    free[goal] = True
    reached = np.zeros((rows, cols), dtype=bool)
    reached[goal] = True
    front = reached.copy()
    dist[goal] = 0
    d = 0
    pad = np.zeros((rows + 2, cols + 2), dtype=bool)
    while front.any():
        d += 1
        pad[1:-1, 1:-1] = front
        grown = np.zeros_like(front)
        for dr in (0, 1, 2):
            for dc in (0, 1, 2):
                grown |= pad[dr : dr + rows, dc : dc + cols]
        grown &= free & ~reached
        dist[grown] = d
        reached |= grown
        front = grown
    return dist
