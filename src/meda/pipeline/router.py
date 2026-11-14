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
with every earlier path reserved; moves to the 8 neighbours or wait, one cell per tick; the
heuristic is the static shortest distance (dead cells and droplets parked for good), which is
admissible. A request fails if its goal is not reached within `GridPlatform.horizon` ticks of
its release.
"""

from __future__ import annotations

import heapq
import math
from dataclasses import dataclass, field, replace

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


def split_window(shape: Shape, spacing: int) -> int:
    """Ticks a part is exempt from the droplet it was split from."""
    return max(shape) + spacing + 1


def exempt_until(a: Body, b: Body, spacing: int) -> float:
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
    window = max(split_window(a.shape, spacing), split_window(b.shape, spacing))
    return max(a.start, b.start) + window


def exempt(a: Body, b: Body, t: int, spacing: int) -> bool:
    """Whether the pair is exempt from spacing at tick `t` (see `exempt_until`)."""
    return t < exempt_until(a, b, spacing)


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
            end: float = req.release + len(path) if req.to_waste else INF
            scratch.put(
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
        state.bodies = scratch.bodies
        state.regions = scratch.regions
        ticks = (
            max(r.release + len(p) - 1 for r, p in zip(requests, paths, strict=True))
            - min(r.release for r in requests)
            if requests
            else 0
        )
        return RouteResult(True, tuple(paths), None, ticks)

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
        until = {b.id: exempt_until(probe, b, gp.spacing) for b in others}
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
        limit = t0 + gp.horizon
        start = (req.source[0], req.source[1], t0)
        heap: list[tuple[int, int, int, int, int]] = [
            (int(dist[req.source]), int(dist[req.source]), t0, *req.source)
        ]
        came: dict[tuple[int, int, int], tuple[int, int, int] | None] = {start: None}
        while heap:
            _, _, t, r, c = heapq.heappop(heap)
            if (r, c) == req.goal:
                path: list[Cell] = []
                node: tuple[int, int, int] | None = (r, c, t)
                while node is not None:
                    path.append((node[0], node[1]))
                    node = came[node]
                return tuple(reversed(path)), ""
            if t >= limit:
                continue
            nb = bad(t + 1)
            for dr, dc in MOVES:
                nr, nc = r + dr, c + dc
                if not (0 <= nr < rows and 0 <= nc < cols) or nb[nr, nc]:
                    continue
                if dr and dc and (fixed[r + dr, c] or fixed[r, c + dc]):
                    continue  # no diagonal past a dead cell
                key = (nr, nc, t + 1)
                if key in came or dist[nr, nc] < 0:
                    continue
                came[key] = (r, c, t)
                hv = int(dist[nr, nc])
                heapq.heappush(heap, (t + 1 - t0 + hv, hv, t + 1, nr, nc))
        return None, f"no path within {gp.horizon} ticks"


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
