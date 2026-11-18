"""Router calls saved from executor runs, as a benchmark for learned routers (plan 8.6 step 4).

`RecordingRouter` wraps a router and, on every call the executor makes, saves the requests and
the chip state as it was before the call (`RouteCall`), with the wrapped router's answer as the
reference. A saved call is a routing problem on its own: rebuild the state with
`RouteCall.state(chip, gp)` and give it to any router with the same contract
(`route(requests, state)`, SPEC 14.8).

The saved state keeps only what can matter to the call: droplets still on the chip at the
earliest release, every body they or a request descend from (exemptions run along `parent`,
0.52), and regions not over by then. Replaying A* on a saved call gives the same paths as in
the run (tested), which checks that nothing that matters was dropped.

A benchmark file is JSON lines, one per executor run: the instance, the producer of the plan,
the platform (`gp_to_dict`) and the run's calls.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, fields
from typing import Any

from meda.pipeline.geometry import Chip, GridPlatform, Rect
from meda.pipeline.interfaces import Router
from meda.pipeline.router import INF, Body, ChipState, Region, RouteRequest, RouteResult


@dataclass(frozen=True, slots=True)
class RouteCall:
    """One router call: the requests, the state before it (bodies and regions; the chip and
    platform come from the run), and the reference router's result and wall time."""

    requests: tuple[RouteRequest, ...]
    bodies: tuple[Body, ...]
    regions: tuple[Region, ...]
    result: RouteResult
    seconds: float

    def state(self, chip: Chip, gp: GridPlatform) -> ChipState:
        return ChipState(chip, gp, {b.id: b for b in self.bodies}, list(self.regions))

    def to_json(self) -> dict[str, Any]:
        r = self.result
        return {
            "requests": [_req_json(q) for q in self.requests],
            "bodies": [_body_json(b) for b in self.bodies],
            "regions": [_region_json(g) for g in self.regions],
            "result": {
                "ok": r.ok,
                "paths": [[list(c) for c in p] for p in r.paths],
                "failed": r.failed,
                "ticks": r.ticks,
                "reason": r.reason,
                "shapes": [[list(s) for s in p] for p in r.shapes],
            },
            "seconds": self.seconds,
        }

    @staticmethod
    def from_json(d: dict[str, Any]) -> RouteCall:
        r = d["result"]
        return RouteCall(
            tuple(_req(q) for q in d["requests"]),
            tuple(_body(b) for b in d["bodies"]),
            tuple(_region(g) for g in d["regions"]),
            RouteResult(
                r["ok"],
                tuple(tuple(_cell(c) for c in p) for p in r["paths"]),
                r["failed"],
                r["ticks"],
                r["reason"],
                tuple(tuple(_cell(s) for s in p) for p in r["shapes"]),
            ),
            d["seconds"],
        )


class RecordingRouter:
    """Wraps `inner`; saves every call in `calls` (state pruned as the module says)."""

    def __init__(self, inner: Router) -> None:
        self.inner = inner
        self.name = f"recording_{inner.name}"
        self.calls: list[RouteCall] = []

    def route(self, requests: list[RouteRequest], state: ChipState) -> RouteResult:
        bodies, regions = relevant(requests, state)
        t = time.perf_counter()
        res = self.inner.route(requests, state)
        dt = time.perf_counter() - t
        self.calls.append(RouteCall(tuple(requests), bodies, regions, res, dt))
        return res


def relevant(
    requests: list[RouteRequest], state: ChipState
) -> tuple[tuple[Body, ...], tuple[Region, ...]]:
    """The bodies and regions a call to route `requests` can depend on."""
    if not requests:
        return (), ()
    t0 = min(q.release for q in requests)
    keep = {b.id for b in state.bodies.values() if b.end >= t0}
    todo = [b.parent for b in state.bodies.values() if b.id in keep]
    todo += [q.parent for q in requests]
    while todo:
        p = todo.pop()
        if p is not None and p not in keep and p in state.bodies:
            keep.add(p)
            todo.append(state.bodies[p].parent)
    bodies = tuple(sorted((state.bodies[i] for i in keep), key=lambda b: b.id))
    regions = tuple(g for g in state.regions if g.t1 >= t0)
    return bodies, regions


def gp_to_dict(gp: GridPlatform) -> dict[str, Any]:
    """The platform as `GridPlatform.from_dict` reads it."""
    out: dict[str, Any] = {}
    for f in fields(GridPlatform):
        v = getattr(gp, f.name)
        if f.name in ("footprints", "mixer_library"):
            v = [{"max_volume": x.max_volume, "shape": list(x.shape), "units": x.units} for x in v]
        elif f.name == "dead_cells":
            v = [list(c) for c in v]
        elif isinstance(v, tuple):
            v = list(v)
        out[f.name] = v
    return out


# ---- JSON for the router's dataclasses (INF as null, cells and shapes as lists)


def _cell(x: list[int]) -> tuple[int, int]:
    return (int(x[0]), int(x[1]))


def _end(x: float) -> float | int | None:
    return None if x == INF else int(x)


def _req_json(q: RouteRequest) -> dict[str, Any]:
    return {
        "body_id": q.body_id,
        "shape": list(q.shape),
        "volume": q.volume,
        "source": list(q.source),
        "goal": list(q.goal),
        "release": q.release,
        "role": q.role,
        "dest": q.dest,
        "parent": q.parent,
        "droplet": q.droplet,
        "to_waste": q.to_waste,
    }


def _req(d: dict[str, Any]) -> RouteRequest:
    return RouteRequest(
        d["body_id"],
        _cell(d["shape"]),
        d["volume"],
        _cell(d["source"]),
        _cell(d["goal"]),
        d["release"],
        d["role"],
        d["dest"],
        d["parent"],
        d["droplet"],
        d["to_waste"],
    )


def _body_json(b: Body) -> dict[str, Any]:
    return {
        "id": b.id,
        "shape": list(b.shape),
        "volume": b.volume,
        "start": b.start,
        "path": [list(c) for c in b.path],
        "end": _end(b.end),
        "role": b.role,
        "dest": b.dest,
        "parent": b.parent,
        "droplet": b.droplet,
        "to_waste": b.to_waste,
        "shapes": [list(s) for s in b.shapes],
    }


def _body(d: dict[str, Any]) -> Body:
    return Body(
        d["id"],
        _cell(d["shape"]),
        d["volume"],
        d["start"],
        tuple(_cell(c) for c in d["path"]),
        INF if d["end"] is None else d["end"],
        d["role"],
        d["dest"],
        d["parent"],
        d["droplet"],
        d["to_waste"],
        tuple(_cell(s) for s in d["shapes"]),
    )


def _region_json(g: Region) -> dict[str, Any]:
    r = g.rect
    return {
        "rect": [r.r, r.c, r.h, r.w],
        "t0": g.t0,
        "t1": _end(g.t1),
        "owner": g.owner,
        "kind": g.kind,
    }


def _region(d: dict[str, Any]) -> Region:
    return Region(
        Rect(*d["rect"]), d["t0"], INF if d["t1"] is None else d["t1"], d["owner"], d["kind"]
    )
