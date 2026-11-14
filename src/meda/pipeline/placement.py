"""Module placement for the grid harness (SPEC 14.7, M11).

`NearestPlacer.place(shape, sources, state, t)` returns the module rectangle for a mix that
starts being served at tick `t`, or None if no position is free. A position is free if the
module grown by its guard ring (`spacing` cells) lies inside the chip's interior (outside the
edge band), covers no dead cell, keeps `spacing` cells from every region active at or after
`t`, and from every droplet position planned at or after `t` (moves and holds). Among free
positions it takes the one with the smallest total Chebyshev distance from the op's input
anchors; ties: the smallest (row, col).
"""

from __future__ import annotations

import numpy as np

from meda.pipeline.geometry import Cell, Rect, Shape, at, window_any
from meda.pipeline.router import ChipState


class NearestPlacer:
    name = "nearest"

    def place(self, shape: Shape, sources: list[Cell], state: ChipState, t: int) -> Rect | None:
        chip, sp = state.chip, state.gp.spacing
        h, w = shape
        busy = state.static_grid()
        for reg in state.regions:
            if reg.t1 > t:
                state.paint(busy, reg.rect.inflate(sp))
        for b in state.bodies.values():
            if b.end <= t:
                continue
            first = max(t, b.start) - b.start
            for p in set(b.path[min(first, len(b.path) - 1) :]):
                state.paint(busy, at(p, b.shape))
        bad = window_any(busy, shape, sp)
        inner = chip.interior
        ok = np.zeros_like(bad)
        r0, r1 = inner.r + sp, inner.r2 - sp - h + 1
        c0, c1 = inner.c + sp, inner.c2 - sp - w + 1
        if r1 < r0 or c1 < c0:
            return None
        ok[r0 : r1 + 1, c0 : c1 + 1] = True
        ok &= ~bad
        if not ok.any():
            return None
        rr, cc = np.indices(ok.shape)
        cost = np.zeros(ok.shape, dtype=np.int64)
        for sr, sc in sources:
            cost += np.maximum(np.abs(rr - sr), np.abs(cc - sc))
        cost[~ok] = np.iinfo(np.int64).max
        k = int(np.argmin(cost))  # row-major: ties go to the smallest (row, col)
        r, c = divmod(k, ok.shape[1])
        return Rect(r, c, h, w)
