"""Chip geometry for the grid harness (SPEC 14.6, milestone M11).

Unit-cell convention: one cell holds one volume unit of the DAG, so a droplet of volume v covers
a rectangle of about v cells (`GridPlatform.footprint`). Cells are `(row, col)`, row 0 at the
north edge, col 0 at the west edge. Every size and position comes from the grid platform config
(`configs/platform/meda_grid.yaml`); `layout` places ports, detectors and waste ports for one
instance's fluids, deterministically.

Spacing: two droplets keep at least `spacing` empty cells between their footprints (Chebyshev
gap, `Rect.gap`), and a droplet keeps out of every module and its guard ring of `spacing` cells.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, fields
from itertools import pairwise
from pathlib import Path
from typing import Any

import numpy as np
from omegaconf import OmegaConf

from meda.core.problem import Instance
from meda.pipeline.platform import PLATFORM_DIR, Platform

Cell = tuple[int, int]
Shape = tuple[int, int]


@dataclass(frozen=True, slots=True, order=True)
class Rect:
    """`h` rows by `w` cols with top-left cell `(r, c)`."""

    r: int
    c: int
    h: int
    w: int

    @property
    def r2(self) -> int:
        return self.r + self.h - 1

    @property
    def c2(self) -> int:
        return self.c + self.w - 1

    def cells(self) -> list[Cell]:
        return [
            (i, j) for i in range(self.r, self.r + self.h) for j in range(self.c, self.c + self.w)
        ]

    def gap(self, other: Rect) -> int:
        """Empty cells between the two rectangles (Chebyshev); -1 if they overlap."""
        dr = max(other.r - self.r2 - 1, self.r - other.r2 - 1)
        dc = max(other.c - self.c2 - 1, self.c - other.c2 - 1)
        return max(dr, dc) if max(dr, dc) >= 0 else -1

    def inflate(self, k: int) -> Rect:
        return Rect(self.r - k, self.c - k, self.h + 2 * k, self.w + 2 * k)

    def contains(self, other: Rect) -> bool:
        return (
            self.r <= other.r and other.r2 <= self.r2 and self.c <= other.c and other.c2 <= self.c2
        )

    def covers(self, cell: Cell) -> bool:
        return self.r <= cell[0] <= self.r2 and self.c <= cell[1] <= self.c2


def at(anchor: Cell, shape: Shape) -> Rect:
    return Rect(anchor[0], anchor[1], shape[0], shape[1])


@dataclass(frozen=True, slots=True)
class SizeRule:
    """Volumes up to `max_volume` (after the previous rule) get `shape`; a mixer also takes
    `units` library time units."""

    max_volume: int
    shape: Shape
    units: int = 0


@dataclass(frozen=True, slots=True)
class GridPlatform:
    """The grid harness platform (14.6-14.10). Durations in ticks are library units times
    `unit_ticks`; 1 tick = one move of a 1 x 1 droplet by one cell."""

    name: str
    grid: Shape
    spacing: int
    edge_margin: int
    ports_per_fluid: int
    detectors: int
    detector_shape: Shape
    waste_ports: int
    dead_cells: tuple[Cell, ...]
    mixers: int | None
    storage_capacity: int | None
    unit_ticks: int
    dispense_units: int
    detect_units: int
    changeover_units: int
    split_ticks: int
    route_horizon: int | None
    footprints: tuple[SizeRule, ...]
    mixer_library: tuple[SizeRule, ...]

    def __post_init__(self) -> None:
        def bad(msg: str) -> None:
            raise ValueError(f"grid platform '{self.name}': {msg}")

        h, w = self.grid
        if h < 5 or w < 5:
            bad(f"grid {self.grid} is smaller than 5 x 5")
        for name, low in (
            ("spacing", 0),
            ("edge_margin", 0),
            ("ports_per_fluid", 1),
            ("detectors", 1),
            ("waste_ports", 1),
            ("unit_ticks", 1),
            ("dispense_units", 0),
            ("detect_units", 1),
            ("changeover_units", 0),
            ("split_ticks", 0),
        ):
            v = getattr(self, name)
            if not isinstance(v, int) or isinstance(v, bool) or v < low:
                bad(f"{name} must be an int >= {low}")
        for name in ("mixers", "storage_capacity", "route_horizon"):
            v = getattr(self, name)
            if v is not None and (not isinstance(v, int) or v < 1):
                bad(f"{name} must be null or an int >= 1")
        for table in ("footprints", "mixer_library"):
            rules = getattr(self, table)
            if not rules or any(a.max_volume >= b.max_volume for a, b in pairwise(rules)):
                bad(f"{table} must be non-empty with increasing max_volume")
        for m in self.mixer_library:
            if m.units < 1:
                bad(f"mixer for volume <= {m.max_volume} needs units >= 1")
            # every droplet a mix of this size can take in or make must fit inside it
            low = max(
                [0, *(x.max_volume for x in self.mixer_library if x.max_volume < m.max_volume)]
            )
            for v in range(low + 1, m.max_volume + 1):
                fh, fw = self.footprint(v)
                if fh > m.shape[0] or fw > m.shape[1]:
                    bad(f"a volume-{v} droplet ({fh}x{fw}) does not fit its mixer {m.shape}")

    # ---- lookups
    def footprint(self, volume: int) -> Shape:
        for rule in self.footprints:
            if volume <= rule.max_volume:
                return rule.shape
        raise ValueError(
            f"grid platform '{self.name}': no footprint for volume {volume} "
            f"(largest {self.footprints[-1].max_volume})"
        )

    def mixer(self, volume: int) -> tuple[Shape, int]:
        """Module shape and mix ticks for a mix of `volume` units."""
        for rule in self.mixer_library:
            if volume <= rule.max_volume:
                return rule.shape, rule.units * self.unit_ticks
        raise ValueError(
            f"grid platform '{self.name}': no mixer for volume {volume} "
            f"(largest {self.mixer_library[-1].max_volume})"
        )

    @property
    def dispense_ticks(self) -> int:
        return self.dispense_units * self.unit_ticks

    @property
    def detect_ticks(self) -> int:
        return self.detect_units * self.unit_ticks

    @property
    def changeover_ticks(self) -> int:
        return self.changeover_units * self.unit_ticks

    @property
    def horizon(self) -> int:
        return self.route_horizon or 4 * (self.grid[0] + self.grid[1])

    def platform(self, mixers: int) -> Platform:
        """The M10 platform with the same durations (ticks), for the priority schedule (14.9).
        `mix_ticks` is the smallest mixer's; the grid graph sets each mix's own duration."""
        return Platform(
            name=self.name,
            tick_ms=1,
            mixers=min(mixers, self.mixers) if self.mixers else mixers,
            ports_per_fluid=self.ports_per_fluid,
            detectors=self.detectors,
            storage_capacity=self.storage_capacity,
            dispense_ticks=self.dispense_ticks,
            mix_ticks=self.mixer_library[0].units * self.unit_ticks,
            detect_ticks=self.detect_ticks,
            changeover_ticks=self.changeover_ticks,
        )

    # ---- config
    @staticmethod
    def from_dict(data: Mapping[str, Any]) -> GridPlatform:
        names = {f.name for f in fields(GridPlatform)}
        if set(data) != names:
            raise ValueError(
                f"grid platform keys mismatch: missing {sorted(names - set(data))}, "
                f"unknown {sorted(set(data) - names)}"
            )

        def rules(rows: Any) -> tuple[SizeRule, ...]:
            return tuple(
                SizeRule(
                    int(x["max_volume"]),
                    (int(x["shape"][0]), int(x["shape"][1])),
                    int(x.get("units", 0)),
                )
                for x in rows
            )

        d = dict(data)
        d["grid"] = (int(d["grid"][0]), int(d["grid"][1]))
        d["detector_shape"] = (int(d["detector_shape"][0]), int(d["detector_shape"][1]))
        d["dead_cells"] = tuple((int(a), int(b)) for a, b in d["dead_cells"])
        d["footprints"] = rules(d["footprints"])
        d["mixer_library"] = rules(d["mixer_library"])
        return GridPlatform(**d)

    @staticmethod
    def load(name: str = "meda_grid", directory: Path = PLATFORM_DIR) -> GridPlatform:
        data = OmegaConf.to_container(OmegaConf.load(directory / f"{name}.yaml"))
        if not isinstance(data, dict):
            raise ValueError(f"grid platform config {name}.yaml is not a mapping")
        return GridPlatform.from_dict({str(k): v for k, v in data.items()})


# ---------------------------------------------------------------- layout


@dataclass(frozen=True, slots=True)
class Chip:
    """One instance's chip: ports per fluid (the cell on the edge where a droplet appears, its
    footprint anchored there), detection sites, waste cells, dead cells and the interior where
    modules may go."""

    instance_id: str
    height: int
    width: int
    ports: Mapping[int, tuple[Cell, ...]]
    detectors: tuple[Rect, ...]
    waste: tuple[Cell, ...]
    dead: frozenset[Cell]
    interior: Rect

    def inside(self, rect: Rect) -> bool:
        return rect.r >= 0 and rect.c >= 0 and rect.r2 < self.height and rect.c2 < self.width

    def port_anchor(self, port: Cell, shape: Shape) -> Cell:
        """Where a dispensed droplet appears: its footprint touches the port cell from inside."""
        r, c = port
        if c == 0:  # west edge
            return (min(r, self.height - shape[0]), 0)
        return (0, min(c, self.width - shape[1]))  # north edge

    def waste_anchor(self, cell: Cell, shape: Shape) -> Cell:
        """Where a droplet must be to leave at waste cell `cell` (south edge)."""
        return (self.height - shape[0], min(cell[1], self.width - shape[1]))


def _spread(n: int, length: int, size: int, gap: int, what: str, instance_id: str) -> list[int]:
    """`n` start offsets evenly along an edge of `length` cells, each block `size` long and at
    least `gap` empty cells apart."""
    if n == 0:
        return []
    pos = [round((i + 1) * length / (n + 1) - size / 2) for i in range(n)]
    pos = [min(max(p, 0), length - size) for p in pos]
    if any(b - a < size + gap for a, b in pairwise(pos)):
        raise ValueError(
            f"{instance_id}: {n} {what} of {size} cells with {gap} between do not fit an edge "
            f"of {length} cells"
        )
    return pos


def layout(instance: Instance, gp: GridPlatform) -> Chip:
    """Ports (half the fluids' ports on the west edge, the rest on the north edge), detection
    sites on the east edge, waste cells on the south edge (SPEC 14.6)."""
    h, w = gp.grid
    iid = instance.instance_id
    big = max(gp.footprints[-1].shape)
    keys = [(f.id, k) for f in instance.fluids for k in range(gp.ports_per_fluid)]
    west_n = (len(keys) + 1) // 2
    west = _spread(west_n, h, big, gp.spacing, "west ports", iid)
    north = _spread(len(keys) - west_n, w, big, gp.spacing, "north ports", iid)
    cells = [(r, 0) for r in west] + [(0, c) for c in north]
    ports: dict[int, list[Cell]] = {}
    for (fid, _), cell in zip(keys, cells, strict=True):
        ports.setdefault(fid, []).append(cell)
    dh, dw = gp.detector_shape
    det_rows = _spread(gp.detectors, h, dh, gp.spacing + big, "detectors", iid)
    detectors = tuple(Rect(r, w - dw, dh, dw) for r in det_rows)
    waste = tuple(
        (h - 1, c) for c in _spread(gp.waste_ports, w, big, gp.spacing, "waste ports", iid)
    )
    m = gp.edge_margin
    interior = Rect(m, m, h - 2 * m, w - 2 * m)
    if interior.h < 1 or interior.w < 1:
        raise ValueError(f"{iid}: edge band {m} leaves no interior on a {h} x {w} grid")
    for cell in gp.dead_cells:
        if not (0 <= cell[0] < h and 0 <= cell[1] < w):
            raise ValueError(f"{iid}: dead cell {cell} outside the {h} x {w} grid")
    return Chip(
        iid,
        h,
        w,
        {k: tuple(v) for k, v in ports.items()},
        detectors,
        waste,
        frozenset(gp.dead_cells),
        interior,
    )


# ---------------------------------------------------------------- masks


def window_any(grid: np.ndarray, shape: Shape, pad: int) -> np.ndarray:
    """`out[r, c]` is True if any cell of `grid` lies in the `shape` rectangle anchored at
    `(r, c)` grown by `pad` on every side (clipped to the grid). Shape of `out`: the anchors
    where `shape` fits, `(H - h + 1, W - w + 1)`."""
    big_h, big_w = grid.shape
    h, w = shape
    s = np.zeros((big_h + 1, big_w + 1), dtype=np.int32)
    s[1:, 1:] = np.cumsum(np.cumsum(grid, axis=0, dtype=np.int32), axis=1)
    r = np.arange(big_h - h + 1)
    c = np.arange(big_w - w + 1)
    r0 = np.clip(r - pad, 0, big_h)[:, None]
    r1 = np.clip(r + h + pad, 0, big_h)[:, None]
    c0 = np.clip(c - pad, 0, big_w)[None, :]
    c1 = np.clip(c + w + pad, 0, big_w)[None, :]
    total = s[r1, c1] - s[r0, c1] - s[r1, c0] + s[r0, c0]
    return np.asarray(total > 0)
