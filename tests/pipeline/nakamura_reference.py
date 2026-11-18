"""An independent reference for routing one size-2 droplet: the weighted graph of Nakamura,
Yamashita, Tomiyama and Gupta (VLSID 2026), searched with Dijkstra (plan 8.6 step 3, E-99).

A state is the pair of cells of the droplet's two halves: side by side (1 x 2), one above the
other (2 x 1), or two 1 x 1 pieces apart (at least 2 apart in a row or column; diagonal
neighbours are not a valid state). Edges and their times (the paper's Fig. 2 and conditions):
- a joined droplet moves to one of 8 neighbours: 1 x 2 one unit up or down, two sideways or
  diagonally; 2 x 1 the mirror;
- it changes shape in two units: one half stays, the other goes to the cell above or below
  (1 x 2) or beside (2 x 1) it;
- it splits in one unit into two pieces apart, each moving at most one cell (if `split`);
- pieces move together in one unit, each by at most one cell, and merge when side by side;
- no diagonal move for a half with a dead cell next to it in a cardinal direction.
The goal is the target pair. Written from the paper, not from our router.
"""

from __future__ import annotations

import heapq
from itertools import product

Cell = tuple[int, int]
D8 = [(dr, dc) for dr in (-1, 0, 1) for dc in (-1, 0, 1) if (dr, dc) != (0, 0)]

# The paper's five test chips (Figs. 8-12): X dead, S start (1 x 2), T target (1 x 2).
PAPER_GRIDS = {
    "exp1": ["...TT", ".X..X", ".....", "..X..", "SS..."],
    "exp2": ["X..TT", ".X...", ".X.X.", "...X.", "SS..."],
    "exp3": ["XX.TT", "...XX", ".....", ".X..X", "SS..X"],
    "exp4": ["....XTT", ".X.....", "....X..", "...X..X", ".......", "X...X..", "SS....."],
    "exp5": [
        "X..X...XTT",
        ".X..X..X..",
        "X.........",
        ".X.X......",
        "X...X..X.X",
        "X....X...X",
        "....X...X.",
        "..........",
        "..X..X....",
        "SS........",
    ],
}


def parse(grid: list[str]) -> tuple[int, int, frozenset[Cell], Cell, Cell]:
    """Height, width, dead cells, and the left cells of the start and the target."""
    h, w = len(grid), len(grid[0])
    cells = [(r, c) for r in range(h) for c in range(w)]
    dead = frozenset(x for x in cells if grid[x[0]][x[1]] == "X")
    start = min(x for x in cells if grid[x[0]][x[1]] == "S")
    target = min(x for x in cells if grid[x[0]][x[1]] == "T")
    return h, w, dead, start, target


def _kind(p: Cell, q: Cell) -> str | None:
    dr, dc = abs(p[0] - q[0]), abs(p[1] - q[1])
    if (dr, dc) == (0, 1):
        return "h"
    if (dr, dc) == (1, 0):
        return "v"
    if dr >= 2 or dc >= 2:
        return "s"
    return None


def route_time(grid: list[str], split: bool = False) -> int | None:
    """The fewest time units from the start to the target (both 1 x 2), or None."""
    h, w, dead, s, t = parse(grid)

    def free(p: Cell) -> bool:
        return 0 <= p[0] < h and 0 <= p[1] < w and p not in dead

    def diag_ok(p: Cell) -> bool:
        return not any((p[0] + a, p[1] + b) in dead for a, b in ((0, 1), (1, 0), (0, -1), (-1, 0)))

    def step_ok(p: Cell, d: tuple[int, int]) -> bool:
        return not (d[0] and d[1]) or diag_ok(p)

    def nexts(p: Cell, q: Cell) -> list[tuple[tuple[Cell, Cell], int]]:
        k = _kind(p, q)
        out: list[tuple[tuple[Cell, Cell], int]] = []
        if k in ("h", "v"):
            for d in D8:
                a, b = (p[0] + d[0], p[1] + d[1]), (q[0] + d[0], q[1] + d[1])
                if free(a) and free(b) and step_ok(p, d) and step_ok(q, d):
                    fast = (d[0] and not d[1]) if k == "h" else (d[1] and not d[0])
                    out.append(((a, b), 1 if fast else 2))
            for keep in (p, q):
                for e in (-1, 1):
                    c = (keep[0] + e, keep[1]) if k == "h" else (keep[0], keep[1] + e)
                    if free(c):
                        out.append(((keep, c), 2))
            moves = [(d1, d2) for d1, d2 in product([(0, 0), *D8], repeat=2)] if split else []
            for d1, d2 in moves:
                a, b = (p[0] + d1[0], p[1] + d1[1]), (q[0] + d2[0], q[1] + d2[1])
                if free(a) and free(b) and _kind(a, b) == "s" and step_ok(p, d1) and step_ok(q, d2):
                    out.append(((a, b), 1))
        else:
            for d1, d2 in product([(0, 0), *D8], repeat=2):
                if d1 == d2 == (0, 0):
                    continue
                a, b = (p[0] + d1[0], p[1] + d1[1]), (q[0] + d2[0], q[1] + d2[1])
                if free(a) and free(b) and _kind(a, b) and step_ok(p, d1) and step_ok(q, d2):
                    out.append(((a, b), 1))
        return out

    def key(a: Cell, b: Cell) -> tuple[Cell, Cell]:
        return (a, b) if a <= b else (b, a)

    start, goal = key(s, (s[0], s[1] + 1)), key(t, (t[0], t[1] + 1))
    dist = {start: 0}
    heap = [(0, start)]
    while heap:
        d, st = heapq.heappop(heap)
        if st == goal:
            return d
        if d > dist[st]:
            continue
        for (a, b), wgt in nexts(*st):
            n = key(a, b)
            if d + wgt < dist.get(n, 1 << 30):
                dist[n] = d + wgt
                heapq.heappush(heap, (d + wgt, n))
    return None
