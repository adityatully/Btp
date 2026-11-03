"""e-MRCM's Table III protocol (Liang et al., TCAD 39(10) 2020, Section VI-B; Q-117, L-7).

Per (reactants N, targets T): `sets` random target sets, each target a random positive vector
over N reactants summing to 512 (P = 1/512; the paper does not say how it draws them), 4 units
per target (4X), mixes of at most 8 units, reactant weights geometric from 1 to 64 (rounded:
the paper gives no values for N = 5). "MRCM" = each target's tree alone, "e-MRCM" = the shared
graph (`meda.solvers.baselines.emrcm`); both with the full MRCM search, or the narrower Eq. 11
reading with `narrow=1`. Printed next to the paper's averages (transcribed from Table III).

    uv run python scripts/emrcm_table3.py sets=20 reactants=2,3 narrow=1

Arguments are key=value: `sets` (default 50), `reactants` (default 2,3,4,5), `targets`
(default 2,3,4,5), `narrow` (0/1, default 1), `seed` (default 7).
"""

from __future__ import annotations

import sys
import time

import numpy as np

from meda.core.composition import Composition, Rational
from meda.solvers.baselines.emrcm import emrcm_graph
from meda.solvers.baselines.paper_graph import NotApplicable, PaperGraph

# Table III: (reactant cost, operations, waste) for MRCM, then e-MRCM.
PAPER: dict[tuple[int, int], tuple[float, ...]] = {
    (2, 2): (334.14, 10.78, 6.32, 327.45, 10.52, 5.93),
    (2, 3): (489.60, 16.76, 9.21, 469.04, 15.98, 8.18),
    (2, 4): (660.35, 22.06, 12.90, 608.57, 20.14, 10.26),
    (2, 5): (829.63, 27.28, 16.38, 762.74, 24.80, 12.26),
    (3, 2): (224.44, 9.94, 10.02, 222.69, 9.82, 9.81),
    (3, 3): (343.03, 14.92, 15.10, 335.91, 14.52, 14.49),
    (3, 4): (460.78, 19.88, 19.83, 445.31, 19.26, 18.64),
    (3, 5): (569.62, 24.80, 25.10, 551.95, 23.82, 23.32),
    (4, 2): (256.66, 9.52, 11.88, 253.53, 9.42, 11.78),
    (4, 3): (395.63, 13.96, 17.99, 394.69, 13.90, 17.92),
    (4, 4): (533.39, 18.76, 23.97, 528.69, 18.56, 23.74),
    (4, 5): (650.26, 23.86, 29.34, 643.36, 23.54, 28.95),
    (5, 2): (297.17, 9.16, 13.04, 297.17, 9.16, 13.04),
    (5, 3): (442.40, 13.68, 19.29, 441.63, 13.62, 19.18),
    (5, 4): (597.50, 18.38, 25.95, 596.85, 18.36, 25.93),
    (5, 5): (756.94, 22.84, 32.94, 750.86, 22.60, 32.64),
}
PRECISION, VOLUME, MIXER = 512, 4, 8


def draw(n: int, rng: np.random.Generator) -> Composition:
    cuts = np.sort(rng.choice(np.arange(1, PRECISION), n - 1, replace=False))
    parts = np.diff(np.concatenate([[0], cuts, [PRECISION]]))
    return Composition.from_ints(tuple(int(x) for x in parts), PRECISION)


def stats(graph: PaperGraph, w: tuple[Rational, ...], targets: int) -> tuple[float, int, int]:
    c = graph.counts(VOLUME * targets)
    drawn = c["drawn"]
    assert isinstance(drawn, tuple)
    nm, nw = c["nm"], c["nw"]
    assert isinstance(nm, int) and isinstance(nw, int)
    return float(sum((a * b for a, b in zip(w, drawn, strict=True)), Rational.of(0))), nm, nw


def main(argv: list[str]) -> None:
    args = dict(a.split("=", 1) for a in argv)
    sets, narrow = int(args.get("sets", 50)), args.get("narrow", "1") != "0"
    rng = np.random.default_rng(int(args.get("seed", 7)))
    print(
        "| N | T | sets (failed) | MRCM cost / ops / waste | e-MRCM | e-MRCM vs MRCM "
        "| paper MRCM | paper e-MRCM | paper gain |"
    )
    print("|---|---|---|---|---|---|---|---|---|")
    for n in map(int, args.get("reactants", "2,3,4,5").split(",")):
        w = tuple(Rational.of(round(64 ** (i / (n - 1)))) for i in range(n))
        for t in map(int, args.get("targets", "2,3,4,5").split(",")):
            alone, shared, failed, t0 = [], [], 0, time.perf_counter()
            for _ in range(sets):
                targets = [(draw(n, rng), VOLUME) for _ in range(t)]
                try:
                    graph = emrcm_graph(targets, w, 2, MIXER, n, narrow)
                    trees = [emrcm_graph([x], w, 2, MIXER, n, narrow) for x in targets]
                except NotApplicable:
                    failed += 1
                    continue
                shared.append(stats(graph, w, t))
                alone.append(tuple(map(sum, zip(*(stats(g, w, 1) for g in trees), strict=True))))
            if not shared:
                print(f"| {n} | {t} | 0 ({failed}) | - | - | - | | | |")
                continue
            a, b, p = np.mean(alone, 0), np.mean(shared, 0), PAPER[(n, t)]
            print(
                f"| {n} | {t} | {len(shared)} ({failed}) | {a[0]:.1f} / {a[1]:.2f} / {a[2]:.2f} "
                f"| {b[0]:.1f} / {b[1]:.2f} / {b[2]:.2f} | {b[0] / a[0] - 1:+.1%} "
                f"| {p[0]} / {p[1]} / {p[2]} | {p[3]} / {p[4]} / {p[5]} | {p[3] / p[0] - 1:+.1%} |",
                flush=True,
            )
            print(f"  ({time.perf_counter() - t0:.0f} s)", file=sys.stderr)


if __name__ == "__main__":
    main(sys.argv[1:])
