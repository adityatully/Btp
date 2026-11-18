"""J0 and J9 of the justification sweeps (D-98, `notes/justification_audit.md` Part 2): read the
final benchmark records (`scripts/bench.py`, method-v3; E-91, E-96), no new runs.

**J0, the time budget (S16, S18, F3).** Per set and goal: how many problems have a plan by t,
the mean cost at t over the cost at 900 s (how far from converged the search is at t, 95%
bootstrap CI), the mean cost at t over the best of the 4 planner runs, and the share of final
plans found by t, at t = 1, 10, 30, 60, 120, 300, 600, 900 CPU s. Then the cap's overshoot
(search CPU seconds: median, 95th percentile, max, runs over 990 s) and the finisher switch
(problems that switched, median leaf it switched at).

**J9, the statistics (V1, V2).** For every published method under its own goal at 900 s
(D-69's comparison): the verdict with the loss threshold at 5%, 10% (D-69) and 20%; and how
stable the verdict is at smaller set sizes (share of 1,000 random subsets of 20 and 40 of its
problems that give the full set's verdict), per method and set.

    uv run python scripts/justify_offline.py j0=outputs/bench/b1_v3_all.jsonl
    uv run python scripts/justify_offline.py j9=outputs/bench/b1_v3_all.jsonl
"""

from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import bench  # noqa: E402

TIMES = (1.0, 10.0, 30.0, 60.0, 120.0, 300.0, 600.0, 900.0)
CAP = 900.0
OVER = 1.1 * CAP
SIZES = (20, 40)
LOSS_SHARES = (0.05, 0.10, 0.20)


def load(paths: str) -> list[dict[str, Any]]:
    recs = []
    for p in paths.split(","):
        recs += [json.loads(x) for x in Path(p).read_text().splitlines() if x.strip()]
    v3 = [r for r in recs if r.get("method", {}).get("portfolio") is False]
    if len(v3) < len(recs):
        print(f"(left out {len(recs) - len(v3)} records that are not method-v3)")
    return v3


def final_time(rec: dict[str, Any]) -> float | None:
    """CPU second at which the search found its final (cheapest) plan."""
    trace = rec["ours"].get("trace") or []
    if not trace:
        return None
    best = min(c for _, c in trace)
    return min(s for s, c in trace if c <= best + 1e-9)


def j0(recs: list[dict[str, Any]]) -> str:
    groups: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for r in recs:
        groups[(r["set"], r["goal"])].append(r)
    lines = [
        "## J0: cost against the time budget (method-v3, CPU s)",
        "",
        "Per cell: problems with a plan by t; mean cost(t) / cost(900 s) [95% CI]; mean"
        " cost(t) / best planner run; share of final plans found by t.",
        "",
        "| set | goal | n | " + " | ".join(f"{t:g} s" for t in TIMES) + " |",
        "|---|---|---|" + "---|" * len(TIMES),
    ]
    for key in sorted(groups):
        rs = groups[key]
        cells = []
        for t in TIMES:
            have = [r for r in rs if bench.ours_at(r, t) is not None]
            end = [r for r in have if bench.ours_at(r, CAP)]
            conv = [bench.ours_at(r, t) / bench.ours_at(r, CAP) for r in end]
            vs = [bench.ours_at(r, t) / r["best_planner"] for r in have if r.get("best_planner")]
            done = sum((final_time(r) or 0) <= t for r in rs if final_time(r) is not None)
            if conv:
                lo, hi = bench.boot_ci(conv)
                c = f"{np.mean(conv):.3f} [{lo:.3f}, {hi:.3f}]"
            else:
                c = "-"
            p = f"{np.mean(vs):.3f}" if vs else "-"
            cells.append(f"{len(have)}; {c}; {p}; {done / len(rs):.0%}")
        lines.append(f"| {key[0]} | {key[1]} | {len(rs)} | " + " | ".join(cells) + " |")
    lines += [
        "",
        "## J0: the cap's overshoot and the finisher switch",
        "",
        "| set | n | search CPU s median / p95 / max | over 990 s | switched | switch leaf"
        " (median) |",
        "|---|---|---|---|---|---|",
    ]
    by_set: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for r in recs:
        by_set[r["set"]].append(r)
    for s in sorted(by_set):
        rs = by_set[s]
        secs = np.array([r["ours"].get("seconds") or 0.0 for r in rs])
        sw = [r["ours"]["switched_at"] for r in rs if r["ours"].get("switched_at")]
        leaf = f"{int(np.median(sw))}" if sw else "-"
        lines.append(
            f"| {s} | {len(rs)} | {np.median(secs):.0f} / {np.percentile(secs, 95):.0f} /"
            f" {secs.max():.0f} | {int((secs > OVER).sum())} | {len(sw)} | {leaf} |"
        )
    return "\n".join(lines)


def _verdict(p: dict[str, Any], loss_share: float) -> str:
    if p["ci"][1] < 1 and p["losses"] <= loss_share * p["n"]:
        return "beats"
    if p["ci"][0] > 1:
        return "loses"
    return "level"


def comparisons(recs: list[dict[str, Any]]) -> list[tuple[str, str, list[float], list[float]]]:
    """(method, set, ours, theirs) for every published method under its own goal at 900 s."""
    fluid = {(r["set"], r["id"]): r for r in recs if r["goal"] == "fluid"}
    out = []
    for m in bench.PUBLISHED:
        goal = bench.OWN_GOAL[m]
        by_set: dict[str, tuple[list[float], list[float]]] = defaultdict(lambda: ([], []))
        for r in recs:
            if r["goal"] != goal or (r["set"], r["id"]) not in fluid:
                continue
            theirs = bench._cost(fluid[(r["set"], r["id"])]["published"].get(m, {}), goal)
            mine = bench.ours_at(r, CAP)
            if theirs is None or mine is None:
                continue
            by_set[r["set"]][0].append(mine)
            by_set[r["set"]][1].append(theirs)
        for s, (a, b) in sorted(by_set.items()):
            out.append((m, s, a, b))
    return out


def j9(recs: list[dict[str, Any]], seed: int = 0) -> str:
    rng = np.random.default_rng(seed)
    lines = [
        "## J9: verdicts against the loss threshold, and their stability at smaller n",
        "",
        "| method | set | n | verdict at 5% / 10% (D-69) / 20% losses | same verdict at n = "
        + " / ".join(str(s) for s in SIZES)
        + " |",
        "|---|---|---|---|---|",
    ]
    for m, s, a, b in comparisons(recs):
        if len(a) < 10:
            continue
        p = bench.paired(a, b)
        verdicts = [_verdict(p, ls) for ls in LOSS_SHARES]
        stable = []
        for size in SIZES:
            if size >= len(a):
                stable.append("-")
                continue
            same = 0
            for _ in range(1000):
                idx = rng.choice(len(a), size=size, replace=False)
                sub = bench.paired([a[i] for i in idx], [b[i] for i in idx])
                same += _verdict(sub, 0.10) == verdicts[1]
            stable.append(f"{same / 1000:.0%}")
        flag = "" if len(set(verdicts)) == 1 else " (changes)"
        lines.append(
            f"| {m} | {s} | {len(a)} | {' / '.join(verdicts)}{flag} | {' / '.join(stable)} |"
        )
    return "\n".join(lines)


def main(argv: list[str]) -> None:
    args = dict(a.split("=", 1) for a in argv)
    if "j0" in args:
        print(j0(load(args["j0"])))
    if "j9" in args:
        print(j9(load(args["j9"])))


if __name__ == "__main__":
    main(sys.argv[1:])
