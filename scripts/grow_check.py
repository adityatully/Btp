"""D-87: does leftover time help? (pilot of `bench.py grow=`, read from its records).

Method-v3 gives the search 256 simulations per move and stops when the plan is played to its
end; on small problems that takes seconds of a 900 s budget (60 s and 900 s read the same:
E-64, E-65, E-67). The pilot spends the leftover time: when a pass ends early, search again
from the root with 4 times the simulations, up to 4,096 (`bench.py grow=4 max_sims=4096`).
The first pass is method-v3 itself, so one run gives both (`ours.rounds`).

Problems: val s1-s3 (`val_s1_v2`, `val_s2_v2`, `val_s3_v2`), fluid goal, all published
methods (NFSP is the one we only tie). Never the test sets (they are the final runs).

Reading (D-87, fixed before the run), pooled over the three sets: leftover time **joins the
method** if the cheapest plan after all passes is cheaper than the first pass on at least 10%
of the problems, or closes at least a third of the mean gap to the proved optima; otherwise it
is dropped and method-v3 stays as frozen (D-83).

    OMP_NUM_THREADS=1 uv run python scripts/bench.py sets=val_s1_v2,val_s2_v2,val_s3_v2 \\
        goals=fluid grow=4 max_sims=4096 workers=4 out=outputs/bench/grow.jsonl
    uv run python scripts/grow_check.py summary=outputs/bench/grow.jsonl
"""

from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np


def first_and_last(rec: dict[str, Any]) -> tuple[float | None, float | None]:
    """The cheapest cost after the first pass (method-v3) and after the last."""
    rounds = rec["ours"].get("rounds") or []
    if not rounds:
        return None, None
    return rounds[0][1], rounds[-1][1]


def _nfsp(rec: dict[str, Any]) -> float | None:
    st = (rec.get("published") or {}).get("nfsp") or {}
    return (st.get("by_goal") or {}).get("fluid")


def _wtl(ours: list[float], other: list[float]) -> str:
    w = sum(a < b - 1e-9 for a, b in zip(ours, other, strict=True))
    lo = sum(a > b + 1e-9 for a, b in zip(ours, other, strict=True))
    return f"{w} / {len(ours) - w - lo} / {lo}"


def summarise(recs: list[dict[str, Any]]) -> list[str]:
    lines = [
        "| set | n | more passes | cheaper than pass 1 | last / pass 1 (mean) | "
        "gap to optimum: pass 1 -> last (n) | vs NFSP w/t/l: pass 1 -> last | "
        "CPU s (median): pass 1 -> last |",
        "|---|---|---|---|---|---|---|---|",
    ]
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for r in recs:
        if r["goal"] == "fluid":
            groups[r["set"]].append(r)
            groups["all"].append(r)
    verdict = ""
    for name in [*sorted(g for g in groups if g != "all"), "all"]:
        rs = [r for r in groups[name] if None not in first_and_last(r)]
        pairs = [first_and_last(r) for r in rs]
        more = sum(len(r["ours"]["rounds"]) > 1 for r in rs)
        cheaper = sum(b < a - 1e-9 for a, b in pairs)
        ratio = np.mean([b / a for a, b in pairs]) if pairs else float("nan")
        proved = [(p, r["optimum"]) for p, r in zip(pairs, rs, strict=True) if r["optimum"]]
        gap1 = np.mean([a / o - 1 for (a, _), o in proved]) if proved else float("nan")
        gap2 = np.mean([b / o - 1 for (_, b), o in proved]) if proved else float("nan")
        nf = [(p, _nfsp(r)) for p, r in zip(pairs, rs, strict=True) if _nfsp(r) is not None]
        before = _wtl([a for (a, _), _ in nf], [c for _, c in nf])
        after = _wtl([b for (_, b), _ in nf], [c for _, c in nf])
        t1 = np.median([r["ours"]["rounds"][0][2] for r in rs]) if rs else float("nan")
        t2 = np.median([r["ours"]["rounds"][-1][2] for r in rs]) if rs else float("nan")
        lines.append(
            f"| {name} | {len(rs)} | {more} | {cheaper} | {ratio:.4f} | "
            f"{gap1:.2%} -> {gap2:.2%} ({len(proved)}) | {before} -> {after} | "
            f"{t1:.0f} -> {t2:.0f} |"
        )
        if name == "all":
            closes = bool(proved) and gap1 > 0 and (gap1 - gap2) >= gap1 / 3
            joins = (bool(rs) and cheaper >= 0.1 * len(rs)) or closes
            verdict = (
                f"Reading (D-87): cheaper than pass 1 on {cheaper} of {len(rs)} (bar 10%), "
                f"gap to proved optima {gap1:.2%} -> {gap2:.2%} (bar: a third closed): "
                + ("leftover time joins the method" if joins else "dropped, method-v3 stays")
            )
    return [*lines, "", verdict]


def main(argv: list[str]) -> None:
    args = dict(a.split("=", 1) for a in argv)
    recs = [json.loads(x) for x in Path(args["summary"]).read_text().splitlines()]
    print("\n".join(summarise(recs)))


if __name__ == "__main__":
    main(sys.argv[1:])
