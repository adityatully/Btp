"""Where the benchmark runs spend their time (offline, from `bench.py` records).

For each records file and each set in it, the CPU time of every part of a record: the exact
search for the answer key, each published method, each of our 4 planner runs, and our method
(the search). Times are what the records hold (CPU seconds, D-78); a part copied from an
earlier run by `reuse=` keeps that run's time, so the totals are what a run from scratch
costs. Parts the records do not time (root bound, plan replay and validation, the novelty
check) are left out.

For our method it also reports how much of its time came after it found its final plan
(the last entry of its trace): budget spent without improving the answer.

    uv run python scripts/time_budget.py outputs/bench/b1_v3_all.jsonl [more.jsonl ...]
"""

from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

HOUR = 3600.0


def parts(rec: dict[str, Any]) -> dict[str, float]:
    """CPU seconds per part of one record."""
    out: dict[str, float] = {}
    exact = rec.get("exact")
    if isinstance(exact, dict) and "seconds" in exact:
        out["answer key: exact search"] = float(exact["seconds"])
    for name, st in (rec.get("published") or {}).items():
        if "seconds" in st:
            out[f"published: {name}"] = float(st["seconds"])
    for name, st in (rec.get("planners") or {}).items():
        out[f"planner: {name}"] = float(st.get("seconds", 0.0))
    ours = rec.get("ours") or {}
    if "seconds" in ours:
        out["ours: search"] = float(ours["seconds"])
    return out


def after_best(rec: dict[str, Any]) -> float | None:
    """Seconds our search ran after its final plan was found."""
    ours = rec.get("ours") or {}
    trace = ours.get("trace")
    if not trace or "seconds" not in ours:
        return None
    return max(0.0, float(ours["seconds"]) - float(trace[-1][0]))


def table(recs: list[dict[str, Any]], title: str) -> list[str]:
    by_part: dict[str, list[float]] = defaultdict(list)
    for r in recs:
        for k, v in parts(r).items():
            by_part[k].append(v)
    total = sum(sum(v) for v in by_part.values())
    lines = [
        f"### {title}: {len(recs)} records, {total / HOUR:.1f} CPU-hours",
        "",
        "| part | CPU-h | share | runs | median s | max s |",
        "|---|---|---|---|---|---|",
    ]
    for k, v in sorted(by_part.items(), key=lambda kv: -sum(kv[1])):
        s = sorted(v)
        lines.append(
            f"| {k} | {sum(v) / HOUR:.1f} | {sum(v) / max(total, 1e-9):.1%} | {len(v)} | "
            f"{s[len(s) // 2]:.1f} | {s[-1]:.0f} |"
        )
    ours = [r for r in recs if "seconds" in (r.get("ours") or {})]
    if ours:
        idle = [x for x in (after_best(r) for r in ours) if x is not None]
        capped = sum(1 for r in ours if r["ours"].get("capped"))
        by60 = sum(
            1 for r in ours if r["ours"].get("trace") and float(r["ours"]["trace"][-1][0]) <= 60.0
        )
        lines += [
            "",
            f"Our search: {capped} of {len(ours)} hit the time cap; final plan found by 60 s "
            f"on {by60}; time after the final plan {sum(idle) / HOUR:.1f} CPU-h "
            f"({sum(idle) / max(total, 1e-9):.1%} of all).",
        ]
    return [*lines, ""]


def summarise(paths: list[Path]) -> str:
    out = ["# Benchmark time budget", ""]
    for p in paths:
        recs = [json.loads(x) for x in p.read_text().splitlines() if x.strip()]
        out += [f"## {p}", ""]
        out += table(recs, "all sets")
        sets = sorted({r.get("set", "?") for r in recs})
        if len(sets) > 1:
            for s in sets:
                out += table([r for r in recs if r.get("set") == s], s)
    return "\n".join(out)


if __name__ == "__main__":
    print(summarise([Path(a) for a in sys.argv[1:]]))
