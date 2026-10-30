"""Fair-time test (paper log D-42): is search's gain on full-size stages 3-5 lookahead, or time?

E-36: search with planner finishes is below the best planner on 85 of 99 set C problems, but
it ran up to 900 s against the planner's ~1 s. Here the planner gets the same 900 s: the
sharing pass restarted again and again with random choices, keeping the cheapest plan (a
randomised greedy with restarts). Each restart takes a random target order (a quarter of the
time one of the pass's own orders) and, for every delivery group, one of the `top` cheapest
candidates, the i-th with weight 2^-i. The start is the best planner's cost, so the answer is
never dearer than it. Seeded per problem; every plan replayed and validated.

Reading, fixed in advance (D-42), on all of set C, search = `net` at 256 from E-36's records
(`search=`), against the planner at 900 s; ratio = mean of search cost / planner cost:
- **lookahead is real:** ratio <= 0.95 and the search cheaper on >= 50 of the problems;
- **the gain was time:** ratio >= 0.99;
- otherwise **partly**: report how much of E-36's 16% survives.
The search ran up to 900 s, less where it finished early (C3 median 162 s), so the planner here
gets at least the search's time: the test leans the planner's way.

    uv run python scripts/fair_time.py seconds=900 workers=28 \
        search=outputs/expert/round2/search_c.jsonl out=outputs/expert/round2/fair_time.jsonl
    uv run python scripts/fair_time.py summary=outputs/expert/round2/fair_time.jsonl \
        search=outputs/expert/round2/search_c.jsonl

Arguments are key=value: `seconds` (budget per problem, default 900), `top` (candidates drawn
from, default 4), `limit=N` problems per stage, `workers`, `search` (E-36's records, optional).
"""

from __future__ import annotations

import json
import sys
import time
import zlib
from collections import defaultdict
from multiprocessing import Pool
from pathlib import Path
from typing import Any

import numpy as np

from meda.core.actions import Action
from meda.core.composition import Rational
from meda.core.problem import Instance
from meda.core.rules import IllegalActionError
from meda.core.state import PrepState
from meda.solvers.baselines import share

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import headroom_share as hs  # noqa: E402
import search_check as sc  # noqa: E402

WEIGHTS = hs.WEIGHTS
CHECKPOINTS = (1, 10, 60, 300, 900)
# Fixed in advance (module docstring).
REAL_RATIO, REAL_WINS, TIME_RATIO = 0.95, 50, 0.99


def random_plan(inst: Instance, rng: np.random.Generator, top: int) -> share._Builder | None:
    """One randomised run of the sharing pass, or None if it gets stuck."""
    own = share.orders(inst, WEIGHTS)
    if rng.random() < 0.25:
        order = own[int(rng.integers(len(own)))]
    else:
        order = [int(i) for i in rng.permutation(len(inst.targets))]
    b = share._Builder(inst, PrepState.initial(inst), [], WEIGHTS)
    try:
        for tid in order:
            while b.state.remaining_demand[tid]:
                options = share._options(b, tid)
                if not options:
                    return None
                k = min(top, len(options))
                w = 0.5 ** np.arange(k)
                b = options[int(rng.choice(k, p=w / w.sum()))][2]
    except IllegalActionError:
        return None
    return b if b.state.is_success else None


def run_task(task: dict[str, Any], opts: dict[str, Any]) -> dict[str, Any]:
    inst = Instance.from_json(task["instance"])
    rng = np.random.default_rng([zlib.crc32(inst.instance_id.encode()), 42])
    t0 = time.perf_counter()
    best_cost: Rational | None = None
    best_plan: tuple[Action, ...] | None = None
    found: list[tuple[float, float]] = []  # (seconds, cost) at each improvement on the planner
    tries = stuck = 0
    while time.perf_counter() - t0 < opts["seconds"]:
        b = random_plan(inst, rng, opts["top"])
        tries += 1
        if b is None:
            stuck += 1
            continue
        if best_cost is None or b.cost < best_cost:
            best_cost, best_plan = b.cost, tuple(b.actions)
            found.append((time.perf_counter() - t0, float(b.cost)))
    elapsed = time.perf_counter() - t0
    stats = hs.plan_stats(inst, "share_random", best_plan, elapsed)  # replays and validates
    planner = task["planner"]
    at = {}
    for t in CHECKPOINTS:
        costs = [c for s, c in found if s <= t]
        at[str(t)] = min([planner, *costs])
    rec = {k: v for k, v in task.items() if k != "instance"}
    rec.update(
        tries=tries,
        stuck=stuck,
        random_cost=stats["cost"],
        cost=min(planner, stats["cost"]) if stats["cost"] is not None else planner,
        at=at,
        seconds=round(elapsed, 1),
    )
    return rec


def _star(args: tuple[dict[str, Any], dict[str, Any]]) -> dict[str, Any]:
    return run_task(*args)


def summarise(path: Path, search: Path | None) -> str:
    recs = [json.loads(x) for x in path.read_text().splitlines()]
    lines = [
        "planner = best of the three planners; random = the planner restarted with random "
        "choices for the budget (never dearer than the planner).",
        "",
        "| set | n | random below planner at 1 / 10 / 60 / 300 / 900 s | mean random / planner "
        "at 900 s | tries (median) | stuck (share) |",
        "|---|---|---|---|---|---|",
    ]
    by_set: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for r in recs:
        by_set[r["set"]].append(r)
    for s in sorted(by_set):
        rs = by_set[s]
        below = [sum(r["at"][str(t)] < r["planner"] - 1e-9 for r in rs) for t in CHECKPOINTS]
        ratio = float(np.mean([r["cost"] / r["planner"] for r in rs]))
        tries = float(np.median([r["tries"] for r in rs]))
        stuck = sum(r["stuck"] for r in rs) / max(1, sum(r["tries"] for r in rs))
        lines.append(
            f"| {s} | {len(rs)} | {' / '.join(map(str, below))} | {ratio:.3f} | {tries:.0f} "
            f"| {stuck:.2f} |"
        )
    if search is None:
        return "\n".join(lines)
    planner = {r["id"]: r["cost"] for r in recs}
    runs: dict[tuple[str, int], dict[str, float]] = defaultdict(dict)
    for x in search.read_text().splitlines():
        r = json.loads(x)
        if r["id"] in planner and r["search_cost"] is not None:
            runs[(r["kind"], r["sims"])][r["id"]] = r["search_cost"]
    lines += [
        "",
        "Search (E-36) against the planner at 900 s, all of C:",
        "",
        "| search | sims | n | cheaper | equal | dearer | mean search / planner at 900 s |",
        "|---|---|---|---|---|---|---|",
    ]
    for kind, sims in sorted(runs):
        got = runs[(kind, sims)]
        r = [got[i] / planner[i] for i in got]
        lines.append(
            f"| {kind} | {sims} | {len(r)} | {sum(x < 1 - 1e-9 for x in r)} "
            f"| {sum(abs(x - 1) <= 1e-9 for x in r)} | {sum(x > 1 + 1e-9 for x in r)} "
            f"| {np.mean(r):.3f} |"
        )
    net = runs.get(("net", 256))
    if net:
        lines += ["", f"Reading (fixed in advance, D-42): {verdict(net, planner)}"]
    return "\n".join(lines)


def verdict(search: dict[str, float], planner: dict[str, float]) -> str:
    r = [search[i] / planner[i] for i in search]
    ratio, wins = float(np.mean(r)), sum(x < 1 - 1e-9 for x in r)
    head = f"net 256 / planner at 900 s = {ratio:.3f}, cheaper on {wins} of {len(r)}"
    if ratio <= REAL_RATIO and wins >= REAL_WINS:
        return head + ": lookahead is real, not time"
    if ratio >= TIME_RATIO:
        return head + ": the gain was time, not lookahead"
    return head + ": partly lookahead, partly time"


def main(argv: list[str]) -> None:
    args = dict(a.split("=", 1) for a in argv)
    search = Path(args["search"]) if "search" in args else None
    if "summary" in args:
        print(summarise(Path(args["summary"]), search))
        return
    opts = {"seconds": float(args.get("seconds", 900)), "top": int(args.get("top", 4))}
    limit = int(args["limit"]) if "limit" in args else None
    found = sc.tasks({"C"}, limit)
    out = Path(args.get("out", "outputs/fair_time.jsonl"))
    out.parent.mkdir(parents=True, exist_ok=True)
    workers = int(args.get("workers", 1))
    todo = [(t, opts) for t in found]
    print(f"{len(found)} problems x {opts['seconds']:.0f} s, {workers} workers", flush=True)
    t0, done = time.perf_counter(), 0
    with out.open("w") as fh:
        results = map(_star, todo) if workers == 1 else Pool(workers).imap_unordered(_star, todo)
        for rec in results:
            fh.write(json.dumps(rec) + "\n")
            fh.flush()
            done += 1
            if done % 10 == 0 or done == len(todo):
                print(f"{done}/{len(todo)}  {time.perf_counter() - t0:.0f} s", flush=True)
    print(f"wrote {done} records to {out}\n")
    print(summarise(out, search))


if __name__ == "__main__":
    main(sys.argv[1:])
