"""Order check (D-96): does the sharing pass need all of its target orders?

The pass (`share.finish_from`) runs once per distinct target order (given, dearest first,
cheapest first, nearest neighbour; `share.ORDER_RULES`) and keeps the cheapest. The search's
leaf finisher calls it at every leaf, so each order it can drop is time for more leaves.
Only problems with at least 2 distinct orders are checked (s1-s3 have one target).

Per problem: a short flat-prior search (`seconds` CPU, `sims` per move, the plain sharing
pass at its leaves) records the states it finishes from; up to `states` of them (the root
first, the rest evenly spaced in visit order) and `cuts` states along its best plan (the deep
states a 900 s search reaches, a short one not) are finished once per order, each order on
its own, with the plain pass and (with `mrcm=1`) greedy MRCM, whose cache the root's own
finish fills first, as in the search. Each run records its open order, cost and CPU time.

**Part 1, repeats.** `finish_from` filters finished targets out of each order but does not
drop orders that become equal; the share of finish time spent on such repeats is free to
save (same answer at every leaf).
**Part 2, which order wins.** Over the finishes with at least 2 distinct open orders, how
often each rule is cheapest (ties count for all), and for the best 1 and 2 rules the share
of finishes they hold a cheapest plan and their mean excess over the cheapest.
**Reading, fixed in advance (D-96):** fewer orders are worth a search pilot if a set of at
most 2 rules is cheapest-or-tied on >= 95% of these finishes and its mean excess is < 0.5%,
on the root, leaf and cut finishes, with the plain pass (and with MRCM where run);
otherwise every order earns its place. Part 1 is reported as found.

    OMP_NUM_THREADS=1 uv run python scripts/order_check.py workers=4 out=outputs/order.jsonl
    uv run python scripts/order_check.py summary=outputs/order.jsonl

Arguments are key=value: `sets` (default val_s4_v2,val_s5_v2,val_emrcm_v1), `limit` problems
per set, `seconds` (60), `sims` (32), `states` (20), `cuts` (10), `mrcm` (0), `workers`, `out`.
"""

from __future__ import annotations

import itertools
import json
import sys
import traceback
from dataclasses import replace
from multiprocessing import Pool
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import search_check as sc  # noqa: E402
from meda.analysis.evaluate import build_env  # noqa: E402
from meda.core.actions import Action  # noqa: E402
from meda.core.problem import Instance  # noqa: E402
from meda.core.rules import IllegalActionError, apply  # noqa: E402
from meda.core.state import PrepState  # noqa: E402
from meda.generators.benchmark_io import BenchmarkSet  # noqa: E402
from meda.solvers.baselines import share  # noqa: E402
from meda.solvers.fallback import RELAXED_STEP_LIMIT  # noqa: E402

SETS = ("val_s4_v2", "val_s5_v2", "val_emrcm_v1")
CLOCK = sc.CLOCK
WEIGHTS = sc.WEIGHTS
# D-96, fixed in advance
MOST_RULES, HOLD, EXCESS = 2, 0.95, 0.005
TIE = 1e-9


class Recorder:
    """The plain sharing pass at every leaf, keeping each state it finishes from."""

    def __init__(self, inst: Instance) -> None:
        self.inst = inst
        self.states: list[PrepState] = []

    def __call__(self, state: PrepState) -> Any:
        self.states.append(state)
        return share.finish_from(state, self.inst, WEIGHTS)


def leaf_states(
    inst: Instance, seconds: float, sims: int
) -> tuple[list[PrepState], tuple[Action, ...] | None]:
    """The states a short flat-prior search finishes from, the root first, and its best plan."""
    cfg = {"costs": sc._load(sc.COSTS), "env": sc._load("env/sample_prep.yaml")}
    cfg["env"]["T_pad"] = max(cfg["env"]["T_pad"], len(inst.targets))
    env = build_env(cfg, [inst])
    rec = Recorder(inst)
    search = sc.Search(env, inst, None, CLOCK() + seconds, True, 0.01, finish=rec)
    out = search.run(sims)
    return rec.states, out["plan"]


def cut_states(inst: Instance, plan: tuple[Action, ...] | None, n: int) -> list[PrepState]:
    """States along `plan` at n evenly spaced cuts strictly inside it: the deep states a
    900 s search reaches that a short one does not."""
    if not plan or n <= 0:
        return []
    state, out = PrepState.initial(inst), []
    cuts = {round(len(plan) * k / (n + 1)) for k in range(1, n + 1)} - {0, len(plan)}
    for i, action in enumerate(plan, start=1):
        state, _, _ = apply(state, action, inst)
        if i in cuts:
            out.append(state)
    return out


def pick(states: list[PrepState], n: int) -> list[PrepState]:
    """The root and up to n - 1 more, evenly spaced in visit order."""
    if len(states) <= n:
        return states
    rest = states[1:]
    step = len(rest) / (n - 1)
    return [states[0]] + [rest[int(i * step)] for i in range(n - 1)]


def per_order(
    state: PrepState, inst: Instance, cache: dict[Any, list[Any]] | None
) -> list[dict[str, Any]]:
    """`finish_from` split into its orders: one run per order, as it runs them."""
    relaxed = replace(inst, step_limit=RELAXED_STEP_LIMIT)
    pool = {d.uid for d in state.droplets}
    out = []
    for rules, order in share.named_orders(inst, WEIGHTS):
        open_order = [t for t in order if state.remaining_demand[t]]
        t0 = CLOCK()
        try:
            start = share._Builder(relaxed, state, [], WEIGHTS, set(pool), mrcm=cache)
            cost: float | None = float(share._run(start, open_order, 0).cost)
        except IllegalActionError:
            cost = None
        out.append({"rules": list(rules), "open": open_order, "cost": cost, "sec": CLOCK() - t0})
    return out


def run_task(task: dict[str, Any], opts: dict[str, Any]) -> dict[str, Any]:
    inst = Instance.from_json(task["instance"])
    t0 = CLOCK()
    states, plan = leaf_states(inst, opts["seconds"], opts["sims"])
    search_sec = CLOCK() - t0
    kept = pick(states, opts["states"])
    cuts = cut_states(inst, plan, opts["cuts"])
    kinds = ["root"] + ["leaf"] * (len(kept) - 1) + ["cut"] * len(cuts)
    kept = kept + cuts
    finishes: dict[str, list[list[dict[str, Any]]]] = {
        "share": [per_order(s, inst, None) for s in kept]
    }
    if opts["mrcm"]:
        cache: dict[Any, list[Any]] = {}
        share.finish_from(PrepState.initial(inst), inst, WEIGHTS, mrcm=cache)  # fills it
        finishes["mrcm"] = [per_order(s, inst, cache) for s in kept]
    return {
        "set": task["set"],
        "id": inst.instance_id,
        "targets": len(inst.targets),
        "orders": len(share.named_orders(inst, WEIGHTS)),
        "leaves": len(states),
        "search_sec": search_sec,
        "kinds": kinds,
        "finishes": finishes,
    }


def _star(args: tuple[dict[str, Any], dict[str, Any]]) -> dict[str, Any]:
    """`run_task`, with a failure recorded as an `error` record instead of ending the run."""
    task, _ = args
    try:
        return run_task(*args)
    except Exception:
        name = Instance.from_json(task["instance"]).instance_id
        return {"set": task["set"], "id": name, "error": traceback.format_exc()}


def tasks(sets: tuple[str, ...], limit: int | None) -> list[dict[str, Any]]:
    out = []
    for name in sets:
        insts = [i for i in BenchmarkSet.read(name).instances if len(share.orders(i, WEIGHTS)) > 1]
        for inst in insts[:limit]:
            out.append({"set": name, "instance": inst.to_json()})
    return out


def repeats(runs: list[dict[str, Any]]) -> tuple[float, float, int, int]:
    """(time of all runs, time of runs repeating an earlier open order, runs, distinct)."""
    seen: set[tuple[int, ...]] = set()
    total = wasted = 0.0
    for r in runs:
        key = tuple(r["open"])
        total += r["sec"]
        if key in seen:
            wasted += r["sec"]
        seen.add(key)
    return total, wasted, len(runs), len(seen)


def rule_costs(runs: list[dict[str, Any]]) -> dict[str, float]:
    """Each rule's cost on this finish (inf where its order fails)."""
    out = {}
    for r in runs:
        for rule in r["rules"]:
            out[rule] = r["cost"] if r["cost"] is not None else float("inf")
    return out


def choosy(runs: list[dict[str, Any]]) -> bool:
    """A finish where the order can matter: at least 2 distinct open orders, one finishes."""
    return len({tuple(r["open"]) for r in runs}) >= 2 and any(r["cost"] is not None for r in runs)


def subset_stats(finishes: list[dict[str, float]], rules: tuple[str, ...]) -> tuple[float, float]:
    """(share of finishes where `rules` hold a cheapest plan, mean excess over the cheapest)."""
    hold, excess = 0, 0.0
    for fc in finishes:
        best = min(fc.values())
        mine = min(fc.get(r, float("inf")) for r in rules)
        if mine <= best + TIE:
            hold += 1
        excess += (mine / best - 1) if mine < float("inf") else 1.0
    n = max(len(finishes), 1)
    return hold / n, excess / n


def reading(finishes: list[dict[str, float]]) -> dict[str, Any]:
    """Win shares per rule and the best sets of 1 and 2 rules (by hold, then excess)."""
    n = len(finishes)
    wins = {
        rule: sum(1 for fc in finishes if fc.get(rule, float("inf")) <= min(fc.values()) + TIE)
        for rule in share.ORDER_RULES
    }
    best = {}
    for size in (1, MOST_RULES):
        scored = [
            (*subset_stats(finishes, combo), combo)
            for combo in itertools.combinations(share.ORDER_RULES, size)
        ]
        hold, exc, combo = max(scored, key=lambda s: (s[0], -s[1]))
        best[size] = {"rules": list(combo), "hold": hold, "excess": exc}
    return {"n": n, "wins": wins, "best": best}


def summarise(path: Path) -> str:
    recs = [json.loads(x) for x in path.read_text().splitlines() if x.strip()]
    errors = [r for r in recs if "error" in r]
    recs = [r for r in recs if "error" not in r]
    lines = [f"# Order check (D-96): {len(recs)} problems, {path}", ""]
    if errors:
        lines += [f"**{len(errors)} problems failed:** {', '.join(r['id'] for r in errors)}", ""]
    finishers = sorted({f for r in recs for f in r["finishes"]})
    verdict_ok = True
    for fin in finishers:
        lines.append(f"## Finisher `{fin}`")
        for scope in [*sorted({r["set"] for r in recs}), "all"]:
            rs = [r for r in recs if scope in ("all", r["set"]) and fin in r["finishes"]]
            total = wasted = 0.0
            runs = distinct = 0
            for r in rs:
                for f in r["finishes"][fin]:
                    t, w, n, d = repeats(f)
                    total, wasted, runs, distinct = total + t, wasted + w, runs + n, distinct + d
            lines.append(
                f"- **{scope}** ({len(rs)} problems): part 1, repeats {runs - distinct} of "
                f"{runs} runs, {wasted:.1f} of {total:.1f} s ({wasted / max(total, 1e-9):.1%})"
            )
            for where in ("root", "leaf", "cut"):
                fcs = [
                    rule_costs(f)
                    for r in rs
                    for kind, f in zip(r["kinds"], r["finishes"][fin], strict=True)
                    if kind == where and choosy(f)
                ]
                if not fcs:
                    continue
                rd = reading(fcs)
                b1, b2 = rd["best"][1], rd["best"][MOST_RULES]
                ok = b2["hold"] >= HOLD and b2["excess"] < EXCESS
                if scope == "all":
                    verdict_ok = verdict_ok and ok
                wins = ", ".join(f"{k} {v}" for k, v in rd["wins"].items())
                lines.append(
                    f"  - {where}, {rd['n']} finishes with a choice: cheapest-or-tied {wins}; "
                    f"best 1 {'+'.join(b1['rules'])} {b1['hold']:.1%} (+{b1['excess']:.2%}); "
                    f"best 2 {'+'.join(b2['rules'])} {b2['hold']:.1%} (+{b2['excess']:.2%})"
                    f"{' [meets bar]' if ok else ''}"
                )
        lines.append("")
    lines.append(
        "**Reading (D-96):** "
        + (
            "at most 2 orders suffice: a search pilot with fewer orders is worth running."
            if verdict_ok
            else "every order earns its place on at least one scope."
        )
    )
    return "\n".join(lines)


def main(argv: list[str]) -> None:
    args = dict(a.split("=", 1) for a in argv)
    if "summary" in args:
        print(summarise(Path(args["summary"])))
        return
    sets = tuple(args["sets"].split(",")) if "sets" in args else SETS
    limit = int(args["limit"]) if "limit" in args else None
    opts = {
        "seconds": float(args.get("seconds", 60)),
        "sims": int(args.get("sims", 32)),
        "states": int(args.get("states", 20)),
        "cuts": int(args.get("cuts", 10)),
        "mrcm": bool(int(args.get("mrcm", 0))),
    }
    out = Path(args.get("out", "outputs/order_check.jsonl"))
    out.parent.mkdir(parents=True, exist_ok=True)
    done = set()
    if out.exists():  # resume: skip problems already recorded without an error
        for line in out.read_text().splitlines():
            if line.strip():
                r = json.loads(line)
                if "error" not in r:
                    done.add((r["set"], r["id"]))
    todo = [
        (t, opts)
        for t in tasks(sets, limit)
        if (t["set"], Instance.from_json(t["instance"]).instance_id) not in done
    ]
    print(f"{len(todo)} problems to run, {len(done)} already done", flush=True)
    workers = int(args.get("workers", 1))
    results = map(_star, todo) if workers == 1 else Pool(workers).imap_unordered(_star, todo)
    with out.open("a") as fh:
        for rec in results:
            fh.write(json.dumps(rec) + "\n")
            fh.flush()
            if "error" in rec:
                print(rec["set"], rec["id"], "FAILED\n" + rec["error"], flush=True)
            else:
                print(rec["set"], rec["id"], rec["leaves"], "leaves", flush=True)
    print(summarise(out))


if __name__ == "__main__":
    main(sys.argv[1:])
