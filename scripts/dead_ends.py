"""Dead-end check (Q-115, after E-40): does the value head fail on full-size problems because it
never saw a position that cannot be finished?

Two parts, one question each:
- `part=practice`: how much does the practice search (`expert_iteration.py`'s, planner finishes)
  throw away? `Search.samples` keeps a position only if a whole plan was seen through it. Here
  every position the search touched is counted as kept, dropped with no whole plan seen through
  it (a dead end as far as the search knows), or ended (the episode stopped there: a finished
  plan or a failure). Fresh stage draws with the design's final weights (`seed`, default 3; the
  rounds used 0-2).
- `part=zero`: where the no-planner search fails, what did the value head think? Each set C
  problem is searched as in `search_check.py` (`zero`, 256 simulations); the moves it played are
  replayed, and at every position the planner's finish (`finish_from`, within the step limit)
  says whether it can still be finished. Recorded per position: the value head's remaining cost,
  the planner's remaining cost (or none), and how the game ended (dead end, step limit, done).

By construction the value data has no dead end: every dropped position is left out. Reading,
fixed in advance. (2) is the test of Q-115: on the failed games, the value head's remaining
cost at the first unfinishable position is no higher than at the position before it (median
ratio <= 1.1: it does not see the cliff). (1) says where the fix's data comes from: the
practice search drops at least 1% of its positions. (2) and (1) → keep the dropped positions in
the value data, labelled with the worst value. (2) without (1) → too few to keep; take dead ends
from the no-planner search's own failed games instead. Not (2) → the head sees the cliff; the
failures come from somewhere else (step limit, search depth), and Q-115 is not the cause.

    uv run python scripts/dead_ends.py part=practice count=56 seconds=600 workers=28 \
        agent=outputs/expert/round2/copy_v/imitation.pt \
        out=outputs/expert/round3/dead_practice.jsonl
    uv run python scripts/dead_ends.py part=zero workers=28 \
        agent=outputs/expert/round3/copy/imitation.pt out=outputs/expert/round3/dead_zero.jsonl
    uv run python scripts/dead_ends.py summary=outputs/expert/round3/dead_zero.jsonl

Arguments are key=value: `part`, `agent` (practice: the prior, default flat; zero: required),
`count`, `seed`, `seconds` (cap per problem, default 600 practice / 900 zero), `sims` (256),
`limit` (zero: problems per stage), `workers`, `out`, `summary`.
"""

from __future__ import annotations

import json
import math
import sys
import time
from collections import Counter
from multiprocessing import Pool
from pathlib import Path
from typing import Any

import numpy as np

from meda.analysis.evaluate import build_env
from meda.core.problem import Instance
from meda.solvers.baselines.share import finish_from

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import expert_iteration as ei  # noqa: E402
import search_check as sc  # noqa: E402

# Fixed in advance (module docstring).
DEAD_SHARE, CLIFF_RATIO = 0.01, 1.1


class CountingSearch(sc.Search):
    """The practice search, counting what `samples` keeps and what it drops."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.counts: Counter[str] = Counter()

    def _keep(self, node: sc.Node) -> None:
        if node.terminal:
            self.counts["ended"] += 1
        elif not math.isfinite(node.best):
            self.counts["dropped"] += 1
        else:
            self.counts["kept"] += 1
        super()._keep(node)


def _cfg() -> dict[str, Any]:
    return {"costs": sc._load("costs/geometric.yaml"), "env": sc._load("env/sample_prep.yaml")}


def run_practice(task: tuple[str, dict[str, Any]]) -> dict[str, Any]:
    inst_json, opts = task
    inst = Instance.from_json(inst_json)
    agent = ei._agent(opts["agent"]) if opts["agent"] else None
    rng = np.random.default_rng([opts["seed"], 115])
    t0 = time.perf_counter()
    search = CountingSearch(
        build_env(_cfg(), [inst]), inst, agent, t0 + opts["seconds"], rng=rng, collect=64
    )
    found = search.run(opts["sims"])
    return {
        "id": inst.instance_id,
        "cell": inst.metadata.get("cell", ""),
        **search.counts,
        "finished": found["plan"] is not None,
        "seconds": round(time.perf_counter() - t0, 1),
    }


def replay(inst: Instance, path: tuple[Any, ...], agent: Any) -> dict[str, Any]:
    """Per position on `path`: the value head's and the planner's remaining cost."""
    env = build_env(_cfg(), [inst])
    obs, _ = env.reset(options={"instance": inst})
    probe = sc.Search(env, inst, agent, math.inf, planner_leaf=False)
    rows: list[tuple[float, float | None]] = []
    info: dict[str, Any] = {}
    for k in range(len(path) + 1):
        if k == len(path) or info:
            break
        state = env.state
        found = finish_from(state, inst, sc.WEIGHTS)
        fits = found is not None and state.step + len(found[0]) <= inst.step_limit
        snap = env.snapshot()
        head = probe.net(obs, len(env.legal))[1]
        env.restore(snap)
        rows.append((head, float(found[1]) if fits and found is not None else None))
        obs, _, term, trunc, step_info = env.step(env.legal.index(path[k]))
        if term or trunc:
            info = step_info
    end = (
        "done"
        if info.get("success")
        else "dead end"
        if info.get("dead_end")
        else ("step limit" if info else "stopped")
    )
    return {"positions": rows, "end": end, "steps": len(path)}


def run_zero(task: tuple[dict[str, Any], dict[str, Any]]) -> dict[str, Any]:
    t, opts = task
    inst = Instance.from_json(t["instance"])
    agent = ei._agent(opts["agent"])
    env = build_env(_cfg(), [inst])
    search = sc.Search(
        env, inst, agent, time.perf_counter() + opts["seconds"], False, opts["reward_scale"]
    )
    found = search.run(opts["sims"])
    rec = {"set": t["set"], "id": t["id"], "finished": found["plan"] is not None}
    rec.update(replay(inst, search.path, agent))
    return rec


def cliff(positions: list[list[Any]]) -> float | None:
    """Value head's remaining cost at the first unfinishable position / at the one before."""
    for k in range(1, len(positions)):
        if positions[k][1] is None and positions[k - 1][1] is not None:
            before = max(positions[k - 1][0], 1e-9)
            return float(positions[k][0] / before)
    return None


def summarise(path: Path) -> str:
    recs = [json.loads(x) for x in path.read_text().splitlines()]
    if recs and "positions" not in recs[0]:
        kept = sum(r.get("kept", 0) for r in recs)
        dropped = sum(r.get("dropped", 0) for r in recs)
        ended = sum(r.get("ended", 0) for r in recs)
        share = dropped / max(1, kept + dropped)
        lines = [
            f"practice search, {len(recs)} problems: positions kept {kept}, dropped (no whole "
            f"plan seen through them) {dropped}, ended {ended}; dropped share {share:.3%}",
            f"(1) dead ends to add (dropped share >= {DEAD_SHARE:.0%}): {share >= DEAD_SHARE}",
        ]
        return "\n".join(lines)
    failed = [r for r in recs if not r["finished"]]
    ends = Counter(r["end"] for r in failed)
    ratios = [c for r in failed if (c := cliff(r["positions"])) is not None]
    lost_at = []
    for r in failed:
        pos = r["positions"]
        first = next((k for k, p in enumerate(pos) if p[1] is None), None)
        if first is not None:
            lost_at.append(first / max(1, len(pos)))
    head_ok = [p[0] / p[1] for r in recs for p in r["positions"] if p[1]]
    head_lost = [p[0] for r in failed for p in r["positions"] if p[1] is None]
    head_fine = [p[0] for r in recs for p in r["positions"] if p[1] is not None]
    med = float(np.median(ratios)) if ratios else math.nan
    lines = [
        f"zero search on C: {len(recs)} problems, failed {len(failed)}; endings {dict(ends)}",
        f"failed games: first unfinishable position at {np.median(lost_at):.0%} of the game "
        f"(median, {len(lost_at)} with one)"
        if lost_at
        else "failed games: none lost",
        f"value head / planner remaining cost where finishable: median {np.median(head_ok):.2f}"
        if head_ok
        else "no finishable positions",
        f"value head remaining cost: finishable median {np.median(head_fine):.0f}, "
        f"unfinishable median {np.median(head_lost):.0f}"
        if head_lost and head_fine
        else "",
        f"at the cliff (first unfinishable / position before): median {med:.2f} "
        f"over {len(ratios)} games",
        f"(2) the head does not see the cliff (median <= {CLIFF_RATIO}): {med <= CLIFF_RATIO}",
    ]
    return "\n".join(x for x in lines if x)


def main(argv: list[str]) -> None:
    args = dict(a.split("=", 1) for a in argv)
    if "summary" in args:
        print(summarise(Path(args["summary"])))
        return
    part = args.get("part", "practice")
    workers = int(args.get("workers", 1))
    sims = int(args.get("sims", 256))
    if part == "practice":
        opts = {
            "agent": args.get("agent", ""),
            "seed": int(args.get("seed", 3)),
            "seconds": float(args.get("seconds", 600)),
            "sims": sims,
        }
        probs = ei.stage_problems(int(args.get("count", 30)), opts["seed"], 5, "final")
        todo: list[Any] = [(i.to_json(), opts) for i in probs]
        fn: Any = run_practice
    elif part == "zero":
        if "agent" not in args:
            raise SystemExit("part=zero needs agent=<imitation.pt>")
        opts = {
            "agent": args["agent"],
            "seconds": float(args.get("seconds", 900)),
            "sims": sims,
            "reward_scale": float(args.get("reward_scale", 0.01)),
        }
        limit = int(args["limit"]) if "limit" in args else None
        todo = [(t, opts) for t in sc.tasks({"C"}, limit)]
        fn = run_zero
    else:
        raise SystemExit(f"part must be practice or zero, got {part!r}")
    out = Path(args.get("out", f"outputs/dead_ends_{part}.jsonl"))
    out.parent.mkdir(parents=True, exist_ok=True)
    print(f"{part}: {len(todo)} problems, {workers} workers", flush=True)
    t0, done = time.perf_counter(), 0
    with out.open("w") as fh:
        results = map(fn, todo) if workers == 1 else Pool(workers).imap_unordered(fn, todo)
        for rec in results:
            fh.write(json.dumps(rec) + "\n")
            fh.flush()
            done += 1
            if done % 10 == 0 or done == len(todo):
                print(f"{done}/{len(todo)}  {time.perf_counter() - t0:.0f} s", flush=True)
    print(f"wrote {done} records to {out}\n")
    print(summarise(out))


if __name__ == "__main__":
    main(sys.argv[1:])
