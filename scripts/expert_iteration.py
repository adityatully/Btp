"""Expert iteration (plan 4.2): search writes what the network learns from. Round 1 (paper log
D-38): the teacher's plans. Round 2 (D-40): also what the search observed (`samples=1`).

Draws practice problems and plans each two ways, for the two copy arms of
`scripts/imitation_check.py demos=...`:
- **planner:** the sharing pass (`share_plan`), the teacher of every network so far;
- **search:** MCTS with a flat prior and planner finishes (`search_check.Search`, the best
  searcher of E-33), kept only if cheaper than the planner's plan, so never dearer.
Every plan is replayed and validated. Problems (all new; none is a test problem):
- `small` per cell of the E2 grid (`scripts/headroom.py cells("small")`, the cells of the search
  test's set A), drawn from the same seeds' scheme at indices from 1000 (E2 used 0-19);
- `stages` fresh stage 1-2 draws, held out from every validation and held-out set (as the copy
  test draws them), so the network keeps what it knows.
Any draw equal to a search-test problem (sets A and B) is dropped.

Round 2 (D-40) differs in four ways: the search's prior is a network (`agent=`, round 1's
search-taught one), still with planner finishes; ties break at random (seeded per problem);
problems are new again (`first=2000 seed=1`) and the stage draws cover stages 1-5 (`upto=5`);
and with `samples=1` each record also keeps what the search observed (`Search.samples`): visit
counts where a position was visited at least `min_visits` times, and at every kept position
the remaining cost of the cheapest whole plan seen through it. `load_targets(path)` reads them.

    uv run python scripts/expert_iteration.py small=30 stages=400 workers=30 \
        out=outputs/expert/round1/demos.jsonl

Arguments are key=value: `small`, `stages`, `upto` (stages drawn, default 2), `first` (first
E2 index, default 1000), `sims` (default 256), `seconds` (cap per problem, default 300), `seed`,
`agent` (the prior; default a flat one), `samples` (0/1), `min_visits` (default 64),
`value_cap` (default 16), `workers`, `out`. Each output line: the instance (JSON), both plans
(pickled, base64) and their costs, and with `samples=1` the samples (pickled, base64);
`load_demos(path, arm)` reads the plans back.
"""

from __future__ import annotations

import base64
import json
import pickle
import sys
import time
import zlib
from multiprocessing import Pool
from pathlib import Path
from typing import Any

import numpy as np

from meda.analysis.evaluate import build_env
from meda.core.actions import Action
from meda.core.chip import ChipLimits
from meda.core.problem import Instance
from meda.generators.benchmark_io import BenchmarkSet, instance_hash
from meda.generators.sampler import GeneratorConfig, sample_instance
from meda.generators.stages import StageDesign
from meda.solvers.baselines.share import share_plan

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import headroom  # noqa: E402
import search_check as sc  # noqa: E402

WEIGHTS = sc.WEIGHTS
FIRST_INDEX = 1000  # E2 drew indices 0-19 per cell


def small_problems(per_cell: int, first: int = FIRST_INDEX) -> list[Instance]:
    out = []
    for cell, cfg, grad in headroom.cells("small"):
        for index in range(first, first + per_cell):
            rng = np.random.default_rng([zlib.crc32(cell.encode()), index])
            try:
                inst = sample_instance(GeneratorConfig.from_dict(cfg), headroom.CHIP, rng, WEIGHTS)
                if grad:
                    inst = headroom.as_gradient(inst, rng)
            except ValueError as e:  # counted, not hidden
                print(f"skipped {cell}-{index}: {e}"[:160], flush=True)
                continue
            inst = Instance(
                f"ei-{cell}-{index}",
                inst.fluids,
                inst.targets,
                inst.chip,
                inst.step_limit,
                {**inst.metadata, "cell": cell},
            )
            out.append(inst)
    return out


def stage_problems(count: int, seed: int, upto: int = 2) -> list[Instance]:
    load = sc._load
    design = StageDesign.from_dict(load("stages/track2_v1.yaml")).first(upto)
    chip = ChipLimits.from_dict(load(f"chip/{design.chip}.yaml"))
    held = set(BenchmarkSet.read(design.held_out_val).hashes())
    for s in design.stages:
        held |= BenchmarkSet.read(s.val).hashes()
    source = design.source(
        {s.name: 1 / len(design.stages) for s in design.stages}, chip, WEIGHTS, frozenset(held)
    )
    rng = np.random.default_rng([seed, 38])
    out = []
    for i in range(count):
        inst = source(rng)
        out.append(
            Instance(
                f"ei-stage-{i}",
                inst.fluids,
                inst.targets,
                inst.chip,
                inst.step_limit,
                inst.metadata,
            )
        )
    return out


def search_test_hashes() -> set[str]:
    return {instance_hash(Instance.from_json(t["instance"])) for t in sc.tasks({"A", "B"}, None)}


def _pack(obj: tuple[Any, ...]) -> str:
    return base64.b64encode(pickle.dumps(obj)).decode()


def _unpack(text: str) -> tuple[Any, ...]:
    obj = pickle.loads(base64.b64decode(text))  # our own file, written by `run_task`
    assert isinstance(obj, tuple)
    return obj


_AGENT: dict[str, Any] = {}


def _agent(path: str) -> Any:
    if path not in _AGENT:
        import torch

        from meda.agents.ppo.policy import PPOAgent

        torch.set_num_threads(1)
        _AGENT[path] = PPOAgent.load(path)
    return _AGENT[path]


def run_task(task: tuple[Any, ...]) -> dict[str, Any]:
    """task: (instance JSON, sims, seconds[, options]); options as in `main` (round 2)."""
    inst_json, sims, seconds = task[:3]
    opts: dict[str, Any] = task[3] if len(task) > 3 else {}
    inst = Instance.from_json(inst_json)
    t0 = time.perf_counter()
    planner = share_plan(inst, WEIGHTS)
    if planner is None:
        raise RuntimeError(f"{inst.instance_id}: the sharing pass found no plan")
    p_stats = sc.hs.plan_stats(inst, "share", planner, 0.0)
    cfg = {"costs": sc._load("costs/geometric.yaml"), "env": sc._load("env/sample_prep.yaml")}
    agent = _agent(opts["agent"]) if opts.get("agent") else None
    rng = None
    if "seed" in opts:
        rng = np.random.default_rng([opts["seed"], zlib.crc32(inst.instance_id.encode())])
    collect = opts["min_visits"] if opts.get("samples") else None
    search = sc.Search(build_env(cfg, [inst]), inst, agent, t0 + seconds, rng=rng, collect=collect)
    found = search.run(sims)
    s_stats = sc.hs.plan_stats(inst, "search", found["plan"], 0.0) if found["plan"] else None
    improved = s_stats is not None and s_stats["cost"] < p_stats["cost"] - 1e-9
    best = found["plan"] if improved else planner
    extra: dict[str, Any] = {}
    if opts.get("samples"):
        assert rng is not None, "samples need a seed"
        got = search.samples(opts["value_cap"], rng)
        extra = {
            "samples": _pack(tuple(got)),
            "n_policy": sum(v is not None for _, v, _ in got),
            "n_value": len(got),
        }
    return {
        "id": inst.instance_id,
        "cell": inst.metadata.get("cell", inst.metadata.get("stage", "")),
        "instance": inst_json,
        "planner_cost": p_stats["cost"],
        "search_cost": s_stats["cost"] if improved else p_stats["cost"],
        "improved": improved,
        "capped": found["capped"],
        "seconds": round(time.perf_counter() - t0, 1),
        "planner_plan": _pack(planner),
        "search_plan": _pack(best),
        **extra,
    }


def load_demos(path: str | Path, arm: str) -> tuple[list[Instance], dict[str, tuple[Action, ...]]]:
    """The practice problems and, per problem id, the plan of `arm` (search or planner)."""
    if arm not in ("search", "planner"):
        raise ValueError(f"arm must be search or planner, got {arm!r}")
    insts, plans = [], {}
    for line in Path(path).read_text().splitlines():
        rec = json.loads(line)
        inst = Instance.from_json(rec["instance"])
        insts.append(inst)
        plans[inst.instance_id] = _unpack(rec[f"{arm}_plan"])
    return insts, plans


def load_targets(path: str | Path) -> list[tuple[Instance, tuple[sc.Sample, ...]]]:
    """Round 2 (D-40): per practice problem, what the search observed (`Search.samples`)."""
    out = []
    for line in Path(path).read_text().splitlines():
        rec = json.loads(line)
        if "samples" not in rec:
            raise ValueError(f"{path}: record {rec['id']} has no samples (run with samples=1)")
        out.append((Instance.from_json(rec["instance"]), _unpack(rec["samples"])))
    return out


def summarise(path: Path) -> str:
    recs = [json.loads(x) for x in path.read_text().splitlines()]
    ratio = [r["search_cost"] / r["planner_cost"] for r in recs]
    small = [r for r in recs if r["id"].startswith("ei-") and not r["id"].startswith("ei-stage")]
    samples = ""
    if recs and "samples" in recs[0]:
        samples = (
            f"; samples {sum(r['n_value'] for r in recs)} "
            f"(with visit counts {sum(r['n_policy'] for r in recs)})"
        )
    return (
        f"{len(recs)} problems ({len(small)} small, {len(recs) - len(small)} stages): search "
        f"cheaper than the planner on {sum(r['improved'] for r in recs)} "
        f"(small {sum(r['improved'] for r in small)}); mean search/planner cost "
        f"{np.mean(ratio):.3f}; capped {sum(r['capped'] for r in recs)}; median "
        f"{np.median([r['seconds'] for r in recs]):.0f} s{samples}"
    )


def main(argv: list[str]) -> None:
    args = dict(a.split("=", 1) for a in argv)
    if "summary" in args:
        print(summarise(Path(args["summary"])))
        return
    seed = int(args.get("seed", 0))
    probs = small_problems(
        int(args.get("small", 30)), int(args.get("first", FIRST_INDEX))
    ) + stage_problems(int(args.get("stages", 400)), seed, int(args.get("upto", 2)))
    held = search_test_hashes()
    kept = [i for i in probs if instance_hash(i) not in held]
    print(f"{len(probs)} problems drawn, {len(probs) - len(kept)} equal to a test problem dropped")
    sims, seconds = int(args.get("sims", 256)), float(args.get("seconds", 300))
    opts: dict[str, Any] = {}
    if args.get("agent") or args.get("samples", "0") != "0":  # round 2 (D-40)
        opts = {
            "agent": args.get("agent", ""),
            "seed": seed,
            "samples": args.get("samples", "0") != "0",
            "min_visits": int(args.get("min_visits", 64)),
            "value_cap": int(args.get("value_cap", 16)),
        }
    todo = [(i.to_json(), sims, seconds, opts) for i in kept]
    out = Path(args.get("out", "outputs/expert/round1/demos.jsonl"))
    out.parent.mkdir(parents=True, exist_ok=True)
    workers = int(args.get("workers", 1))
    t0, done = time.perf_counter(), 0
    with out.open("w") as fh:
        results = (
            map(run_task, todo) if workers == 1 else Pool(workers).imap_unordered(run_task, todo)
        )
        for rec in results:
            fh.write(json.dumps(rec) + "\n")
            fh.flush()
            done += 1
            if done % 20 == 0 or done == len(todo):
                print(f"{done}/{len(todo)}  {time.perf_counter() - t0:.0f} s", flush=True)
    print(f"wrote {done} records to {out}")
    print(summarise(out))


if __name__ == "__main__":
    main(sys.argv[1:])
