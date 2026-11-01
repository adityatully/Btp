"""AlphaZero loop pilot (D-49, after E-44): train the network on the real outcomes of its own
play, with no planner inside the search.

E-44: the network copies what it is shown, also on new problems; what it was shown is the
planner's view (visit counts of planner-judged searches, value targets from planner finishes).
This pilot changes only the training signal, as AlphaZero trains:
- **play** (`part=play`): the no-planner search (`search_check`'s `zero`: the network's prior
  and value head, no planner call; the env's hint columns are unchanged) plays fresh practice
  problems, with Dirichlet noise on the root prior for exploration. Only the positions it
  actually played become targets: the **visit counts** there, and as value the **remaining cost
  of the cheapest whole plan the search itself reached through that position**, or, where it
  reached none, the failure penalty: `failure_penalty_factor` times the cost of the cheapest
  construction of the whole problem (a planner-free bound, as AlphaTensor's), raised to that
  factor times the dearest finish on the same line of play if one is dearer, so giving up is
  never scored cheaper than finishing badly (pilot 1's agent learned to give up, F-log).
  Problems: the E2 grid's small cells and stage 1-3 draws, where the no-planner search
  finishes most (curriculum; stages 4-5 only if this climbs). None is a test problem.
- **train** (`part=train`): `imitation_check.py arm=visits` on the last `buffer` turns' games,
  from the previous turn's network.
- **eval** (`part=eval`): the no-planner search on sets A, B and C3 (66 problems with known
  planner costs), 256 simulations.
`part=loop` runs eval on the starting network (turn 0), then `turns` times play → train → eval,
each in its own process, and appends one line per turn to `<out>/loop.jsonl`.

Reading, fixed in advance (D-49), on the eval set at the last turn against turn 0: **it
climbs** if the search is below the best planner on at least max(5, 20%) more problems and
leaves no more unfinished. Climbs → the targets were what held the network back; extend the
curriculum to stages 4-5 and race it against planner finishes. Unfinished falls but no cost
gain → it learns to finish, not to economise. Neither → own-outcome targets do not lift it at
this scale either; stop and report.

    uv run python scripts/zero_loop.py part=loop turns=8 \
        start=outputs/expert/round3/copy/imitation.pt out=outputs/zero/pilot workers=36

Arguments are key=value: `part`, `start` (turn 0's network), `turns`, `out`, `workers`,
`small` (problems per small cell and turn, default 10), `stages` (stage 1-3 draws per turn,
default 300), `sims` (256), `seconds` (cap per problem, play 120, eval 300), `buffer` (turns of
games trained on, default 3), `epochs` (per turn, default 2), `device` (cuda), `threads` (16),
`alpha`, `eps` (root noise, default 0.3, 0.25); play/train/eval take `turn` and `agent`.
"""

from __future__ import annotations

import json
import math
import subprocess
import sys
import time
import zlib
from multiprocessing import Pool
from pathlib import Path
from typing import Any

import numpy as np

from meda.analysis.evaluate import build_env
from meda.core.problem import Instance
from meda.generators.benchmark_io import instance_hash
from meda.solvers.fallback import cheapest_construction

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import expert_iteration as ei  # noqa: E402
import search_check as sc  # noqa: E402

GROW, GROW_MIN = 0.2, 5  # fixed in advance (D-49)
EVAL_SETS = ("A", "B", "C3")
FIRST_SMALL = 5000  # the small cells' draws at indices FIRST_SMALL + 100 * turn (rounds: 1000-3999)


class ZeroGame(sc.Search):
    """The no-planner search as a self-play game: root noise, and targets only on the
    positions played."""

    def __init__(
        self,
        *args: Any,
        factor: float,
        construction: float,
        alpha: float,
        eps: float,
        **kwargs: Any,
    ) -> None:
        super().__init__(*args, planner_leaf=False, collect=1, **kwargs)
        self.factor, self.construction, self.alpha, self.eps = factor, construction, alpha, eps
        self.penalty = factor * construction  # raised in `samples` if a finish is dearer
        self._noised: sc.Node | None = None

    def simulate(self, root: sc.Node) -> None:
        if root is not self._noised:
            self._noised = root
            if self.eps > 0 and len(root.prior) > 1 and self.rng is not None:
                noise = self.rng.dirichlet(np.full(len(root.prior), self.alpha))
                root.prior = (1 - self.eps) * root.prior + self.eps * noise
        super().simulate(root)

    def _take(self, top: sc.Node) -> None:
        """Off-path positions are not targets (their values depend on how often they were
        tried, E-44)."""

    def _keep(self, node: sc.Node) -> None:
        if node.terminal:
            return
        remaining = node.best - float(node.snap.plan_cost) if math.isfinite(node.best) else None
        self._policy.append((node.actions, [int(v) for v in node.visits], remaining))  # type: ignore[arg-type]

    def samples(self, value_cap: int, rng: np.random.Generator) -> list[sc.Sample]:
        """The played positions, the failure penalty filled in where no plan was reached."""
        got = super().samples(0, rng)
        finished = [r for _, _, r in got if r is not None]
        self.penalty = self.factor * max([self.construction, *finished])
        return [(a, v, self.penalty if r is None else r) for a, v, r in got]


def _cfg() -> dict[str, Any]:
    return {"costs": sc._load("costs/geometric.yaml"), "env": sc._load("env/sample_prep.yaml")}


def problems(turn: int, small: int, stages: int) -> list[Instance]:
    first = FIRST_SMALL + 100 * turn
    if small > 100:
        raise ValueError(f"small={small}: at most 100 per cell and turn (indices overlap)")
    probs = ei.small_problems(small, first) + ei.stage_problems(stages, 1000 + turn, 3, "equal")
    held = ei.search_test_hashes()
    return [i for i in probs if instance_hash(i) not in held]


def play(task: tuple[str, dict[str, Any]]) -> dict[str, Any]:
    inst_json, opts = task
    inst = Instance.from_json(inst_json)
    cfg = _cfg()
    built = cheapest_construction(inst, sc.WEIGHTS)
    if built is None:
        raise RuntimeError(f"{inst.instance_id}: no construction, so no failure penalty")
    rng = np.random.default_rng([opts["turn"], zlib.crc32(inst.instance_id.encode())])
    t0 = time.perf_counter()
    game = ZeroGame(
        build_env(cfg, [inst]),
        inst,
        ei._agent(opts["agent"]),
        t0 + opts["seconds"],
        reward_scale=opts["reward_scale"],
        rng=rng,
        factor=float(cfg["env"]["failure_penalty_factor"]),
        construction=float(built[1]),
        alpha=opts["alpha"],
        eps=opts["eps"],
    )
    found = game.run(opts["sims"])
    got = game.samples(0, rng)
    cost = sc.hs.plan_stats(inst, "zero", found["plan"], 0.0)["cost"] if found["plan"] else None
    return {
        "id": inst.instance_id,
        "cell": inst.metadata.get("cell", ""),
        "instance": inst_json,
        "finished": cost is not None,
        "cost": cost,
        "construction": float(built[1]),
        "capped": found["capped"],
        "seconds": round(time.perf_counter() - t0, 1),
        "samples": ei._pack(tuple(got)),
        "n_value": len(got),
        "n_policy": len(got),
        "penalty": game.penalty,
        "n_failed": sum(1 for _, _, r in got if r == game.penalty),
    }


def eval_stats(recs: list[dict[str, Any]]) -> dict[str, Any]:
    """Per eval set and over all: below the best planner, unfinished, mean cost / planner
    where finished."""
    out: dict[str, Any] = {}
    for name in (*EVAL_SETS, "all"):
        rs = [r for r in recs if name == "all" or r["set"] == name]
        done = [r for r in rs if r["search_cost"] is not None]
        out[name] = {
            "n": len(rs),
            "below": sum(1 for r in done if r["search_cost"] < r["planner"] - 1e-9),
            "unfinished": len(rs) - len(done),
            "ratio": float(np.mean([r["search_cost"] / r["planner"] for r in done]))
            if done
            else None,
        }
    return out


def reading(first: dict[str, Any], last: dict[str, Any]) -> str:
    """D-49's rule on the whole eval set."""
    need = max(GROW_MIN, math.ceil(GROW * first["below"]))
    climbs = last["below"] >= first["below"] + need and last["unfinished"] <= first["unfinished"]
    if climbs:
        return f"climbs (below {first['below']} → {last['below']}, needed +{need})"
    if last["unfinished"] < first["unfinished"]:
        return "learns to finish, not to economise"
    return "does not climb: own-outcome targets do not lift it at this scale"


def _pool(fn: Any, todo: list[Any], workers: int, out: Path, every: int = 20) -> list[Any]:
    out.parent.mkdir(parents=True, exist_ok=True)
    recs, t0 = [], time.perf_counter()
    with out.open("w") as fh, Pool(workers) as pool:
        for rec in pool.imap_unordered(fn, todo):
            fh.write(json.dumps(rec) + "\n")
            fh.flush()
            recs.append(rec)
            if len(recs) % every == 0 or len(recs) == len(todo):
                print(f"{len(recs)}/{len(todo)}  {time.perf_counter() - t0:.0f} s", flush=True)
    return recs


def _eval_one(args: tuple[dict[str, Any], dict[str, Any]]) -> dict[str, Any]:
    task, opts = args
    return sc.run_task(task, "zero", opts["sims"], opts)


def main(argv: list[str]) -> None:
    args = dict(a.split("=", 1) for a in argv)
    part = args.get("part", "loop")
    out = Path(args.get("out", "outputs/zero/pilot"))
    workers = int(args.get("workers", 1))
    sims = int(args.get("sims", 256))
    turn = int(args.get("turn", 0))
    if part == "play":
        opts = {
            "agent": args["agent"],
            "turn": turn,
            "sims": sims,
            "seconds": float(args.get("seconds", 120)),
            "reward_scale": float(args.get("reward_scale", 0.01)),
            "alpha": float(args.get("alpha", 0.3)),
            "eps": float(args.get("eps", 0.25)),
        }
        probs = problems(turn, int(args.get("small", 10)), int(args.get("stages", 300)))
        print(f"play, turn {turn}: {len(probs)} problems, {workers} workers", flush=True)
        recs = _pool(play, [(i.to_json(), opts) for i in probs], workers, out / f"play{turn}.jsonl")
        n = sum(r["n_value"] for r in recs)
        print(
            f"finished {sum(r['finished'] for r in recs)}/{len(recs)}; {n} positions, "
            f"{sum(r['n_failed'] for r in recs)} with the failure penalty"
        )
    elif part == "train":
        buffer = int(args.get("buffer", 3))
        demos = ",".join(
            str(out / f"play{t}.jsonl") for t in range(max(1, turn - buffer + 1), turn + 1)
        )
        cmd = [
            sys.executable,
            str(ROOT / "scripts/imitation_check.py"),
            "stages=3",
            f"demos={demos}",
            "arm=visits",
            "value_coef=0.5",
            f"init={args['agent']}",
            f"epochs={args.get('epochs', 2)}",
            f"eval_every={args.get('epochs', 2)}",
            f"device={args.get('device', 'cuda')}",
            f"threads={args.get('threads', 16)}",
            f"out={out / f'net{turn}'}",
        ]
        subprocess.run(cmd, check=True)
    elif part == "eval":
        opts = {
            "agent": args["agent"],
            "sims": sims,
            "seconds": float(args.get("seconds", 300)),
            "hints": "planner",
            "reward_scale": float(args.get("reward_scale", 0.01)),
        }
        todo = [(t, opts) for t in sc.tasks({"A", "B", "C"}, None) if t["set"] in EVAL_SETS]
        recs = _pool(_eval_one, todo, workers, out / f"eval{turn}.jsonl")
        stats = eval_stats(recs)
        with (out / "loop.jsonl").open("a") as fh:
            fh.write(json.dumps({"turn": turn, "agent": args["agent"], **stats}) + "\n")
        print(json.dumps(stats))
    elif part == "loop":
        if "summary" in args:
            print(summary_loop(out), end="")
            return
        if (out / "loop.jsonl").exists():
            raise SystemExit(f"{out / 'loop.jsonl'} exists: pick a new out= (no overwrite)")
        turns = int(args.get("turns", 8))
        common = [
            a for a in argv if a.split("=", 1)[0] not in ("part", "turn", "agent", "start", "turns")
        ]

        def step(name: str, t: int, agent: str) -> None:
            subprocess.run(
                [sys.executable, __file__, f"part={name}", f"turn={t}", f"agent={agent}", *common],
                check=True,
            )

        agent = args["start"]
        step("eval", 0, agent)
        for t in range(1, turns + 1):
            step("play", t, agent)
            step("train", t, agent)
            agent = str(out / f"net{t}" / "imitation.pt")
            step("eval", t, agent)
        print(summary_loop(out))
    else:
        raise SystemExit(f"part must be play, train, eval or loop, got {part!r}")


def summary_loop(out: Path) -> str:
    rows = [json.loads(x) for x in (out / "loop.jsonl").read_text().splitlines() if x]
    lines = [
        "| turn | "
        + " | ".join(f"{s} below / unfinished / ratio" for s in (*EVAL_SETS, "all"))
        + " |"
    ]
    lines.append("|---" * (len(EVAL_SETS) + 2) + "|")
    for r in rows:
        cells = []
        for s in (*EVAL_SETS, "all"):
            st = r[s]
            ratio = "-" if st["ratio"] is None else f"{st['ratio']:.3f}"
            cells.append(f"{st['below']} / {st['unfinished']} / {ratio}")
        lines.append(f"| {r['turn']} | " + " | ".join(cells) + " |")
    if len(rows) > 1:
        lines.append(
            f"Reading (fixed in advance, D-49): {reading(rows[0]['all'], rows[-1]['all'])}"
        )
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    main(sys.argv[1:])
