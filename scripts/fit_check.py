"""Fit check (Q-111, after E-43): does the network fail on stages 4-5 because it cannot fit them,
or because what it fits does not carry over to new problems?

One trained network, one measure, two kinds of problem: the practice problems it was trained
on, and the held-out ones (the last 5% of the demos, split exactly as `imitation_check.py`
splits them). Per stage (from each problem's cell; the E2 grid's small problems are `small`):
- **agree:** how often the network's top move is the search's most visited move, at positions
  with visit counts (the copy log's `val_agree` is a different measure: agreement with the
  sharing pass on the validation sets);
- **value error:** mean absolute error of the value head in cost units, and relative to the
  mean target;
- **pair order:** at two positions of the same problem the same number of moves in (sibling
  alternatives), with targets more than 1% apart, how often the head says which is cheaper.
  This is what the search needs from a judge.

Reading, fixed in advance, on stages 4 and 5 together. Guide: train agree < 0.6 → it cannot fit
them (the network: Q-111's architecture branch); train >= 0.6 and held-out agree < train - 0.15
→ it fits but does not generalise (data or what it reads); otherwise it fits and generalises.
Judge, the same with pair order (cannot fit < 0.7; does not generalise if held-out < train -
0.1). Held-out pair order < 0.6 means its ranking of alternatives is close to a coin flip.

    uv run python scripts/fit_check.py agent=outputs/expert/round3/copy/imitation.pt \
        demos=outputs/expert/round2/demos.jsonl,outputs/expert/round3/demos.jsonl workers=36

Arguments are key=value: `agent`, `demos` (comma-separated, as the copy step got them),
`per_stage` (training problems per stage, sampled; default 100), `workers` (replay processes),
`threads`, `device`, `reward_scale`, `out` (optional JSON of the table).
"""

from __future__ import annotations

import json
import sys
import time
from collections import defaultdict
from multiprocessing import Pool
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt
import torch

from meda.agents.ppo.policy import PPOAgent, batch_obs, masked_distribution
from meda.agents.ppo.ppo import chunk_by_width
from meda.analysis.evaluate import build_env
from meda.core.problem import Instance
from meda.generators.stages import StageDesign

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import expert_iteration as ei  # noqa: E402
import imitation_check as ic  # noqa: E402
import search_check as sc  # noqa: E402

DEFAULTS: dict[str, Any] = {
    "agent": "",
    "demos": "",
    "per_stage": 100,
    "workers": 8,
    "threads": 8,
    "device": "cpu",
    "reward_scale": 0.01,
    "max_chunk_rows": 32768,
    "out": "",
}
FIT_AGREE, GAP_AGREE, FIT_PAIR, GAP_PAIR, COIN = 0.6, 0.15, 0.7, 0.1, 0.6
PAIR_MARGIN = 0.01


def _cfg() -> dict[str, Any]:
    return {"costs": sc._load("costs/geometric.yaml"), "env": sc._load("env/sample_prep.yaml")}


def stage_names() -> dict[str, str]:
    """Cell name → stage name of the stage design."""
    design = StageDesign.from_dict(sc._load("stages/track2_v1.yaml"))
    return {c.config.name: s.name for s in design.stages for c in s.cells}


def split(lines: list[dict[str, Any]], per_stage: int, seed: int = 0) -> list[tuple[str, int]]:
    """(train or held, index) of the records to replay: the held-out 5% at the end, as the copy
    step holds them out, and up to `per_stage` training problems per stage."""
    cells = stage_names()
    cut = len(lines) - max(1, len(lines) // 20)
    by_stage: dict[str, list[int]] = defaultdict(list)
    for i in range(cut):
        by_stage[cells.get(lines[i]["cell"], "small")].append(i)
    rng = np.random.default_rng(seed)
    out = []
    for idx in by_stage.values():
        take = idx if len(idx) <= per_stage else rng.choice(idx, per_stage, replace=False)
        out += [("train", int(i)) for i in sorted(take)]
    return out + [("held", i) for i in range(cut, len(lines))]


def replay(task: tuple[str, str, float]) -> ic.SearchTargets:
    inst_json, samples, reward_scale = task
    inst = Instance.from_json(inst_json)
    store = ic.SearchTargets()
    store.add_problem(build_env(_cfg(), [inst]), inst, ei._unpack(samples), reward_scale)
    return store


def predict(
    net: Any, store: ic.SearchTargets, device: torch.device, max_rows: int
) -> tuple[npt.NDArray[np.bool_], npt.NDArray[np.float64]]:
    """Per position: whether the top move is the most visited (False where no visit counts)
    and the value head's output."""
    agree = np.zeros(len(store), dtype=bool)
    value = np.zeros(len(store))
    index = np.arange(len(store))
    with torch.no_grad():
        for chunk in chunk_by_width(index, store.candidate_widths(), max_rows):
            obs, pi, _ = store.minibatch(chunk)
            t = batch_obs(obs, device)
            logits, v = net(t)
            top = masked_distribution(logits, t["action_mask"]).logits.argmax(-1).cpu().numpy()
            agree[chunk] = (pi.sum(-1) > 0) & (top == pi.argmax(-1))
            value[chunk] = v.cpu().numpy()
    return agree, value


def pair_order(
    problem: npt.NDArray[np.int64],
    depth: npt.NDArray[np.int64],
    target: npt.NDArray[np.float64],
    pred: npt.NDArray[np.float64],
) -> tuple[int, int]:
    """(right, pairs) over positions of one problem at one depth whose targets differ by more
    than `PAIR_MARGIN` of the larger."""
    buckets: dict[tuple[int, int], list[int]] = defaultdict(list)
    for i in np.flatnonzero(~np.isnan(target)):
        buckets[(int(problem[i]), int(depth[i]))].append(int(i))
    right = pairs = 0
    for idx in buckets.values():
        for a in range(len(idx)):
            for b in range(a + 1, len(idx)):
                i, j = idx[a], idx[b]
                if abs(target[i] - target[j]) <= PAIR_MARGIN * max(abs(target[i]), abs(target[j])):
                    continue
                pairs += 1
                right += int(np.sign(pred[i] - pred[j]) == np.sign(target[i] - target[j]))
    return right, pairs


def metrics(rows: dict[str, npt.NDArray[Any]], reward_scale: float) -> dict[str, Any]:
    has_pi, has_v = rows["has_pi"], ~np.isnan(rows["target"])
    err = np.abs(rows["pred"][has_v] - rows["target"][has_v])
    right, pairs = pair_order(rows["problem"], rows["depth"], rows["target"], rows["pred"])
    return {
        "problems": len(np.unique(rows["problem"])),
        "positions": len(has_pi),
        "with_visits": int(has_pi.sum()),
        "agree": float(rows["agree"][has_pi].mean()) if has_pi.any() else None,
        "value_mae_cost": float(err.mean() / reward_scale) if has_v.any() else None,
        "value_rel": float(err.mean() / np.abs(rows["target"][has_v]).mean())
        if has_v.any()
        else None,
        "pairs": pairs,
        "pair_order": right / pairs if pairs else None,
    }


def reading(train: dict[str, Any], held: dict[str, Any]) -> list[str]:
    """The rule fixed in advance, on stages 4-5 together."""
    out = []
    for what, key, fit, gap in (
        ("guide", "agree", FIT_AGREE, GAP_AGREE),
        ("judge", "pair_order", FIT_PAIR, GAP_PAIR),
    ):
        tr, he = train[key], held[key]
        if tr is None or he is None:
            out.append(f"{what}: no data")
        elif tr < fit:
            out.append(f"{what}: cannot fit stages 4-5 ({key} on training {tr:.2f} < {fit})")
        elif he < tr - gap:
            out.append(f"{what}: fits stages 4-5 but does not generalise ({tr:.2f} → {he:.2f})")
        else:
            out.append(f"{what}: fits and generalises ({tr:.2f} → {he:.2f})")
    if held["pair_order"] is not None and held["pair_order"] < COIN:
        out.append(f"judge on new problems ranks alternatives close to a coin flip (< {COIN})")
    return out


def _fmt(v: Any) -> str:
    return "-" if v is None else f"{v:.3f}" if isinstance(v, float) else str(v)


def main(argv: list[str]) -> None:
    args = dict(DEFAULTS)
    for item in argv:
        key, _, value = item.partition("=")
        if key not in args:
            raise SystemExit(f"unknown argument '{key}'; known: {sorted(args)}")
        args[key] = type(DEFAULTS[key])(value)
    if not args["agent"] or not args["demos"]:
        raise SystemExit("agent= and demos= are required")
    started = time.perf_counter()
    torch.set_num_threads(args["threads"])
    lines = [
        json.loads(x)
        for p in args["demos"].split(",")
        for x in Path(p).read_text().split("\n")
        if x
    ]
    chosen = split(lines, args["per_stage"])
    cells = stage_names()
    tasks = [(lines[i]["instance"], lines[i]["samples"], args["reward_scale"]) for _, i in chosen]
    with Pool(args["workers"]) as pool:
        stores = pool.map(replay, tasks, chunksize=1)
    print(f"replayed {len(stores)} problems in {time.perf_counter() - started:.0f}s", flush=True)

    device = torch.device(args["device"])
    net = PPOAgent.load(args["agent"], device).net
    net.eval()
    groups: dict[tuple[str, str], dict[str, list[Any]]] = defaultdict(lambda: defaultdict(list))
    for (part, i), store in zip(chosen, stores, strict=True):
        if not len(store):
            continue
        agree, pred = predict(net, store, device, args["max_chunk_rows"])
        stage = cells.get(lines[i]["cell"], "small")
        for key in (stage, "s4-s5") if stage in ("s4_targets", "s5_storage") else (stage,):
            g = groups[(part, key)]
            g["has_pi"].append(np.array([p is not None for p in store.pi]))
            g["agree"].append(agree)
            g["pred"].append(pred)
            g["target"].append(np.array(store.value, dtype=np.float64))
            g["depth"].append(np.array(store.depth, dtype=np.int64))
            g["problem"].append(np.full(len(store), i, dtype=np.int64))
    table = {
        f"{part}/{key}": metrics({k: np.concatenate(v) for k, v in g.items()}, args["reward_scale"])
        for (part, key), g in sorted(groups.items(), key=lambda x: (x[0][1], x[0][0]))
    }
    cols = [
        "problems",
        "positions",
        "with_visits",
        "agree",
        "value_mae_cost",
        "value_rel",
        "pairs",
        "pair_order",
    ]
    print("| problems | " + " | ".join(cols) + " |")
    print("|---" * (len(cols) + 1) + "|")
    for name, row in table.items():
        print(f"| {name} | " + " | ".join(_fmt(row[c]) for c in cols) + " |")
    train, held = table.get("train/s4-s5"), table.get("held/s4-s5")
    if train and held:
        for line in reading(train, held):
            print(f"Reading (fixed in advance): {line}")
    print(f"{time.perf_counter() - started:.0f}s")
    if args["out"]:
        Path(args["out"]).write_text(json.dumps(table, indent=1))


if __name__ == "__main__":
    main(sys.argv[1:])
