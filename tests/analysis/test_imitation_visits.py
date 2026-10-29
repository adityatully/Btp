"""`scripts/imitation_check.py arm=visits` (D-40): what the search observed replays into
training positions whose visit counts fit the env's candidates, and the loss trains both heads."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch

from meda.agents.ppo.networks import SetEncoderNet
from meda.agents.ppo.policy import batch_obs
from meda.agents.ppo.ppo import PPOConfig
from meda.analysis.evaluate import build_env
from meda.generators.benchmark_io import BENCHMARKS_DIR, BenchmarkSet

ROOT = Path(__file__).resolve().parents[2]
TINY = BenchmarkSet.read("tiny_exact_v1", BENCHMARKS_DIR)


def module(name: str) -> Any:
    sys.path.insert(0, str(ROOT / "scripts"))
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / f"{name}.py")
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


EI = module("expert_iteration")
IC = module("imitation_check")


def test_search_targets_replay_and_loss(tmp_path: Path) -> None:
    torch.manual_seed(0)
    inst = TINY.instances[0]
    opts = {"agent": "", "seed": 0, "samples": True, "min_visits": 4, "value_cap": 8}
    path = tmp_path / "demos.jsonl"
    path.write_text(json.dumps(EI.run_task((inst.to_json(), 8, 30.0, opts))) + "\n")
    [(loaded, samples)] = EI.load_targets(path)
    cfg = {"costs": IC.load("costs/geometric.yaml"), "env": IC.load("env/sample_prep.yaml")}
    store = IC.SearchTargets()
    store.add_problem(build_env(cfg, [loaded]), loaded, samples, 0.01)
    assert len(store) == len(samples)
    assert sorted(store.value) == sorted(-r * 0.01 for _, _, r in samples)
    for p in store.pi:
        assert p is None or abs(p.sum() - 1) < 1e-9

    net = SetEncoderNet(PPOConfig.from_dict(IC.load("agent/ppo.yaml")).net)
    obs, pi, value = store.minibatch(np.arange(len(store)))
    t = batch_obs(obs, torch.device("cpu"))
    f32 = torch.float32
    tensors = (torch.as_tensor(pi, dtype=f32), torch.as_tensor(value, dtype=f32))
    opt = torch.optim.Adam(net.parameters(), lr=1e-3)
    first = None
    for _ in range(30):
        opt.zero_grad()
        ce, mse, _, _ = IC.search_loss(net, t, *tensors)
        assert torch.isfinite(ce) and torch.isfinite(mse)
        (ce + mse).backward()
        opt.step()
        first = first if first is not None else float((ce + mse).detach())
    assert float(ce + mse) < first  # both heads learn from the targets
