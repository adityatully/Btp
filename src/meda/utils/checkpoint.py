"""Save and load PPO training checkpoints (SPEC 13; M8 acceptance (c)).

A checkpoint holds everything the next update depends on: weights, optimiser state, the update
and step counters, the torch RNG (and CUDA's, if present), the trainer's numpy generator, and
one `EnvSnapshot` per vector-env worker, so half-finished episodes continue exactly
(`SamplePrepEnv.snapshot`, SPEC 9.1). `extra` carries the state of whatever the caller built
around the trainer (for example an instance source that keeps state).

Files are written with `torch.save` and contain pickled env snapshots, so they are loaded with
`weights_only=False`: load only checkpoints this code wrote.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
import torch

from meda.agents.ppo.ppo import PPOTrainer

CHECKPOINT_FORMAT = "meda-ppo-checkpoint/1"


def save_checkpoint(
    path: str | Path, trainer: PPOTrainer, extra: Mapping[str, Any] | None = None
) -> Path:
    """Write the trainer's full state to `path` (written to a temp file, then renamed)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "format": CHECKPOINT_FORMAT,
        "config": asdict(trainer.cfg),
        "sizes": asdict(trainer.sizes),
        "model": trainer.net.state_dict(),
        "optimizer": trainer.optimizer.state_dict(),
        "update": trainer.update,
        "global_step": trainer.global_step,
        "seed": trainer.seed,
        "torch_rng": torch.get_rng_state(),
        "cuda_rng": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
        "numpy_rng": trainer.rng.bit_generator.state,
        "env_snapshots": list(trainer.envs.get_attr("env_snapshot")),
        "extra": dict(extra or {}),
    }
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, tmp)
    tmp.replace(path)
    return path


def load_checkpoint(path: str | Path, trainer: PPOTrainer) -> dict[str, Any]:
    """Restore `trainer` (built with the same config and env sizes) and its envs; return
    `extra`. Raises if the checkpoint was made with a different config or observation layout."""
    data = torch.load(Path(path), map_location=trainer.device, weights_only=False)
    if data.get("format") != CHECKPOINT_FORMAT:
        raise ValueError(f"{path}: not a {CHECKPOINT_FORMAT} file")
    for key, current in (("config", asdict(trainer.cfg)), ("sizes", asdict(trainer.sizes))):
        if data[key] != current:
            diff = sorted(k for k in current if data[key].get(k) != current[k])
            raise ValueError(f"{path}: {key} differs from this run in {diff}")
    snapshots = data["env_snapshots"]
    if len(snapshots) != trainer.envs.num_envs:
        raise ValueError(f"{path}: {len(snapshots)} env snapshots for {trainer.envs.num_envs} envs")

    trainer.net.load_state_dict(data["model"])
    trainer.optimizer.load_state_dict(data["optimizer"])
    trainer.update = int(data["update"])
    trainer.global_step = int(data["global_step"])
    trainer.seed = int(data["seed"])
    torch.set_rng_state(data["torch_rng"])
    if data["cuda_rng"] is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(data["cuda_rng"])
    trainer.rng = np.random.default_rng()
    trainer.rng.bit_generator.state = data["numpy_rng"]
    trainer.envs.set_attr("env_snapshot", snapshots)
    observed = trainer.envs.call("observe")
    trainer.next_obs = {k: np.stack([o[k] for o in observed]) for k in observed[0]}
    return dict(data["extra"])
