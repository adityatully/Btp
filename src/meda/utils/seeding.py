"""Seed Python, NumPy and Torch; create per-worker generators (SPEC 13).

All randomness in `meda.core`, `meda.generators` and `meda.solvers` flows through a passed-in
`numpy.random.Generator` (rule 6), so this only covers the global libraries that training
touches. Torch is imported lazily: `core`, `generators` and `solvers` must stay usable without
any ML library installed.
"""

from __future__ import annotations

import os
import random

import numpy as np


def seed_everything(seed: int, deterministic_torch: bool = True) -> None:
    """Seed Python, NumPy and (if installed) Torch."""
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    try:
        import torch
    except ImportError:
        return
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if deterministic_torch:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def worker_generators(seed: int, count: int) -> list[np.random.Generator]:
    """`count` independent generators from one seed, for vectorised envs."""
    return [np.random.default_rng(s) for s in np.random.SeedSequence(seed).spawn(count)]
