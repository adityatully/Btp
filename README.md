# meda — RL design automation for MEDA biochips

Research codebase. The current target is the **sample-preparation agent**: given fluids and
target mixtures for many experiments, produce one mixing DAG (with shared intermediates) that
meets every target at low cost.

- `SPEC.md` — the specification and source of truth.
- `CLAUDE.md` — short working rules.
- `PROJECT_CONTEXT.md` — research context and the reasons behind decisions.

## Install

Requires Python ≥ 3.11 and [uv](https://docs.astral.sh/uv/).

```bash
uv sync
```

On Linux (the training workstation) `torch` comes from the PyTorch CUDA 13.0 index (needs NVIDIA driver ≥ 580). On macOS
it is the CPU build (development and tests only).

Optional extras: `uv sync --extra tracking` (wandb), `uv sync --extra sb3` (MaskablePPO cross-check).

Pre-commit hooks (after the git repository exists): `uv run pre-commit install`.

## Test and check

```bash
uv run pytest -q
uv run ruff check
uv run mypy src/meda/core
```

Long acceptance runs are marked `slow` and skipped by default:

```bash
uv run pytest -m slow
```

## Run

```bash
uv run python -m meda.train agent=tabular_q          # trains and reports gaps to the optima
uv run python -m meda.train agent=ppo "train.instances=[tiny-sharing-m2]" env.L_pad=128   # M8 (a)
uv run python -m meda.train agent=ppo train.resume=<run>/checkpoints/last.pt                 # resume
uv run python -m meda.evaluate_cli "eval.producers=[random,naive,exact_search]" eval.benchmark=tiny_exact_v1
uv run python scripts/profile_env.py generator=<name>
```

Configs live in `configs/` (Hydra); override any value on the command line. Run outputs go to
`outputs/` (gitignored).

## Status

Milestones M0–M7 are done: scaffold, exact core, mixing DAG, environment, generators and
benchmarks, solvers, tabular Q-learning, evaluation harness. M8 (PPO with a set encoder) is in
progress: observation and network v2 read any number of fluids, targets and droplets on any chip
(SPEC 1.4), and acceptance (a), (c) and (d) are met; (b) waits for the stage design. See SPEC
section 16 for the milestone order.
