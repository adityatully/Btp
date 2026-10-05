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
uv run python -m meda.train agent=ppo generator=tiny_dilution
uv run python -m meda.evaluate_cli producer=<name> benchmark=<set>
uv run python scripts/profile_env.py generator=<name>
```

Configs live in `configs/` (Hydra); override any value on the command line. Run outputs go to
`outputs/` (gitignored).

## Status

Milestones M0–M3 (scaffold, exact core, mixing DAG, environment) are done. See SPEC section 16 for the milestone order.
