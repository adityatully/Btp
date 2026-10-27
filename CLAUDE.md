# CLAUDE.md

Research codebase for RL-based design automation of MEDA biochips. Current focus: the
sample-preparation agent. **`SPEC.md` is the source of truth. Read sections 1–5 before any change.**

## How to work in this repo
- Implement milestones in the order of SPEC section 16. Do not start a milestone before the
  previous one's acceptance criteria pass.
- Write or update tests with every change (SPEC section 17). Run `uv run pytest`, `uv run ruff check`,
  and `uv run mypy src/meda/core` before saying a task is done.
- If a change touches a public interface, the DAG format, or a default physical assumption,
  update `SPEC.md` in the same change and say so.
- Ask before adding a dependency not listed in SPEC section 3.
- Do not add features, options or files the current milestone does not need.
- Read `notes/plan.md` at the start of a session: it says where the project stands and which
  step comes next. Update a step's status there the day it changes, with its paper log ID.
- Keep `notes/paper_log.md` current: the same day a decision is made, an experiment or run
  reports numbers, or a bug changes behaviour or results, add a dated entry there with its
  evidence (commit, SPEC version, file, run folder) and the numbers. Read its rules at the top.
- Keep `notes/questions.md` current: add every new question the day it comes up; an open one
  points to the plan step that answers it, a closed one to the experiment or decision. Close a
  question in the same change that adds its paper log entry.

## Non-negotiable rules (SPEC section 2)
1. The environment computes all arithmetic exactly; networks only score candidates.
2. No floats in `meda.core`. Compositions are exact integer numerators over a denominator.
3. No `torch` imports in `meda.core`, `meda.generators`, `meda.solvers`.
4. Illegal actions are masked, never penalised.
5. Core state is immutable and hashable; `rules.apply` is a pure function.
6. Deterministic given the seed; all randomness via a passed-in `numpy.random.Generator`.
7. Every solver and agent outputs a `MixingDAG` that passes `dag.validate`.
8. Physical assumptions live in config (`configs/chip`, `configs/costs`), never as constants.
9. Never truncate or clip silently: raise an error naming the instance and the limit.

## Commands
- Tests: `uv run pytest -q`
- Train: `uv run python -m meda.train agent=ppo generator=tiny_dilution`
- Evaluate: `uv run python -m meda.evaluate_cli eval.producers=[<name>,...] eval.benchmark=<set>`
- Profile env: `uv run python scripts/profile_env.py generator=<name>`
