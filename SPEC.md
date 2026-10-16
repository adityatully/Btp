# MEDA Design Automation — Codebase Specification

Version 0.9 (draft) · 2026-10-16

Changes in 0.9 (M8, in progress): env `snapshot` / `restore` / `observe` for exact resume, and
the `SAME_STEP` autoreset requirement on vector envs (9.1).

Changes in 0.8 (M7): the evaluation harness is built. `dag_metrics` with exact definitions and
the list-schedule config (12.1); `Producer`, `evaluate`, the gap columns and where each is
filled (12.2); `InstanceReference.lower_bound` now exists, written to JSON only when set (8.4);
`evaluate_cli` takes `eval.producers` (12.2, 13); `meda.train` grades tabular Q through
`evaluate` and writes `results.parquet` + `summary.md` instead of `results.json` (13); M7
acceptance as met (16). Then: exact search returns a proved lower bound and its best plan on
timeout (10.2); `val_stage1_v2`, the same 8 instances with 3 proved optima and 5 intervals
(8.4); M8 acceptance (b) measured on it as a gap to the best known cost (16).

Changes in 0.7 (planning before M7; no code yet): benchmark tiers 0–6 and their claim rules
(8.4); an optional `lower_bound` on `InstanceReference` and `SolveResult` (8.4, 10.1); exact
search returns a proved lower bound on timeout, and the branch-and-bound pruning is now in the
code (10.2); `Curriculum(..., exclude=)` and `problem_hash` close the train/benchmark leak (8.3, 8.4); interval gaps `gap_to_lower_bound`,
`gap_to_best_known` and `bound_ratio` (12.2); tier labels and wall time in reports (12.3).
`val_stage1_v1` timeout findings recorded (8.4).

Changes in 0.6 (M5, M6): branch-and-bound exact search and the `include_step` key (10.2); the
cost-to-finish fallback and its scope (10.3); bitscan delegating to the naive construction and
the facda stub (10.4); `tiny_exact_v1` regenerated at a matched accuracy level (8.4); tabular Q
step size, prior and `bind` (11.3); the `meda.train` entry point (13); corrected expectations
for hand-checked cases 2 and 3, which minimise mixes rather than cost (17.3); M5 and M6
acceptance as actually met, including the M = 8 instance where the table breaks (16).

Changes in 0.5 (M4): `GeneratorConfig` fields and the feasibility witness every instance carries
(8.1); recipes (8.2); `Curriculum` and the fifth stage config `dilution_single` (8.3); the
benchmark file format, canonical `instance_hash` and `exclude` holdouts (8.4); `naive.py` and
`PlanBuilder` moved to M4, which needs them (10.1, 10.4, 16); `configs/chip/meda_m4.yaml`;
`env.T_max` raised to 12 to cover `mixture_multi`.

Changes in 0.4 (M3): env constructor, episode-end semantics and info keys (9.1); observation
dtypes and row layout (9.2); exact feature definitions (9.3); Mix sources listed in canonical
order (5.6); `SolveResult` fields and the `failed` status (10.1); `dilution_grid` (8.1); the
`slow` test marker (17).

Changes in 0.3 (M2): DAG builder, validator modes and check names, JSON `format` key (7.1–7.4).

Changes in 0.2 (M1): exact rational cost weights (6.3); `Output` may take from a reservoir and
uses `Source` (5.5); partial-take rule applies to outputs and equivalent actions are listed once
(5.6); shared id counter for droplet uids and DAG node ids (5.4, 7.1); `check_legal`,
`terminal_status`, `replay` (5.7, 5.8); `unit_volume` is informational (5.2).

This document specifies the research codebase for RL-based design automation of MEDA
(Micro-Electrode-Dot-Array) digital microfluidic biochips. The first and current target is
the **sample-preparation agent**: given fluids and target mixtures for many experiments,
produce one mixing DAG (with shared intermediates) that meets every target at low cost.
Later stages (scheduling, routing, washing, and a full-chip cost oracle) get their own
packages; their place is reserved here but they are not built yet.

Read sections 1–5 before writing any code. Implement in the milestone order of section 16.

---

## 1. Goals and non-goals

### Goals
1. An **exact rules engine** for MEDA sample preparation: integer volumes, exact rational
   compositions, k:l mixing with k + l ≤ M, storage limits. No floating point in the rules.
2. A **Gymnasium-compatible environment** on top of it, with action masking, for training
   RL agents over a distribution of problems.
3. A **problem generator** (the start-state distribution p(s₀)) and **versioned benchmark sets**.
4. **Reference solvers**: an exact search for small instances (the "answer key") and classical
   baselines behind one interface.
5. **Agents**: random, tabular Q-learning (tiny instances), PPO with a permutation-invariant
   set encoder and masked candidate scoring. Room for MCTS / AlphaZero-style search later.
6. An **evaluation harness** that compares any agent or baseline on any benchmark set with the
   same metrics.
7. A **shared mixing-DAG format** that every solver and agent outputs, and that later stages
   (scheduler, router) will consume.

### Non-goals (for now)
- No grid, droplet positions, routing, electrode degradation, contamination or runtime faults
  inside the sample-prep environment. Those belong to later packages.
- No hardware interface.
- No hyperparameter-tuning framework beyond config overrides.

---

## 2. Hard design rules (apply everywhere)

1. **The environment does all arithmetic. The network only scores.** Compositions, volumes,
   feasibility, costs and resulting states are computed exactly in `meda.core`. Networks never
   compute a ratio.
2. **No floats in `meda.core`.** Compositions are exact rationals (integer numerators over a
   common integer denominator). Floats appear only when building observations for networks.
3. **No `torch` (or any ML library) import in `meda.core`, `meda.generators`, `meda.solvers`.**
   These must be usable by classical code and tests without a GPU.
4. **Hard constraints are enforced by masking, never by reward penalties.** An illegal action
   must be impossible to select.
5. **Core state is immutable and hashable; transitions are pure functions.**
   `apply(state, action) -> (next_state, step_cost)` with no hidden side effects. This is
   required for exact search, tabular methods, MCTS and caching.
6. **Determinism.** Same instance + same action sequence → same states and costs, bit for bit.
   All randomness goes through a passed-in `numpy.random.Generator`.
7. **Every solver and agent outputs the same `MixingDAG` format** (section 7), validated by the
   same validator.
8. **Every open physical assumption is a config value**, not a constant in code (section 15).
9. **Permutation invariance.** Droplet order and target order carry no meaning. Canonical
   ordering is used only for hashing and for stable candidate indexing.

---

## 3. Technology stack

Python 3.11 or newer. Use the latest stable releases at setup time; pin them in the lock file.

| Purpose | Library | Notes |
| --- | --- | --- |
| Environment manager | `uv` | `pyproject.toml` + `uv.lock`. Plain pip also works |
| Arrays | `numpy` | Observations, vectorised features |
| Exact arithmetic | Python `int` / `math.gcd` | Arbitrary precision; no `fractions.Fraction` in hot paths (slow) |
| Env API | `gymnasium` (1.x) | `reset`, `step`, `Dict` observation, `Discrete` action |
| Deep learning | `torch` | CUDA build matching the workstation driver (RTX A5000) |
| Config | `hydra-core`, `omegaconf` | Composable YAML configs, command-line overrides, multirun sweeps |
| Logging | `tensorboard` | Default |
| Experiment tracking | `wandb` | Optional, off by default |
| Results | `pandas`, `pyarrow` | Results tables as Parquet |
| Plots | `matplotlib` | |
| DAG utilities | `networkx` | Validation, topological order, critical path. Not used in the env hot path |
| Console | `rich`, `tqdm` | |
| Tests | `pytest`, `hypothesis` | Property-based tests for conservation and exactness |
| Lint / format | `ruff` | |
| Types | `mypy` (or `pyright`) | Strict on `meda.core` |
| Hooks | `pre-commit` | ruff + mypy + pytest -q on core |
| Optional baseline trainer | `sb3-contrib` (`MaskablePPO`) | Works because the action is `Discrete` + mask (section 9). The main PPO is in-house |
| Optional speed-up (later) | `numba` | Only after profiling shows a need |

Why an in-house PPO (CleanRL style) and not only Stable-Baselines3: the policy must score a
variable list of candidate actions using pointer-style gathers from a set encoder. That is
simpler to write and debug in a short, single-purpose implementation. `MaskablePPO` is kept as
an optional cross-check.

---

## 4. Repository layout

```
meda-da/
├── CLAUDE.md                  # rules for Claude Code (short; points here)
├── SPEC.md                    # this document
├── README.md                  # how to install, run, test
├── pyproject.toml
├── uv.lock
├── .pre-commit-config.yaml
├── .gitignore                 # outputs/, wandb/, .venv/, __pycache__/, *.ckpt
├── configs/
│   ├── config.yaml            # Hydra root: defaults list
│   ├── chip/                  # physical limits (section 5.2)
│   │   ├── meda_m8.yaml
│   │   └── dmfb_m2.yaml       # 1:1 mixing only, for comparison with DMFB literature
│   ├── costs/
│   │   └── default.yaml       # cost weights (section 6.5)
│   ├── generator/             # p(s0) definitions (section 8)
│   │   ├── tiny_dilution.yaml
│   │   ├── dilution_multi.yaml
│   │   ├── mixture_single.yaml
│   │   └── mixture_multi.yaml
│   ├── env/
│   │   └── sample_prep.yaml   # padding sizes, step limit, observation options
│   ├── agent/
│   │   ├── random.yaml
│   │   ├── tabular_q.yaml
│   │   └── ppo.yaml
│   ├── train/
│   │   └── default.yaml
│   └── eval/
│       └── default.yaml
├── benchmarks/                # versioned instance sets (section 8.4), JSON
│   └── README.md
├── src/meda/
│   ├── __init__.py
│   ├── core/                  # exact rules; no ML imports
│   │   ├── composition.py     # exact rational composition vectors
│   │   ├── fluids.py          # Fluid, FluidKind
│   │   ├── chip.py            # ChipLimits
│   │   ├── problem.py         # Target, Experiment, Instance
│   │   ├── state.py           # Droplet, PrepState (immutable, hashable)
│   │   ├── actions.py         # Mix, Output, Discard (+ canonical keys)
│   │   ├── rules.py           # legal_actions(), apply(), is_terminal()
│   │   ├── costs.py           # CostWeights, StepCost, cost accounting
│   │   ├── dag.py             # MixingDAG, builder, validator, JSON I/O
│   │   └── noise.py           # NoiseModel interface (phase 2); NoNoise default
│   ├── envs/
│   │   ├── sample_prep_env.py # Gymnasium env
│   │   ├── encoding.py        # state -> padded numpy observation; candidate features
│   │   └── wrappers.py        # episode stats, curriculum hook, preference weights
│   ├── generators/
│   │   ├── sampler.py         # random instances from a generator config
│   │   ├── synthetic.py       # forward-built instances with a known feasible plan
│   │   ├── curriculum.py      # staged difficulty schedule
│   │   └── benchmark_io.py    # read/write versioned benchmark files
│   ├── solvers/
│   │   ├── base.py            # Solver protocol
│   │   ├── exact_search.py    # uniform-cost / A* over canonical states
│   │   ├── fallback.py        # cheap cost-to-finish estimate (truncation penalty)
│   │   └── baselines/
│   │       ├── naive.py       # each target built independently
│   │       ├── bitscan.py     # 1:1 binary dilution (DMFB classic)
│   │       └── facda.py       # factorisation-based MEDA dilution (from the paper)
│   ├── agents/
│   │   ├── base.py            # Agent protocol
│   │   ├── random_agent.py
│   │   ├── tabular_q.py
│   │   ├── ppo/
│   │   │   ├── networks.py    # set encoder, candidate scorer, value head
│   │   │   ├── policy.py      # masked categorical over candidates
│   │   │   ├── rollout.py     # buffer, GAE
│   │   │   └── ppo.py         # update loop
│   │   └── mcts/              # reserved; AlphaZero-style search (milestone 9)
│   ├── analysis/
│   │   ├── dag_metrics.py     # proxy and structural metrics of a DAG (section 12)
│   │   ├── evaluate.py        # run solver/agent on a benchmark -> results table
│   │   ├── compare.py         # tables: gaps to optimum and to baselines
│   │   └── plots.py
│   ├── pipeline/              # reserved for later stages; interfaces only (section 14)
│   │   └── interfaces.py
│   ├── utils/
│   │   ├── seeding.py
│   │   ├── logging.py
│   │   └── checkpoint.py
│   ├── train.py               # Hydra entry point for training
│   └── evaluate_cli.py        # Hydra entry point for evaluation
├── scripts/
│   ├── profile_env.py         # env steps per second
│   └── make_benchmark.py      # build and freeze a benchmark set
├── tests/
│   ├── core/
│   ├── envs/
│   ├── generators/
│   ├── solvers/
│   ├── agents/
│   └── analysis/
├── notebooks/                 # exploration only; never imported by src/
└── outputs/                   # Hydra run directories (gitignored)
```

---

## 5. Domain model (`meda.core`)

### 5.1 Fluids and compositions

**Fluid** (`fluids.py`), frozen dataclass:
- `id: int` — index 0..F−1 within an instance.
- `name: str`
- `kind: FluidKind` — `SAMPLE`, `REAGENT`, `BUFFER`.

**Composition** (`composition.py`): an exact point on the probability simplex over the F fluids
of an instance.
- Stored as `num: tuple[int, ...]` (length F, all ≥ 0) and `den: int` (> 0), with
  `sum(num) == den` and `gcd(*num, den) == 1` (canonical, reduced form).
- Constructors: `pure(fluid_id, F)`, `from_ints(num, den)` (reduces), `from_ratio(parts)`
  (e.g. `(2, 1, 1)` → 2/4, 1/4, 1/4).
- `mix(c1, k, c2, l) -> Composition`: exact `(k·c1 + l·c2) / (k + l)`, reduced.
- `linf_distance(c1, c2) -> Rational`: max over fluids of |c1_f − c2_f|, returned as an exact
  `Rational`. Never a float.
- `Rational` (same module): reduced fraction with positive denominator; `+ − ×`, comparisons,
  `Rational.parse` for config values (int, `"a/b"`, decimal string, or float read through its
  shortest decimal repr, so `0.1` → 1/10). `float(r)` exists only for observations and rewards.
- `to_float_array() -> np.ndarray`: only for observations.
- Hashable and totally ordered (for canonical sorting).

### 5.2 Chip limits

**ChipLimits** (`chip.py`), frozen, loaded from `configs/chip/*.yaml`:

| Field | Meaning | Default (meda_m8) |
| --- | --- | --- |
| `max_mix_volume` (M) | Largest volume one mix may produce: k + l ≤ M | 8 |
| `min_mix_volume` | Smallest mix volume | 2 |
| `unit_volume` | Volume unit (all volumes are integers in this unit). Informational: the core only counts units | 1 |
| `min_droplet_volume` | Smallest droplet allowed after any take or split | 1 |
| `allow_partial_take` | A mix may take k < v units from a droplet of volume v (the rest stays as a droplet) | true |
| `storage_capacity` | Max number of non-reservoir droplets on chip after any action | 10 |
| `max_denominator` | Mixes whose result denominator exceeds this are illegal (sets the accuracy level) | 256 |
| `allow_same_composition_merge` | Allow merging two droplets of equal composition (volume consolidation) | false |

`dmfb_m2.yaml` sets `max_mix_volume: 2`, `min_mix_volume: 2` (1:1 only) for comparison with
DMFB methods.

### 5.3 Problem instance

**Target** (`problem.py`), frozen:
- `id: int`
- `experiment_id: int`
- `composition: Composition` — the desired ratio vector.
- `tolerance: Rational` — satisfied if `linf_distance(droplet, target) ≤ tolerance`.
- `droplet_volume: int` — volume of each delivered droplet, in units.
- `demand: int` — number of droplets required.

**Instance**, frozen:
- `instance_id: str`
- `fluids: tuple[Fluid, ...]`
- `targets: tuple[Target, ...]`
- `chip: ChipLimits`
- `step_limit: int`
- `metadata: dict` — generator name, seed, generator version, optional known upper bound.
- JSON round-trip (`to_json`, `from_json`); compositions stored as `{"num": [...], "den": d}`.

### 5.4 State

**Droplet**, frozen:
- `uid: int` — unique within an episode, assigned in creation order (stable ids for the DAG).
- `composition: Composition`
- `volume: int`
- `dag_node: int` — the DAG node that produced it.

**Reservoirs** are not droplets in the state. Each fluid f is an implicit source with unlimited
volume and composition `pure(f)`. Actions refer to a reservoir by `Source.reservoir(f)` and to a
droplet by `Source.droplet(uid)`. Reservoirs never count toward storage.

**PrepState**, frozen and hashable:
- `droplets: tuple[Droplet, ...]` — kept sorted canonically (by composition, then volume,
  then uid).
- `remaining_demand: tuple[int, ...]` — per target.
- `step: int`
- `next_uid: int`
- **Ids:** droplet uids and DAG node ids share one counter, `next_uid`. Reservoir f is DAG node
  f, so `next_uid` starts at F. Every new MIX, OUTPUT or WASTE node takes the next value; a
  droplet made by a Mix has `uid == dag_node`. A droplet reduced by a partial take keeps its uid
  and node.
- `canonical_key() -> tuple` — multiset of `(composition, volume)` (uids dropped) +
  `remaining_demand`. Two states with the same key are the same decision situation. Used by
  exact search, tabular Q and caches. `step` is excluded unless the step limit is binding;
  provide `canonical_key(include_step: bool)`.

### 5.5 Actions

Three action types (`actions.py`), frozen dataclasses:

1. **Mix(a: Source, k: int, b: Source, l: int)** — take k units from source a and l units from
   source b, merge and mix. Creates one new droplet of volume k + l and composition
   `mix(c_a, k, c_b, l)`. A taken droplet's remaining volume (v − k) stays on chip as the same
   droplet with reduced volume; if it becomes 0, the droplet is removed.
2. **Output(src: Source, t: target_id)** — deliver `droplet_volume(t)` units of `src` to target
   t. `src` is a droplet, or a reservoir when the target is that pure fluid within tolerance.
   Decrements `remaining_demand[t]` by 1. Remainder stays on chip (or is removed if 0).
3. **Discard(d: uid)** — send the whole droplet d to waste.

Each action has a `canonical_key()` that does not use droplet uids (it uses the referenced
droplets' `(composition, volume)` plus k, l, target). Tabular methods key on this, never on a
list index.

Not included in v1 (add only by changing this spec): explicit `Split`, explicit `Dispense`
(dispensing is implicit in taking from a reservoir), `Store` (storing is implicit: any droplet
on chip is stored).

### 5.6 Legality (`rules.legal_actions(state, instance) -> tuple[Action, ...]`)

A **Mix(a, k, b, l)** is legal iff all hold:
- a ≠ b (as sources); a reservoir may not be mixed with the same reservoir.
- k ≥ 1, l ≥ 1, `min_mix_volume ≤ k + l ≤ max_mix_volume`.
- For a droplet source with volume v: k ≤ v, and if k < v then `allow_partial_take` is true and
  v − k ≥ `min_droplet_volume`.
- The result composition's denominator ≤ `max_denominator`.
- Storage after the action ≤ `storage_capacity` (count: existing droplets − droplets fully
  consumed + 1).
- If `chip.allow_same_composition_merge` is false (default), the result must differ in
  composition from both sources (no pure merges of equal compositions). Set it to true if plans
  need to combine two small droplets of the same composition into a larger one.
- Symmetry: Mix(a, k, b, l) and Mix(b, l, a, k) are the same action. Enumerate only the
  canonical one: sources in canonical key order, where droplet keys `("D", composition,
  volume)` sort before reservoir keys `("R", fluid_id)`; for two identical droplets, k ≤ l.

An **Output(src, t)** is legal iff `remaining_demand[t] > 0`, droplet volume ≥ `droplet_volume(t)`,
and `linf_distance(c_src, c_t) ≤ tolerance(t)`. Taking fewer units than the droplet holds follows
the same partial-take rule as Mix (`allow_partial_take`, remainder ≥ `min_droplet_volume`).
Output directly from a reservoir is allowed if the target is a pure fluid within tolerance.

A **Discard(d)** is legal for any droplet on chip.

No action is legal once all demand is met or `step == step_limit`.

**Equivalent actions are listed once.** Two droplets with the same `(composition, volume)` make
actions with the same canonical key; only the one using the smaller uid is enumerated (`apply`
still accepts the other, with the same canonical result).

**Order of the returned tuple is canonical and deterministic** (Output actions first, then Mix,
then Discard; each group sorted by canonical key). The environment's candidate index i refers to
position i.

Legality is defined once, by predicates shared by `legal_actions` and `check_legal(state,
action, instance)`, which raises `IllegalActionError` naming the instance, the step and the
violated rule.

### 5.7 Transition (`rules.apply(state, action, instance) -> tuple[PrepState, StepCost, DagDelta]`)

- Pure function. Raises `IllegalActionError` if the action is not legal (assert in debug, always
  in tests).
- Returns the next state, the cost breakdown of this step (section 6), and the DAG nodes/edges
  this action adds (section 7).
- `step` increases by 1 for every action.
- On success, every droplet left on chip goes to WASTE in the same step; the returned state has
  no droplets.
- `replay(instance, actions) -> (final_state, total_cost, deltas)` applies a plan from the
  initial state (tests, solvers).
- `DagDelta` holds the MIX/OUTPUT/WASTE nodes and edges the action adds. Edges from reservoir f
  use node id f; RESERVOIR nodes themselves are added by the DAG builder (M2).

### 5.8 Termination

- **Success:** all `remaining_demand` are 0. Droplets still on chip become waste (charged in the
  final step's cost).
- **Truncation:** `step == step_limit` without success. The final cost adds a truncation penalty
  = `fallback.cost_to_finish(state, instance)` (section 10.3), so an unfinished plan is never
  cheaper than a finished one.
- **Dead end:** no legal action and not success. Treat as truncation (same penalty). Track the
  rate in evaluation; a high rate means masking or generation needs work.
- `rules.terminal_status(state, instance) -> TerminalStatus | None` returns `SUCCESS`,
  `TRUNCATED` or `DEAD_END` (checked in that order); `is_terminal` is `terminal_status(...) is
  not None`.

### 5.9 Noise hook (phase 2; implement interface only)

`NoiseModel.apply(rng, mix_result: Droplet, a, k, b, l) -> Droplet`. `NoNoise` returns the input
unchanged and is the default. A later `SplitErrorNoise` will perturb taken volumes. When any
noise model other than `NoNoise` is active, the env must report `info["stochastic"] = True` and
exact search is disabled.

---

## 6. Costs and reward

### 6.1 Cost components per step (`StepCost`, all exact integers or rationals)
- `reagent_by_fluid: tuple[int, ...]` — units taken from each reservoir in this step.
- `mix_ops: int` — 1 for a Mix, else 0.
- `waste_volume: int` — units sent to waste in this step (Discard, final leftovers).
- `outputs: int` — droplets delivered.

### 6.2 Episode totals
Sum of step costs, plus derived totals: total reagent, sample volume (fluids of kind SAMPLE),
total waste, total mixes.

### 6.3 Scalar cost
```
cost = Σ_f w_fluid[kind(f)] · reagent_by_fluid[f]  +  w_mix · mix_ops  +  w_waste · waste_volume
```
Weights from `configs/costs/default.yaml`:
```yaml
w_fluid: {SAMPLE: 10.0, REAGENT: 2.0, BUFFER: 0.5}
w_mix: 1.0
w_waste: 1.0
```
These defaults are placeholders: the sample is the scarcest fluid. Make every weight overridable.

Weights are **exact rationals** (`CostWeights`, parsed with `Rational.parse`; YAML may use ints,
decimals such as `0.5`, or strings such as `"1/3"`). The scalar cost is therefore an exact
`Rational`, and exact search compares plan costs without rounding. Negative weights are rejected
(search needs nonnegative edge costs).

### 6.4 Reward
`reward_t = −float(cost_t)` (the float is made from the exact rational cost, in the env only). γ = 1 (episodes always end).
The return is minus the total plan cost, so maximising return minimises cost.

### 6.5 Preference-conditioned option (multi-objective, later)
If `env.preference_conditioning: true`, each episode samples a weight vector w from a
configured distribution, the reward uses w, and w is appended to the observation globals.
Default false.

### 6.6 No shaping in v1
No distance-to-target shaping (it can punish building shared intermediates). If shaping is
added later, it must be potential-based: `F = Φ(s′) − Φ(s)` with Φ defined in one place.

---

## 7. Mixing DAG format (`core/dag.py`)

The single output format for every agent and solver, and the input format for later stages.

### 7.1 Nodes
| Field | Type | Meaning |
| --- | --- | --- |
| `id` | int | Unique in the DAG. Reservoir f has id f; other ids come from the state's `next_uid` counter (5.4) |
| `type` | enum | `RESERVOIR`, `MIX`, `OUTPUT`, `WASTE` |
| `fluid_id` | int or null | For `RESERVOIR` |
| `target_id` | int or null | For `OUTPUT` |
| `composition` | `{num, den}` | Composition of the fluid at this node |
| `volume` | int | Volume drawn in total (RESERVOIR), produced (MIX), delivered (OUTPUT) or wasted (WASTE) |
| `step` | int | Action index that created the node (for ordering); first use for a RESERVOIR |

One `RESERVOIR` node per fluid that is used. One `MIX` node per Mix action. One `OUTPUT` node per
Output action. One `WASTE` node per Discard and per leftover droplet at the end.

`MixingDAG.from_deltas(instance, deltas, totals, producer)` joins the `DagDelta`s of an episode
(in order) and adds the RESERVOIR nodes. `totals` is the producer's reported `StepCost` sum.

### 7.2 Edges
`{src, dst, volume}` — `volume` units flow from node src to node dst.

### 7.3 Validator (`validate(dag, instance, complete=True) -> ValidationReport`)
The report lists `DagError(check, message)` items; `report.ok`, `report.failed_checks`, and
`report.raise_if_invalid()` (raises `DagValidationError` naming the instance). With
`complete=False` (a truncated or dead-end episode) demand may be unmet and MIX droplets may still
be on chip; every other check is unchanged.

`STRUCTURE` is checked first (unique ids, edges between existing nodes, no edge into a reservoir
or out of an OUTPUT/WASTE, OUTPUT/WASTE have exactly one parent, `fluid_id`/`target_id` set
exactly for RESERVOIR/OUTPUT, ids below F only for reservoirs). If it fails, nothing else runs.
Then, each a separate `DagCheck`:
1. `ACYCLIC` (if it fails, nothing else runs).
2. `VOLUME_CONSERVATION`: for every MIX node, inflow = its volume, and its volume = total
   outflow; OUTPUT/WASTE inflow = volume; RESERVOIR outflow = its drawn volume.
3. `COMPOSITION`: each MIX node's composition equals the exact volume-weighted mix of its
   inputs; OUTPUT/WASTE equal their parent; RESERVOIR is the pure fluid.
4. `MIX_LIMITS`: every MIX node volume within `[min_mix_volume, max_mix_volume]`; exactly two
   distinct parents in v1; not two equal compositions unless `allow_same_composition_merge`;
   denominator ≤ `max_denominator`.
5. `OUTPUT_TOLERANCE`: every OUTPUT node's composition within tolerance of its target and its
   volume equals the target's droplet volume.
6. `DEMAND`: number of OUTPUT nodes per target equals its demand.
7. `TOTALS`: reported totals (reagent by fluid, mixes, waste, outputs) match totals recomputed
   from the DAG.
8. `STORAGE`: replaying nodes in `(step, id)` order, the droplets on chip after each step never
   exceed `storage_capacity`.
9. `TAKE` (same replay): a node only uses droplets already on chip; a take that leaves a
   remainder needs `allow_partial_take` and a remainder ≥ `min_droplet_volume`; a WASTE node
   takes the whole droplet.

### 7.4 I/O
`to_json(path)`, `from_json(path)`, `to_dict()`/`from_dict()`, `to_networkx()` (networkx is
imported lazily; analysis only). The JSON header has `format: "meda-mixing-dag/1"`,
`instance_id`, `producer` (solver or agent name + checkpoint) and the totals; `from_dict` rejects
any other format.

---

## 8. Problem generation: p(s₀) (`meda.generators`)

### 8.1 Random sampler (`sampler.py`)
`sample_instance(cfg: GeneratorConfig, chip, rng, weights=None, exclude=None) -> Instance`.
`GeneratorConfig.from_dict` reads one `configs/generator/*.yaml` and validates it; unknown keys
are an error. Fields:
- `num_fluids`: how many of each kind (`samples`, `reagents`, `buffers`), each a fixed count or
  a `[low, high]` range drawn per instance. Fluid 0 is always a SAMPLE.
- `num_experiments`, `targets_per_experiment`, `demand`, `walk_mixes`: `[low, high]` ranges.
- `composition_sampler`: one of
  - `dirichlet_snapped`: sample on the simplex over `support_size` fluids, then snap to
    `accuracy_denominator` by largest-remainder rounding, every share at least one unit (so the
    support size is exact). Fluid 0 is always in the support.
  - `reachable_walk`: compositions produced by random legal mixes from the reservoirs
    (`generators.synthetic`), so they are reachable under the chip limits. Redrawn until the
    support really has `support_size` fluids, since a short walk can leave one unused.
  - `fixed_list`: explicit `fixed_compositions` (for tests).
- `support_size`: how many fluids are nonzero in a target (dilution = 2: one sample + buffer).
- `droplet_volume`, `tolerance`.
- `step_limit`: an absolute value, or `null` to size it as `step_limit_multiple` times the
  witness plan length. A step limit below the witness length is an error.

**Every sampled instance is solvable by construction.** The sampler builds a *feasibility
witness* — a legal plan meeting all demand — before returning, and raises an error naming the
instance if it cannot. The witness comes from the walk's own recipes (`reachable_walk`, stored
in `metadata["recipes"]` so it stays re-derivable from the frozen instance) or from the naive
construction (10.4). `metadata` records the generator name and version, the config hash, the
composition sampler, `witness_actions`, `witness_source` and, when `weights` is given,
`known_upper_bound` (the witness cost as an exact `"a/b"` string).

`exclude` is a set of canonical instance hashes (8.4) to avoid. A small stage needs it: stage 1
holds only 31 distinct problems, so training must be told which ones the frozen evaluation sets
hold. A stage with nothing left to draw raises rather than looping.

`sample_instances(..., count, unique=True, exclude=None)` draws several; with `unique` it
resamples duplicates and raises if the stage is too small to supply `count` distinct problems.

`dilution_grid(chip, denominator, step_limit, ...)`: every single-target dilution n/D of one
sample with one buffer, deduplicated (the "sweep all targets at an accuracy level" set of 9.2 in
PROJECT_CONTEXT). A deterministic helper, not a p(s₀).

### 8.2 Recipes and synthetic solved instances (`synthetic.py`)
A **recipe** is a mix tree built by random legal mixes from the reservoirs:
- `Draw(fluid_id)` — units taken straight from a reservoir (unlimited);
- `Blend(a, k, b, l)` — build `a`, take k units; build `b`, take l units; mix. The partial take
  is what lets a recipe reach compositions like 3/10 (take 3 of a 5-unit droplet).

A sub-recipe's leftover is discarded, and a recipe used twice is simply built twice: recipes
never share a droplet, which is what makes them *naive* witnesses. `composition_of` gives the
exact product; `witness_plan(instance, recipes)` emits the plan that delivers every target from
its recipe and checks each product lies within the target's tolerance.

`synthetic_instance(...)` marks the products of random recipes as targets, so the plan's cost is
a known **upper bound**. Used for (a) checking agents find plans at least this good, (b) the
`reachable_walk` sampler, and (c) optional imitation warm-start (the AlphaTensor idea:
generating a solved problem forward is easy; solving it backward is hard).

### 8.3 Curriculum (`curriculum.py`)
`Curriculum(stages, rng, weights, exclude=None)` matches the protocol `envs.wrappers.CurriculumWrapper`
expects: `source()` returns an instance source for the current stage, `record(info)` takes one
finished episode's `info`. Each `Stage` is a generator config plus its chip and its exit rule:
`min_success_rate` and, optionally, a mean plan cost within `cost_within` of `reference_cost`,
over the last `window` finished episodes. Promotion only ever moves forward — the stages are a
syllabus, not a controller. `exclude` is the frozen sets' hashes (`BenchmarkSet.hashes()`); every
draw avoids them. `state_dict` / `load_state_dict` carry the stage across checkpoints.

Stage order (matches the research ladder), one config file each:
1. `tiny_dilution` — dilution, one target, M = 2 (DMFB-like), accuracy 32.
2. `dilution_single` — dilution, one target, M = 8.
3. `dilution_multi` — dilution, many targets with demand (sharing becomes useful).
4. `mixture_single` — mixture (3+ fluids), one target.
5. `mixture_multi` — mixture, many targets, many experiments.

An accuracy denominator must factorise into mix volumes the chip allows, or the naive witness
cannot build it (32 = 2⁵ under M = 2; 64 = 8·8 under M = 8).

### 8.4 Benchmarks (`benchmark_io.py`, `scripts/make_benchmark.py`, `benchmarks/`)
A benchmark set is a frozen JSON file (`format: "meda-benchmark/1"`) holding `name`, `version`,
`description`, the generator config it came from, `seeds`, `reference` and the `instances`.
`BenchmarkSet.write` refuses to overwrite a published version (bump `version` instead);
`BenchmarkSet.read(name)` takes a file stem (`tiny_exact_v1`) or a name (newest version).

`reference` maps an instance id to an `InstanceReference(cost, status, producer, detail)`: the
stored answer key. M4 stores the naive witness cost with status `feasible` (an upper bound); M5
replaces it with the exact optimum (status `optimal`) through `with_reference`.

**Lower bounds.** `InstanceReference` has an optional `lower_bound: Rational` (M7; written to
JSON only when set, so older files are unchanged, and never above `cost`), with its producer
named in `detail["lower_bound_producer"]`. A reference is then an interval
[`lower_bound`, `cost`] on the true optimum: `optimal` means the two are equal, `feasible`
means only the upper end is a real plan. Exact search supplies the bound when it times out
(10.2). Every set beyond tier 0 below will hold instances that exact search cannot finish, and
without a bound such an instance can only be graded against another heuristic. A bound is
stored only if it comes from an admissible method; a guessed or extrapolated number is never
written as one (rule 9).

`instance_hash(instance)` is the canonical identity of a *problem*: the fluid kinds, the multiset
of targets and the chip limits. It deliberately ignores the instance id, the metadata and the
step limit, so a training draw cannot slip past the overlap check by being renamed or given more
steps. `check_no_overlap(training, frozen_sets)` raises and names every clash.

**Benchmark tiers.** Each tier answers a different question, and a number from one tier is
never reported as evidence for another tier's question. Tiers 0–1 only show that a producer
works and is near optimal on problems small enough to prove. They say nothing about the problem
class this project targets (MTMD mixture preparation on MEDA with a storage limit). That is
the job of tiers 3–6.

| Tier | Question | Sets | Answer key | Built in |
|---|---|---|---|---|
| 0 Correctness | Does the engine/producer reach a proved optimum? | `tiny_exact_v1` | `optimal` | M5 (done) |
| 1 In-distribution | Does the agent learn its training stage? | `val_stage<k>_v<n>`, k = 1..5 | `optimal`, else [bound, best known] | stage 1: M7–M8; others with their stage |
| 2 Engine check (DMFB) | Does our core reproduce published 1:1 results? | `dmfb_sweep_v1` | exact search, bitscan, published DMFB numbers | M9 |
| 3 Literature | Do we reproduce, then beat, MEDA methods on their own problems? | `literature_<paper>_v1` | the paper's reported numbers + our reproduction | M9 |
| 4 Generalisation | Does the policy transfer to unseen and larger problems? | `unseen_ratios_v1`, `larger_v1` | [bound, best known] | after M8 stage 2 |
| 5 Frontier | How does the agent do where no published method applies? | `frontier_v1` | [bound, best known] | after a general mixture fallback (10.3) |
| 6 Assays | Does it work on ratios from real protocols? | `assays_v1` | [bound, best known] | M9 |

Tier 2 and 3 rules:
- **Tier 2 is a test, not a result.** `dmfb_sweep_v1` holds every n/2^d at d = 4..8 under
  `dmfb_m2` (15, 31, 63, 127, 255 targets). It checks the core against published DMFB results
  (bit scanning step counts, GORMA's optimal reactant use, REMIA/WARA/CoDOS from the UCR
  framework). Under M = 2 our action space is strictly larger than a DMFB method's, so a win
  here is meaningless and is never reported as one.
- **Tier 3 is transcribed by hand from the paper PDF**, one set per paper (WSPM, FacDA, MRCM,
  e-MRCM, then DFMM/MTWM). It records the paper's own protocol: target list or sweep, M or Sc,
  accuracy δ and rounding, cost or objective, mixing model, and whether a parent may take only
  1 unit of an intermediate. Many papers report an aggregate over a sweep ("mean operations over
  all 255 targets at δ = 1/256") rather than per-instance numbers. Those go in a set-level
  `published` block that names the paper, table and value, and are compared against the same
  aggregate from `evaluate`. Numbers from memory or from secondary sources are never stored. A
  baseline counts only after its reimplementation reproduces its `published` block (10.4).

Tier 5 rules:
- `frontier_v1` is the intersection that BASELINES.md section 1.2 identifies as open: k:l
  mixing, N ≥ 3 fluids, many targets, a per-target demand vector, and a storage capacity that
  binds. It is laid out as **scaling axes**, not a random pile. A base instance is varied along
  one axis at a time: number of targets η, number of reagents N, demand, storage capacity,
  accuracy level, and M. That gives cost-vs-size and time-vs-size curves, including the point
  where exact search stops finishing within its budget.
- The reference for each instance is the interval [lower bound, best known], where best known
  is the cheapest of naive (which handles demand), exact search at a long budget, and any
  published method adapted to the instance. Adapted results are labelled `adapted`, for example
  e-MRCM run per target and then replicated to meet demand, and are never presented as the
  published method.
- Before this set can be used, two things are needed: env padding sizes (`F_max`, `T_max`,
  `storage_capacity_max`, `L_max`) that cover its largest instance, which otherwise raises
  under rule 9, and a cost-to-finish fallback that supports mixtures (10.3).
- **Claim rule.** "No published method can attempt these" is false as written. Naive always
  produces a plan, and an exact solver can attempt any instance given unlimited time. Report
  what was measured instead: the instance lies outside every published method's input class,
  exact search did not finish within budget B, and the agent's cost is within x of the lower
  bound and y below the best adapted baseline, at inference time t.

Tier 6: `assays_v1` uses target ratios from real protocols, with a citation per instance: PCR
master mix, the colorimetric protein serial dilution, and the scaled Protein 1/4/5/7 family.
Multiplexed IVD is excluded because it has no splitting and no storage, so it is not a
sample-preparation problem (PROJECT_CONTEXT 3.2).

Sets per tier:
- `tiny_exact_v1` — small enough for exact search; every instance has its optimum stored, proved
  by the exact search (`make_benchmark.py --store-optima`). Contents: the four hand-checked cases
  of 17.3 (so M5's two acceptance criteria are one check) plus every distinct dilution n/8 at
  M = 2. Mixed chips, since 17.3 needs M = 2, 4 and 8.

  Each instance runs its chip at **the accuracy level its own targets need**
  (`max_denominator` = lcm of the target denominators). This is what makes the set live up to its
  name: at the chip default of 256 the search has 129 compositions to wander through and does not
  finish, while at the matched level every instance solves in seconds (4.7 s for all ten). It is
  also how the DMFB literature defines things — accuracy level d means denominators of exactly
  2^d, not "up to 256". Storage capacity was left at the chip default, having been measured to
  make almost no difference.

  This set was regenerated in place in M5 rather than being bumped to v2, the one deliberate
  exception to the freeze rule below: v1 had never backed a reported number, and it did not
  satisfy its own defining property, so keeping a known-unsearchable "exact" set would have been
  a trap. A test now enforces the accuracy-level property so this cannot recur silently.
- `val_<stage>_v<n>` — in-distribution validation per curriculum stage. `val_stage1_v1` holds
  8 of stage 1's 31 problems, disjoint from `tiny_exact_v1`, leaving 16 for training. Its answer
  key holds naive costs only, and naive is up to 2.1× the optimum (7/32: 83/2 against 39/2).

  `val_stage1_v2` keeps the same 8 instances (same hashes) and replaces the answer key
  (`make_benchmark.py:build_val_stage1_v2`). Budgets are node counts, not seconds, so it rebuilds
  identically on any machine: 100,000 expanded nodes at the instance's own accuracy (256), and
  50,000 at the matched accuracy level for a better upper end. A plan found at the matched level
  is legal on the real chip too, and is replayed and validated there before it counts. Result:

  | Target | v1 (naive) | v2 | best known / lower bound |
  |---|---|---|---|
  | 1/16 | 20 | optimal 20 | 1 |
  | 9/32 | 32 | optimal 39/2 | 1 |
  | 7/32 | 83/2 | optimal 39/2 | 1 |
  | 11/16 | 39 | [23, 53/2] | 1.15 |
  | 17/32 | 32 | [23, 29] | 1.26 |
  | 19/32 | 83/2 | [23, 29] | 1.26 |
  | 25/32 | 83/2 | [23, 83/2] | 1.80 |
  | 15/16 | 97/2 | [47/2, 97/2] | 2.06 |

  11/16's 53/2 was proved optimal only at the matched accuracy (16). The real instance allows
  256 and so admits more plans, so 53/2 is a best known there, not a proved optimum (an
  earlier note in this spec had it as proved). The loose intervals (25/32, 15/16) are loose at
  the upper end: nothing beat naive there.
- `dmfb_sweep_v1` — tier 2, see above.
- `literature_<paper>_v1` — tier 3, one set per paper, filled manually from the PDF.
- `unseen_ratios_v1` — compositions excluded from training.
- `larger_v1` — more targets / fluids than any training stage (generalisation test).
- `frontier_v1` — tier 5, scaling axes over the MTMD + storage problem class.
- `assays_v1` — tier 6, protocol-derived ratios.

Train/test separation: training draws from the sampler; evaluation uses only frozen sets.
`check_no_overlap` verifies it by canonical instance hash. On a small stage the separation is
not automatic — training **must** pass the frozen sets' hashes as `exclude` (the `Curriculum`
does), or it will redraw the held-out instances. `BenchmarkSet.hashes()` returns both
`instance_hash` and `problem_hash`; the latter ignores `max_denominator`, because
`tiny_exact_v1` sets it per instance and the same dilution would otherwise hash differently
there and in a stage-1 draw. `check_no_overlap` compares `problem_hash`.

---

## 9. Environment (`meda.envs.sample_prep_env.SamplePrepEnv`)

### 9.1 API (Gymnasium)
```python
env = SamplePrepEnv(env_config, instance_source, cost_weights, cost_to_finish)
# instance_source(rng) -> Instance: sampler, curriculum stage, or FixedInstances(list)
# cost_to_finish(state, instance) -> Rational: truncation penalty (solvers.fallback, M5)
obs, info = env.reset(seed=None, options=None) # options may pass a specific Instance
obs, reward, terminated, truncated, info = env.step(action_index)
mask = env.action_masks()                      # bool[L_max]; same as obs["action_mask"]
```
- `action_space = Discrete(L_max)`. The action is an **index into the current candidate list**,
  which is `rules.legal_actions(state)` in canonical order, padded to `L_max`.
- If the number of legal actions exceeds `L_max`, raise a clear error naming the instance (do
  not truncate silently). `L_max` is a config value.
- Selecting an index that is not a legal candidate raises `IllegalActionError`.
- `reset` raises (naming the instance) if the instance does not fit the padding sizes, if a
  target's `droplet_volume` exceeds `max_mix_volume`, or if no action is legal at the start.
  `reset(options={"instance": inst})` runs a specific instance.
- Episode end: success → `terminated`; dead end → `terminated` + penalty; step limit →
  `truncated` + penalty. The penalty already estimates the remaining cost, so agents must not
  bootstrap a value after a truncated step.
- `info` contains: `instance_id`, `cost_breakdown` (this step), `legal_count`, `stochastic`, and
  at episode end `status` (`TerminalStatus`), `success`, `dead_end`, `episode_totals`,
  `plan_cost` and `penalty` (exact `Rational`s), and the `MixingDAG` (as an object, not in obs).
- The env also exposes `env.state`, `env.instance` and `env.legal` (read-only) for solvers,
  tests and search, and `env.set_instance_source(source)` (used by the curriculum wrapper).
- The step limit belongs to each instance (set by the generator), not to the env config.
- **Snapshot and restore (M8).** `env.snapshot() -> EnvSnapshot` holds everything the env keeps
  between steps: the instance, state, candidate list, droplet birth steps, the DAG deltas and
  totals so far, the plan cost, the done flag and the env's generator. `env.restore(snap)` puts
  it back (it raises if the instance does not fit this env's padding), and `env.observe()`
  rebuilds the current observation. The instance source and `cost_to_finish` are fixed at
  construction and not part of it. A checkpoint taken mid-rollout stores one snapshot per
  worker, so a resumed run continues its half-finished episodes exactly (acceptance (c) of
  M8). Vector envs reach it per worker as the attribute `env_snapshot`
  (`get_attr` / `set_attr`) plus `call("observe")`, and must use
  `AutoresetMode.SAME_STEP`: under `NEXT_STEP` an async worker keeps its pending-reset flag in a
  local variable that cannot be saved.

### 9.2 Observation (`gymnasium.spaces.Dict`, padded, float32 unless noted)
| Key | Shape | Content |
| --- | --- | --- |
| `sources` | [S_max, D_src] | One row per reservoir (rows 0..F−1) and per droplet on chip (rows F..F+n−1, canonical order) |
| `source_mask` | [S_max] int8 0/1 | Valid rows |
| `targets` | [T_max, D_tgt] | One row per target |
| `target_mask` | [T_max] int8 0/1 | Valid rows |
| `candidates` | [L_max, D_cand] | Features of each legal action (below) |
| `cand_index` | [L_max, 3] int32 | (op, row of src_a or droplet, row of src_b or target index); op 0 = Mix, 1 = Output, 2 = Discard; −1 if unused |
| `action_mask` | [L_max] int8 0/1 | Valid candidates (`env.action_masks()` returns the same as bool) |
| `globals` | [D_glob] | Step fraction, storage fraction, remaining demand fraction, optional preference weights |

`S_max = F_max + storage_capacity_max`. Compositions are padded to `F_max` fluids. All sizes come
from `configs/env/sample_prep.yaml`. Every float feature lies in [0, 1]; volumes are divided by M.

### 9.3 Features (`encoding.py`) — the network never does arithmetic

Every result composition and distance is formed as exact integers (numpy int64) and divided
only at the end, so each float is the correctly rounded exact value; "satisfies" flags are
compared exactly in integers. To keep int64 exact, `max_denominator` and every target
composition/tolerance denominator must be ≤ 2^20 (checked at reset), and tolerances ≤ 1. The
denominator feature is `1 − 1/den` (bounded, 0 for pure fluids).

**Source row:** composition (F_max floats), volume / M (1 for reservoirs), is_reservoir,
fluid-kind one-hot (reservoirs only), denominator feature, L∞ distance to each open target
(T_max; 1 for met or padded targets), "satisfies open target t" flags (T_max), age (actions
since the droplet was made, / step limit).

**Target row:** composition, tolerance, droplet volume / M, remaining demand / demand, met flag.

**Candidate row (afterstate features):** op one-hot; k, l, k + l (scaled; for Output k is the
delivered volume, for Discard the droplet volume); the **resulting** composition (exact, then
converted to float); its denominator feature; L∞ distance from the result to each open target;
whether the result satisfies any open target; this action's reagent units by fluid kind (/ M);
waste it creates (/ M); storage after the action (/ capacity). Providing the result of each candidate is the
"afterstate" idea (TD-Gammon): the network judges outcomes instead of predicting them.

### 9.4 Vectorisation and speed
- Must work with `gymnasium.vector.AsyncVectorEnv` (picklable, no global state).
- `scripts/profile_env.py` reports steps/second for each generator config.
- Target: at least 10,000 env steps/second in total with 16–32 async workers on the
  workstation for curriculum stages 1–3. Optimise only after profiling (cache legal-action
  enumeration per canonical state; `numba` only if needed).

### 9.5 Wrappers (`wrappers.py`)
- `EpisodeStats`: accumulates totals for logging.
- `CurriculumWrapper`: switches generator stage on promotion.
- `PreferenceWrapper`: samples w per episode when multi-objective mode is on.

---

## 10. Solvers (`meda.solvers`)

### 10.1 Interface (`base.py`)
```python
class Solver(Protocol):
    name: str
    def solve(self, instance: Instance, budget: SolveBudget) -> SolveResult: ...

class PlanBuilder:   # shared by the solvers and by the generators' feasibility witnesses
    """Emits a plan through `core.rules.apply`, so every plan is legal by construction."""
# SolveResult: instance_id, producer, status, dag: MixingDAG | None, totals: StepCost | None,
#   cost: Rational | None (exact plan cost, no penalty), wall_time, nodes_expanded,
#   lower_bound: Rational | None (a proved bound on the optimum when exact search times out; 10.2)
# status: "optimal" | "feasible" | "timeout" | "infeasible" (solvers) | "failed" (an agent
#   episode that did not meet all demand; not a proof of infeasibility)
```

### 10.2 Exact search (`exact_search.py`)
Uniform-cost search (Dijkstra) over `PrepState.canonical_key(include_step)`, edge cost = scalar
step cost (non-negative, exact `Rational`), goal = success. Budget: max nodes, max wall time.
This is the answer key for `tiny_exact_v1`.

`heuristic` selects A*: `fluid_bound` (default) or `zero` (plain Dijkstra). The bound comes from
conservation of each fluid — every droplet still to be delivered holds exactly
`droplet_volume * composition[f]` units of fluid f, which can only come from what is on chip or
from new draws — so remaining cost is at least the weighted shortfall. Mix and waste costs are
ignored, keeping it admissible. Re-opening is allowed (a pop whose `g` is stale is skipped), so
optimality holds even if the heuristic is not consistent.

**Branch and bound.** The search starts from an incumbent plan (the naive baseline) and prunes
any node whose `g + h` already reaches the incumbent cost; if nothing cheaper is found, the
incumbent is proved optimal. Without it the search does not finish: cheap buffer dilutions
wander into fine compositions no target needs.

The pruning was added after M7 planning (it was documented but missing). It leaves the ten
`tiny_exact_v1` optima unchanged, and it does not rescue the four `val_stage1_v1` timeouts,
because naive is too loose an incumbent there.

**Lower bound on timeout (M7).** When a budget stops the search, it returns
`lower_bound = min(f_min, incumbent cost)` instead of nothing, where `f_min` is the smallest
`g + h` on the frontier. This is a proved bound on the optimum: with an admissible `h`, some
node of an optimal plan is always on the frontier with `f ≤ C*`. Stale heap entries only carry
a larger `f`, so they cannot lower the minimum. Nodes pruned by the incumbent have
`f ≥ incumbent`, which is why the incumbent caps the bound. Status stays `timeout`, and the
result carries `lower_bound` together with the best known plan. A goal generated more cheaply
than the incumbent replaces it (after a replay confirms the plan and its cost), which tightens
the pruning and improves the best known end of the interval. A test stops the search at
several node budgets on every `tiny_exact_v1` instance and checks the bound never exceeds the
stored optimum. With no
search at all, `h(root)` is already a valid but weak bound. Tighter instance-level bounds, such
as an LP relaxation over fluid flows, can be added behind the same field, each named in
`lower_bound_producer`. This bound is what lets tiers 1 and 3–6 of 8.4 be graded wherever exact
search does not finish.

**`include_step`** (default true) puts the step count in the key, and it is needed for
correctness in general: with it excluded, a state settled cheaply at step 10 blocks a pricier
arrival at step 3 that still had budget to finish, so the search can return a suboptimal cost or
wrongly report infeasible. Set it false only when the step limit cannot bind.

**Status.** `optimal` only if the search finished; `infeasible` only if the reachable space was
exhausted with no plan; `timeout` if a budget stopped it. The distinction matters: proving 3/10
unreachable under `dmfb_m2` needs exhaustion, which only happens on a tightly bounded instance
— at the chip's default 10 droplets and 30 steps the same search can only report `timeout`.

### 10.3 Fallback cost-to-finish (`fallback.py`)
`NaiveCostToFinish(weights)(state, instance)`: waste everything on chip, then build each
remaining droplet from the reservoirs with the naive construction. Because it is the cost of a
completion that really exists, it can never be below the optimal remaining cost, which is what
the truncation penalty needs — otherwise an unfinished plan could look cheaper than a finished
one. It is hypothetical, so it is computed on a relaxed instance and need not fit the step limit
that was just exceeded. The naive part depends only on the remaining demand, so it is cached per
(instance, remaining demand); the object is picklable for `AsyncVectorEnv`.

**Scope.** The naive construction must be able to build the remaining targets, which covers
every dilution stage. A mixture whose denominator exceeds the mix limit raises
`UnsupportedFallbackError` naming the instance rather than inventing a number (rule 9); use
`supports(instance)` to check before training. Mixture stages therefore need a general baseline
first (reserved for M9). The same estimate also serves as tabular Q's value prior (11.3).

### 10.4 Baselines (`baselines/`)
- `naive.py` (built in M4, which needs it for acceptance): each target droplet made
  independently, no sharing. Two exact constructions — it hits the target composition even when
  tolerance > 0:
  1. **chain** (any number of fluids), when `composition.den <= max_mix_volume`: draw `num[f]`
     units of each fluid in turn, mixing the running droplet with the next reservoir;
  2. **mixed radix** (two fluids), when `den` factorises into factors within
     `[min_mix_volume, max_mix_volume]`: write n in that mixed radix and build
     `s_i = (a_i + s_{i-1}) / e_i` with two 2-input mixes per digit. Under M = 2 this *is*
     bit-scanning. It needs partial takes (it carries 1 unit of the previous droplet).

  Outside that scope (a mixture whose denominator exceeds M, or a prime factor above M)
  `naive_plan` returns `None`: **not found**, never a claim of infeasibility. `NaiveSolver`
  reports status `infeasible` for it, which means only that this construction found nothing.
- `bitscan.py`: classic 1:1 binary dilution for two-fluid targets (DMFB literature). Valid only
  under a 1:1 chip, two fluids, and a dyadic target; `applies_to(instance)` gives the reason when
  it does not apply, and the solver then reports `infeasible`. Under M = 2 the naive mixed-radix
  construction *is* bit-scanning (every radix is 2, so the digits are the bits of n), so this
  baseline checks its preconditions and delegates rather than duplicating the algorithm; a test
  asserts the two agree on every dyadic target. It stays a separate named baseline because the
  literature reports it as one.
- `facda.py`: factorisation-based MEDA dilution. A **stub** that raises, with `verified = False`,
  reserved for M9. Two things must be settled before writing it: the author list is unconfirmed
  (Kundu et al. vs Bhattacharjee et al.), and the paper's reported results must be transcribed
  into `literature_v1` so the reproduction test has something to compare against. A
  plausible-looking approximation would produce numbers nobody has checked.
- Further published methods (multi-target MEDA mixing, demand-driven mixture preparation) are
  added later behind the same interface, each with a reproduction test.

---

## 11. Agents (`meda.agents`)

### 11.1 Interface (`base.py`)
```python
class Agent(Protocol):
    name: str
    def act(self, obs, mask, deterministic: bool) -> int: ...
    def save(self, path) -> None: ...
    @classmethod
    def load(cls, path) -> "Agent": ...
```
Plus a helper `run_episode(agent, env, instance, deterministic=True) -> SolveResult` so every
agent is evaluated exactly like a solver (it returns the episode's `MixingDAG`, complete or not).

### 11.2 Random agent
Uniform over legal candidates. Used to test the env and as the weakest baseline.

### 11.3 Tabular Q-learning (`tabular_q.py`)
For `tiny_exact_v1`-size problems only. Keys: `state.canonical_key(include_step)` and
`action.canonical_key(state)` — never the candidate index, whose meaning depends on the state.
ε-greedy with a decay schedule to a **floor** (0.1, not 0: convergence needs every pair visited
often). γ = 1.

Because the keys come from the exact state, the agent cannot work from the observation alone (a
padded float array does not identify a state). It reads `env.state` and `env.legal` through
`bind(env)`, and `run_episode` works once bound. That dependence is the honest reason the
tabular rung ends here.

Three settings decide whether it reaches the optimum at all, each measured:

- **`include_step` (default true).** These instances have tight step limits (4 for the 1/4 case
  at M = 4), so the same droplets are worth different amounts early and late, when there is no
  longer room to finish. Sharing one entry blurs the two and the greedy policy settles for a
  worse plan.
- **`step_size` (default 0.5, a constant).** 1/N(s, a) averages every target a pair ever saw,
  and with bootstrapping those targets drift as the future is learned, so early pessimistic ones
  keep their weight. Measured on 1/4 at M = 4: with 1/N, Q(1:1) settles at −16.02 after 1248
  visits where the truth is −15.0, and the agent keeps the worse 1-mix plan.
- **`prior` (default `naive`).** The value of an entry with no data. `naive` is
  `-cost_to_finish` (10.3): pessimistic but realistic, so greedy trusts what it learned and ε
  does the exploring. `fluid_bound` and `zero` are optimistic; the bound is loose (≈8.8 against
  a true 36.5 on 7/8), so every unvisited action outranks everything learned and the agent
  chases optimism instead of settling. With an optimistic prior the greedy policy also prefers
  visited actions, since the prior is then a guess rather than a value.

Truncation and dead ends are terminal with no bootstrap: the environment already adds
`cost_to_finish` to that reward, so bootstrapping would count the rest of the plan twice.

Acceptance: matches the exact optimum on every tiny instance it is trained on, **except 3/10 at
M = 8** (see 16).

### 11.4 PPO with a set encoder (`ppo/`)
**Network (`networks.py`):**
1. Source encoder: MLP on each source row → d_model.
2. Target encoder: MLP on each target row → d_model.
3. A learned global token.
4. Set encoder: Transformer encoder (2–4 layers, 4–8 heads, **no positional encoding**) over
   [global token; sources; targets] with key padding masks. Output: contextual embeddings.
5. Candidate scorer: for each candidate, gather the embeddings referenced by `cand_index`
   (source a, source b or target), concatenate with the candidate's own features, MLP → logit.
6. Masked categorical: set logits of invalid candidates to a large negative value before
   softmax (`policy.py`).
7. Value head: MLP on the global token → scalar.

**Algorithm (`ppo.py`, CleanRL style):** vectorised rollouts; GAE(λ); clipped surrogate;
value loss (optionally clipped); entropy bonus; gradient-norm clipping; learning-rate
annealing; several epochs of minibatches per rollout. γ = 1.0 by default (finite episodes).

**Default hyperparameters (`configs/agent/ppo.yaml`), starting points only:**
```yaml
d_model: 128
n_layers: 3
n_heads: 4
num_envs: 32
rollout_steps: 128
gamma: 1.0
gae_lambda: 0.95
clip_coef: 0.2
ent_coef: 0.01
vf_coef: 0.5
max_grad_norm: 0.5
lr: 3.0e-4
update_epochs: 4
num_minibatches: 8
total_env_steps: 5_000_000
```

**Logged every update:** mean episode return, success rate, dead-end rate, mean cost and each
cost component, policy entropy, approximate KL, clip fraction, value explained variance,
legal-candidate count, env steps/second.

### 11.5 Reserved: search (`mcts/`, milestone 9)
AlphaZero-style MCTS (PUCT) using the PPO network's policy and value, on `core.apply` (pure,
so trees can be expanded without copying envs). Trained by making the policy match search visit
counts. Not built now; keep `core` pure so it can be added without refactoring.

### 11.6 Optional cross-check
A thin adapter so `sb3_contrib.MaskablePPO` can train on the same env with a custom feature
extractor. Useful to confirm the in-house PPO is not buggy.

---

## 12. Metrics and evaluation (`meda.analysis`)

### 12.1 DAG metrics (`dag_metrics.py`)
`dag_metrics(dag, instance, schedule) -> DagMetrics`, all exact integers, computed from the DAG
alone (totals are recounted from the nodes, not read from the producer's report).

**Proxy metrics** (what the sample-prep literature reports): `mix_ops`, `outputs`,
`total_reagent` and drawn volume by kind (`sample_volume`, `reagent_volume`, `buffer_volume`),
`waste_volume`, `distinct_intermediates` (distinct MIX compositions) and `sharing_count` (MIX
nodes whose fluid reaches OUTPUT nodes of two or more distinct targets; naive is always 0).

**Structural metrics** (predict how hard the plan is to execute on the chip; no grid needed):
an abstract list schedule. Each MIX or OUTPUT starts as soon as its inputs exist and the mix cap
allows, ties broken by the plan's own (step, id) order. WASTE is instantaneous and happens once
every other use of its droplet has started, so it ends a droplet's life but never adds to the
makespan. Reservoirs are available at slot 0. From it:
- `makespan`: last MIX/OUTPUT finish, in slots;
- `critical_path`: the makespan with unlimited concurrent mixes (longest duration chain);
- `peak_droplets`: most MIX droplets alive at once, a droplet living from its mix's start to
  the start of its last use (to the end, in an unfinished plan). It may exceed
  `storage_capacity`, since the schedule runs operations in parallel that the plan ran in turn;
- `storage_slots`: summed slots that droplets wait on chip between being made and last used.

Config (`eval.schedule`, physical assumptions under rule 8): `mix_duration`,
`output_duration` (slots, >= 1) and `max_concurrent_mixes` (null = unlimited).
These are the "tier-1" cost quantities, unrelated to the benchmark tiers of 8.4.

### 12.2 Evaluation (`evaluate.py`, `compare.py`, `plots.py`, `evaluate_cli.py`)
A `Producer` has a `name` and `produce(instance) -> SolveResult`. `SolverProducer(solver,
budget)` wraps a solver; `AgentProducer(agent, env, deterministic, best_of_k)` wraps an agent
through `run_episode` (greedy, or the cheapest successful plan of k sampled episodes; k > 1 with
`deterministic` is an error). `build_env(cfg, instances)` builds the env both training and
evaluation use.

`evaluate(producer, benchmark, schedule) -> pandas.DataFrame`, one row per instance: benchmark,
instance id, producer, status, `success` (status `optimal` or `feasible`), `cost` (float) and
`cost_exact` (`"a/b"`), the reference (`reference_cost`, `reference_status`, `lower_bound`),
`num_targets`, `total_demand`, wall time, nodes expanded, every metric of 12.1, and gaps.
Gaps are computed exactly from rational costs and only for a successful plan (an unfinished
plan's cost is not comparable):
- `gap_to_optimal` = (cost − optimal) / optimal, only where the reference is `optimal`. A gap
  against a bound or a heuristic is never written into this column.
- `gap_to_best_known` = (cost − reference cost) / reference cost, for every stored reference
  (equal to `gap_to_optimal` where that is proved). Negative means the producer found a plan
  cheaper than every stored reference: a candidate for a new `_v<n+1>` answer key, never
  written back into a frozen set.
- `gap_to_lower_bound` = (cost − lower_bound) / lower_bound, where a positive bound is stored.
  An **upper bound on the true gap to optimal**, the figure reported when no optimum is proved.
- `bound_ratio` = best known / lower_bound per instance: how loose the interval is. A large
  ratio means the instance cannot separate a good producer from a mediocre one.
- `gap_to_<baseline>` for each configured baseline (`compare.add_baseline_gaps`), where both
  plans succeeded.

Every produced DAG is validated (section 7.3), complete if the plan succeeded and partial
otherwise; a failed validation is a hard error, not a metric.

`evaluate_cli` grades `eval.producers` and `eval.baselines` (random, naive, bitscan,
exact_search; tabular Q is graded by `meda.train`, PPO joins in M8) on `eval.benchmark`, with
solvers given `eval.solver_seconds` per instance. Outputs per run: `results.parquet` (one row
per producer and instance), `summary.md` (the benchmark's tier, success rate, and n / mean /
median / 90th percentile per metric over successful plans, plus exact costs per instance) and
`plots/` (cost vs the first baseline, gap-to-best-known histograms with a shared x range, cost
vs number of targets; each producer has a fixed colour across all figures).

### 12.3 Evaluation protocol
- Agents evaluated deterministically (argmax) and optionally with sampling (best of k).
- Fixed seeds; at least 3 training seeds per reported configuration.
- Report on all frozen sets of section 8.4, including `larger_v1` for generalisation. Each
  table states its benchmark tier, and a result answers only its tier's question.
- Report solver and agent wall time alongside cost on every tier. On tier 5, plot both against
  each scaling axis.

---

## 13. Configuration, runs and reproducibility

- Hydra root `configs/config.yaml` with defaults: chip, costs, generator, env, agent, train, eval.
- `meda.train` trains one agent. Tabular Q (M6) trains on a frozen benchmark set, one table per
  instance (`train.per_instance`), because it keys on exact states and the acceptance criterion
  is per instance; PPO (M8) will train from the sampler. Each run writes the resolved config,
  one checkpoint per table, and the greedy policy graded by `evaluate` (12.2):
  `results.parquet`, with the training statistics as extra columns, and `summary.md`.
- Example commands:
  ```bash
  python -m meda.train agent=tabular_q
  python -m meda.train agent=tabular_q train.benchmark=tiny_exact_v1 agent.episodes=40000
  python -m meda.train agent=ppo generator=dilution_multi chip=meda_m8
  python -m meda.train -m agent.lr=1e-4,3e-4 seed=0,1,2        # sweep
  python -m meda.evaluate_cli eval.producers=[random,naive,exact_search] eval.benchmark=tiny_exact_v1
  ```
- Each run directory stores: the resolved config, git commit hash and dirty flag, seed, library
  versions, TensorBoard logs, checkpoints (`last.pt`, `best.pt` by validation cost), and
  evaluation results.
- `utils/seeding.py` seeds Python, NumPy and Torch, and creates per-worker generators.
- Checkpoints contain model weights, optimiser state, step counters and the config, so training
  can resume.

---

## 14. Reserved later stages (`meda.pipeline`)

Do not implement yet. Define only these interfaces in `pipeline/interfaces.py` so the DAG format
is designed with them in mind:
```python
class Scheduler(Protocol):
    def schedule(self, dag: MixingDAG, chip_spec) -> "Schedule": ...      # start time + resource per node
class Router(Protocol):
    def route(self, requests: list["RouteRequest"], chip_state) -> "RouteResult": ...  # (droplet, source, target) list per time slot
class CostOracle(Protocol):
    def evaluate(self, dag: MixingDAG) -> "TrueCost": ...                 # completion time, washes, ...
```
A `CostOracle` result will later replace or refine the reward (tier-2 cost). Plan for it: cache
results by DAG hash, and allow a learned surrogate to predict it.

---

## 15. Open physical decisions (all must stay configurable)

| Decision | Default here | Where it lives | To confirm with |
| --- | --- | --- | --- |
| Maximum mix volume M on the target platform | 8 | `chip.max_mix_volume` | Supervisor |
| Whether partial takes (pseudo-dispense) are physically allowed, and minimum droplet volume | allowed, 1 | `chip.allow_partial_take`, `chip.min_droplet_volume` | Supervisor |
| Storage limit as droplet count vs total volume | droplet count | `chip.storage_capacity` | Supervisor |
| Accuracy (max denominator) and tolerance metric (L∞) | 256, L∞ | `chip.max_denominator`, target tolerance | Assay requirements |
| Cost weights per fluid kind | 10 / 2 / 0.5 | `costs` | Supervisor, assay |
| Dilution first or mixture first | dilution first (curriculum) | `generator` | Supervisor |
| Whether a mix may take inputs from more than two sources | no | `rules` (would change the action set) | Later |
| Split-error model for phase 2 | none | `core/noise.py` | Literature / hardware data |

---

## 16. Milestones and acceptance criteria

Implement strictly in order. Each milestone ends with all tests passing, `ruff` and `mypy` clean.

**M0 — Scaffold.** Layout of section 4, `pyproject.toml`, pre-commit, Hydra root config, empty
modules with docstrings, CI-style `make test` (or `uv run pytest`).
*Accept:* `uv run pytest` runs (with placeholder tests); `python -m meda.train --help` works.

**M1 — Exact core.** `composition`, `fluids`, `chip`, `problem`, `state`, `actions`, `rules`,
`costs`, `noise` (NoNoise).
*Accept:* all core tests in section 17.1 pass, including the property tests.

**M2 — DAG.** Builder (from `DagDelta`s), validator, JSON I/O, networkx export.
*Accept:* replaying any legal random episode yields a DAG that validates; tampering with any
volume or composition makes the validator fail with the right error.

**M3 — Environment.** `SamplePrepEnv`, `encoding`, wrappers; random agent.
*Accept:* passes `gymnasium.utils.env_checker.check_env`; random agent runs 10,000 episodes on
`tiny_dilution` with no exception; every chosen action is legal; observation shapes fixed;
`profile_env.py` reports steps/second.

**M4 — Generators and benchmarks.** Sampler, recipes, curriculum, benchmark I/O,
`make_benchmark.py`; create `tiny_exact_v1` and `val_stage1_v1`. Includes `naive.py` and
`PlanBuilder` (moved here from M5): the acceptance criterion needs the naive construction.
*Accept:* same seed gives identical instances; every sampled instance is solvable (its witness
replays to success and its DAG validates); training draws can be held off the frozen sets
(`exclude` + `check_no_overlap`); every stage fits the env padding sizes.

**M5 — Solvers.** Exact search, fallback, bitscan; `facda` as a stub marked unverified (naive
landed in M4).
*Accept:* the hand-checked cases of 17.3 give the stated optima, with cases 2 and 3 read as
*mix-count* statements (see 17.3); exact search proves and stores an optimum for all ten
instances of `tiny_exact_v1` (4.7 s in total).

**M6 — Tabular Q-learning.**
*Accept:* reaches the exact optimum on 9 of the 10 `tiny_exact_v1` instances it is trained on,
at 8,000 episodes each.

The exception is `tiny-three-tenths-m8` (3/10 at M = 8), and it is the rung the research ladder
expects to fail: "larger M/precision breaks the table → function approximation". A step there
offers 28 candidates at the start and ~80 later, against ~4 under 1:1 mixing, and the optimal
plan is 4 actions deep, so a random walk finds it with probability of order 1e-6 an episode and
a table has nothing to generalise with. The two priors fail in opposite ways, which is the
clearest statement of the problem: optimistic (`fluid_bound`) finds the optimum but never
settles on it, while pessimistic (`naive`) settles but never finds it — at 60,000 episodes and
180,674 table entries its best plan was still 29.00 against the optimum of 20.50. This motivates
M8 rather than blocking M6, and a test records the gap instead of hiding it behind a bigger
budget.

**M7 — Evaluation harness.** `dag_metrics`, `evaluate`, `compare`, `plots`, CLI.
*Accept:* produces `results.parquet` and `summary.md` for random agent, naive, exact search on
`tiny_exact_v1`; gaps computed correctly.

Met: exact search lands on all ten stored optima (gap 0), naive's gaps equal those computed from
the stored costs (it matches 5 of 10; 3/8 costs 27 against 16), and random succeeds on 2 of 10
within the step limits, its failures carrying no gap (`pytest -m slow`, test_evaluate_cli). Still
to do before M8: nothing; the exact-search timeout bound (10.2) and `val_stage1_v2` (8.4)
followed.

**M8 — PPO with set encoder.** Networks, masked policy, rollout, PPO, training entry point,
checkpointing, logging.
*Accept:* (a) overfits a single fixed tiny instance to its optimum; (b) on curriculum stage 1,
success rate > 95% and mean `gap_to_best_known` < 5% on `val_stage1_v2` (on its three proved
instances that is the gap to optimal; on the other five it is a gap to the best plan known, and
`gap_to_lower_bound` is reported alongside as the most the true gap could be); (c) a run resumes
from a checkpoint with identical subsequent metrics (same seed). Training excludes the hashes of
`tiny_exact_v1` and `val_stage1_v2`.

**M9 — Reserved.** MCTS / AlphaZero-style search; literature baselines with reproduction tests;
preference-conditioned training; noise model; pipeline interfaces to scheduler/router.

---

## 17. Tests

### 17.1 Core (unit + property-based with `hypothesis`)
- Composition: canonical form invariants (sum = den, reduced); `mix` exactness; mixing is
  symmetric (`mix(a,k,b,l) == mix(b,l,a,k)`); mixing a composition with itself returns it.
- Volume conservation: for any random legal action sequence, total reagent taken = volume on
  chip + volume output + volume wasted, after every step.
- Composition sum: every droplet composition sums to exactly 1 after every step.
- Legality soundness: every action in `legal_actions` applies without error; a sample of
  illegal actions (wrong volumes, M exceeded, storage exceeded, out-of-tolerance outputs)
  raises `IllegalActionError`.
- Determinism: same instance and action sequence → identical states, costs and DAG.
- Canonical keys: permuting droplet uids does not change `canonical_key()`.
- No float type appears anywhere in `PrepState` or `StepCost` (type check in a test).

### 17.2 Environment
- `check_env` passes; mask equals the legal set; padded rows are masked; selecting a masked
  index raises; episode ends exactly on success, truncation or dead end; truncation penalty
  applied.

### 17.3 Hand-checked solver cases (fluids: S = sample at 1.0, B = buffer; one-unit outputs; tolerance 0)

These cases were written as **mix-count** statements. The exact search minimises *weighted cost*,
and with the default weights (sample 10, mix 1, waste 1) the two objectives disagree on cases 2
and 3, where saving a unit of sample is worth more than an extra mix. Each case below gives both.
The optima proved by the exact search are stored in `tiny_exact_v1`.
1. **Target 1/4, M = 2 (1:1 only).** Optimal mix count = 2: (S1 + B1 → 1/2, volume 2), then
   (1 unit of 1/2 + B1 → 1/4, volume 2), output 1 unit. Reagent: S = 1, B = 2. Waste at end: 2
   units (1 of 1/2, 1 of 1/4).
2. **Target 1/4, M = 4.** One mix suffices: S1 + B3 → 1/4 (volume 4); output 1 unit; waste 3 —
   but it costs 31/2, because it draws 3 units of buffer and wastes 3. The 2-mix plan of case 1
   is legal here too and costs 15, so **the cost optimum is 15 with 2 mixes**. Weighting waste
   and buffer at zero makes the single mix optimal instead, which is the "test both weightings"
   check.
3. **Target 3/10, M = 8.** Not reachable in one mix (denominators ≤ 8). Reachable in two:
   S2 + B3 → 2/5 (volume 5); then 3 units of 2/5 + B1 → 3/10 (volume 4) — 2 mixes, cost 29.
   **The cost optimum is 41/2 with 3 mixes**: S1 + B1 → 1/2; 1 unit of 1/2 + B4 → 1/10; the other
   unit of 1/2 + 1 unit of 1/10 → 3/10. It draws 1 unit of sample instead of 2, and sample costs
   10 a unit against 1 for a mix. Charging mixes heavily recovers the 2-mix plan. Under
   `dmfb_m2` with tolerance 0 it is infeasible (3/10 is not a dyadic fraction); proving that by
   search needs a tightly bounded instance, since infeasibility means exhausting the space.
4. **Sharing: targets 3/4 and 1/4, one droplet each, M = 2.** Optimal = 3 mixes: S1 + B1 → 1/2
   (volume 2); 1 unit of 1/2 + S1 → 3/4; 1 unit of 1/2 + B1 → 1/4. Without sharing the naive
   baseline uses 4 mixes. The exact search must find 3.

### 17.4 Slow tests
Long acceptance runs (e.g. M3's 10,000 random episodes) carry `@pytest.mark.slow` and are
skipped by default; run them with `uv run pytest -m slow` when closing a milestone.

### 17.5 Agents and evaluation
- Tabular Q converges on cases 1–4.
- PPO smoke test: 2,000 steps on a tiny instance without NaNs; loss decreases.
- Evaluation: a hand-made results table gives the expected gaps.

---

## 18. Coding conventions

- Type hints everywhere; `mypy --strict` on `meda.core`.
- Frozen dataclasses (`slots=True`) for domain objects.
- Docstrings state units (volume units, not microlitres) and invariants.
- No silent clipping, truncation or fallback: raise with a message that names the instance and
  the violated limit.
- Public functions in `core`, `envs`, `solvers` get a unit test before or with the implementation.
- Notebooks may import from `src/meda`; `src/meda` never imports from notebooks.
- Any change to a public interface or to the DAG format updates this SPEC in the same commit.
