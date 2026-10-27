# MEDA Design Automation — Codebase Specification

Version 0.28 · 2026-10-27

Changes in 0.28 (planner-free hint ablation, 9.3; paper log E-31, Q-105): the env config gains
`hint_source` (optional, default `planner`, unchanged behaviour). With `problem`, no planner is
called for hints: the recipe columns carry the fewest mixes along a pure-fluid chain of depth
up to 4 (`hints.EXACT_DEPTH`, the aim itself at 0), a fact of the problem's arithmetic, instead
of the distance along the tree's and the sharing pass's recipes; the chain columns are
unchanged. `HintTables.recipe_left` now takes the droplet's volume. `scripts/imitation_check.py`
takes `hints=planner|problem`. (The unmerged literature branch also calls itself 0.28; it
becomes 0.29 when merged.)

Changes in 0.27 (teacher moves on identical droplets, 11.4; paper log F-16): the candidate list
offers one move per class of identical droplets, while a teacher's plan names a particular
droplet. Demonstrations now match a teacher action the list does not hold by canonical key and
keep a uid map, so the teacher's later actions follow the twin the env used
(`imitation.UidMap`). Stage 1–2 demonstrations are unchanged; stages 3–5 no longer crash.

Changes in 0.26 (warm start in PPO, 11.4): `train.init_from` starts a fresh PPO run from saved
weights (the copy test's network), and the PPO config gains `imitation_coef`, `imitation_steps`,
`imitation_batch`, `imitation_problems`: an imitation term on teacher demonstrations whose
weight falls linearly to 0 (off by default). Old PPO checkpoints need the four new keys.

Changes in 0.25 (finish from here; paper log E-16, E-18, E-19, D-32): (1) the truncation /
dead-end penalty is computed by `cost_to_finish` chosen in `configs/env` (5.8, 10.3): `share`
(new default) = the cheaper of the sharing pass started from the current state, reusing every
droplet on chip (`share.finish_from`, `SharingCostToFinish`), and the constructions' rebuild;
`construction` = the old rebuild only. The rebuild charged 2–25× the real cost to finish along
the teacher's own plans (E-18), so building an intermediate earned no credit at truncation.
(2) Ten **hint columns** on the candidate row (9.3, D_cand 17 → 27): one-hots of the fewest mixes
from a Mix result to an open target along the tree's and the sharing pass's recipe, and along
a pure-fluid chain (`solvers.hints`, injected into the env like `cost_to_finish`). (3) The
imitation teacher is the sharing pass by default (11.4). Old checkpoints do not load into the
new network (input width changed).

Changes in 0.24 (teacher study, paper log E-17): a new baseline, the **sharing pass**
(`baselines/share.py`, 10.4), builds the targets on one chip, keeps leftovers in a pool and
reuses them across targets; it is never dearer than `cheapest_construction` and reaches the
proved optimum on 51 of 65 small problems (the construction: 34). `PlanBuilder` gains
`reuse(composition, need)` (10.1), which returns None, so the constructions are unchanged; the
split tree asks it before building each piece. `cheapest_construction`, the failure penalty
and the stored answer keys are unchanged.

Changes in 0.23 (training approach; from the pilot 2 diagnosis, paper log E-11, E-14, D-30,
D-31): an imitation warm start is planned (11.4, *Imitation warm start*): the network first
copies `cheapest_construction` on stages 1–2 only, where it is near-optimal, and the imitation
term then fades to 0 under PPO so RL can override the teacher. An imitation-only copy test
comes first. No interface changes yet.

Changes in 0.22 (training signal; found by the 2026-10-22 pilot, which learned to give up): (1)
the truncation / dead-end penalty is `failure_penalty_factor` (env config, default 2) times the
construction's cost to finish (5.8). At 1, a deliberately failed episode cost only 1.03–1.33×
the best known plan on the stage validation sets, so any finished plan worse than that earned
less than failing; a slow test now requires failing to cost at least 1.8× the best known plan.
(2) Step-limit rule `depth_support` (8.1), used by `track2_v1`: the depth base also counts
(required fluids − 1) mixes per delivery, and the limit is floored at 2× the witness length.
Under `depth`, 13 of 40 stage-2 validation problems allowed exactly the witness plan. The
`_v2` sets are rebuilt with these limits (same problems and best known). (3) A regime whose
stage count × `max_stage_fraction` exceeds 1 is refused, since forced promotions could leave the
last stages untrained (5 × 0.3); the default is now 0.2 (8.3).

Changes in 0.21 (grading): the exact-search bound (10.2) was not admissible on tolerant targets
(it counted each target's exact shares, though a droplet within tolerance may hold up to ε less
of a fluid; a 51% ± 2% target bounded at 66.26 against an optimum of 66). The bound now uses
`share - tolerance`, rounds each fluid's draws up to whole units, and adds waste and mix terms;
it is consistent. The answer keys of the `track2_v1` sets change, so they are rebuilt as
`val_s1_v2` … `val_s5_v2` and `heldout_combo_v2` (same instances, same best known; 8.4); the
design points at them. The `_v1` files stay frozen (the 2026-10-22 pilot ran on them; promotion
uses the gap to best known, which is unchanged).

Changes in 0.20 (training speed): the workers collector, `train.collector=workers` (11.4): each
worker process owns one env and a frozen copy of the policy for a whole rollout (synchronous
on-policy PPO with decentralised actors), with version-stamped rollouts, a learner-side check of
the collected log-probabilities and values, per-env sampling generators saved in checkpoints,
and the old collector (`lockstep`) kept as the default. Env steps about 20% cheaper with
identical outputs (9.4). Resuming on CUDA no longer moves RNG states to the GPU (13).

Changes in 0.19: `scripts/regime_report.py --at-step N|common` grades unfinished or stopped runs
at one shared validation round, from `metrics.jsonl` and the last checkpoint's regime state, so
the comparison can stop early at equal env steps (8.3).

Changes in 0.18 (stage design and training regimes, chosen by the user on 2026-10-22): the stage
design `track2_v1` over the constraint cells, five stages that each add one constraint, one
held-out cell, final stage shares (8.3); the gradient target structure (8.1); `MixtureSource`
and the three regimes `sequential` / `expanding` / `mixed` with a promotion rule on validation
sets (8.3); validation sets `val_s1_v1`..`val_s5_v1` and `heldout_combo_v1` (8.4);
`train.source=regime` and the `stages` / `regime` config groups (13); the regime comparison and
its decision rule, `scripts/regime_report.py` (8.3). M8 (b) moves to `val_s1_v1` (16).

Changes in 0.17 (documentation only): grading references for the benchmark tiers (8.4) from the
Track 2 experiments E1–E3: constraint cells and the precise frontier, what each reference proves,
measured interval widths and bound tightness, the options to strengthen grading, and comparison
with k-input mixes. Tier 5's prerequisites are marked met.

Changes in 0.16 (documentation only): the Track 2 input ranges fixed by the user on 2026-10-20
are recorded in 1.3 (training ranges and probe values, storage meaning, target volume, fixed
64 / 8 / 1 costs, two-input mixes); 1.7 lists only the decisions still open; 1.6 drops the
mixture-fallback limit (resolved in 0.12).


Changes in 0.15 (capability ceilings removed before the stage design; no stage, regime or
benchmark chosen): the split tree picks a recipe that fits the chip's storage (an exact peak),
adds a **blend** of two built droplets, searches under a deterministic budget, and builds a
tolerant target at the coarsest reachable composition within its tolerance (10.4). The
generator draws accuracy per target from a range (only M-smooth D for exact targets), support
size from a range, tolerance from a list, droplet volume up to each instance's M, and sets
`max_denominator` per problem; it can hold out ratios as well as problems, and the depth bound
of a tolerant target uses the coarsest composition within it (8.1). `GeneratorSource` and
`train.source=generator` train PPO from a generator (13). The truncation penalty's cache is
keyed by target content, so it no longer grows per instance (10.3). Rollouts store each env's
observation trimmed and pad per minibatch (11.4). Env padding defaults cover the 1.3 ranges
(9.2). `configs/costs/geometric.yaml`: the fixed 64 / 8 / 1 profile chosen for the MEDA stages
(6.3); PPO's `reward_scale` keeps its returns near unit scale (11.4). `tree` gets a plot colour
(a producer since 0.12, it crashed `evaluate_cli`'s plots).


Changes in 0.14: droplet volume per target ranges up to M, not M/2 (1.3; M/2 had no physical
reason, only the split tree's limit). The split tree gains a **dilute-up** step, one mix of a
built droplet with a pure fluid, so it builds droplets of any volume up to M (10.4); plans it
already built are unchanged. The generator witness and the truncation penalty get this for free.


Changes in 0.13: the step limit is sized by default from the problem alone (`depth` rule,
8.1): 4 × the sum over deliveries of (mix-depth bound + 1), optionally with a multiple drawn
per instance; the five existing stages keep the version-1 `witness` rule explicitly.

Changes in 0.12 (critical path for mixtures): the split-tree construction (10.4) builds every
reachable target over any number of fluids and any M; the truncation penalty takes the cheaper
of naive and tree per target (`ConstructiveCostToFinish`, renamed from `NaiveCostToFinish`;
10.3); the generator witness falls back to the tree, so mixtures can be drawn at a chosen
accuracy, and `witness_source` names the construction (8.1); exact search seeds its
branch and bound with the cheaper construction (10.2); `tree` is an `evaluate_cli` producer.

Changes in 0.11 (M8, mechanical part of the 0.10 redesign): observation v2 and network v2 are
built and v1 is removed (9.2, 9.3, 11.4). The env config keys are now `F_pad`, `T_pad`,
`N_pad`, `L_pad`, `volume_ref`, `count_ref` (9.2); `reset` refuses an instance whose truncation
penalty cannot be computed (9.1). Network v2 as built: relation biases are linear maps of the
pair facts with no constant term, one set per layer; the Transformer layer is written out over
`scaled_dot_product_attention`, because `nn.TransformerEncoderLayer` handles a per-head float
mask differently in eval mode (11.4). Generators may draw M, storage capacity and droplet
volume per instance (8.1); `problem_hash` ignores the chip's name (8.4). `evaluate` writes the
input-axis columns of 1.3 (12.2). M8 acceptance (a), (c) and (d) met on v2 (16). Stage design,
regime and benchmarks are unchanged and still wait for the user (8.3).

Changes in 0.10 (draft, under review): section 1 rewritten around the agent goal: one policy
for any problem and chip in a stated input space (1.1), the problem class (1.2), which
parameters are inputs with suggested ranges (1.3), the design rules that follow and where the
current observation breaks them (1.4), the claim and how it is measured (1.5), known limits
(1.6) and open decisions (1.7). The earlier goals are kept as infrastructure goals (1.8).
Then, specified but not built: observation v2 (9.2, 9.3: fluid tokens, composition fraction
matrices, droplet-target and candidate-target pair matrices, volumes on a relative and an
absolute scale, M and storage capacity in `globals`, typed operands in `cand_index`; v1 kept as
9.2a until replaced) and network v2 (11.4: usage-informed fluid embeddings, compositions as
fraction-weighted sums, relation-biased attention, no parameter shaped by any count). Rule 2.9
covers fluid order; 8.3 marked for redesign against 1.3; M8 gains acceptance (d) (one network
across F, T, M and storage) and an ordered remainder, with the general mixture fallback on the
critical path (16); v2 tests (17.2, 17.5); two supervisor questions (15).

Changes in 0.9 (M8, in progress): env `snapshot` / `restore` / `observe` for exact resume, and
the `SAME_STEP` autoreset requirement on vector envs (9.1). The parts of M8 that do not depend on
the stage design are built: network, masked policy, rollout and GAE, trainer, checkpoints,
logging (11.4); `meda.train agent=ppo` on frozen instances (13); acceptance (a) and (c) met (16).

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

*2026-10-19, updated 2026-10-21. The ranges of 1.3 were fixed by the user on 2026-10-20 from
the literature survey (PLATFORM_PARAMS.md), taking the widest range the literature supports,
since generalisation is the argument. The observation and network meet 1.4 (see the status
note there); the stage design and its validation sets cover 1.3 since 0.18 (8.3, 8.4).*

### 1.1 The goal
**One trained policy that takes a sample-preparation problem and the chip's limits as input
and returns a valid, low-cost mixing DAG within milliseconds, for any problem in the stated
input space, without retraining.** No property of the agent may tie it to a particular chip or
problem size, such as "works for M = 4" or "at most 8 fluids". Within the trained ranges it
should be near the best known plan; beyond them its behaviour is measured and reported.

### 1.2 Problem class
In the field's vocabulary (BASELINES.md 1.3): **MTMD mixture preparation on MEDA with a storage
limit**, with the chip's limits as inputs.
- **Multi-target (MT):** the targets of all concurrent experiments are merged into one target
  set and prepared with one mixing DAG that may share intermediates across experiments.
- **Multi-demand (MD):** a target may need several droplets of the same composition.
- **Mixture:** a target is a ratio over N ≥ 2 fluids (dilution is the case N = 2 with a buffer).
- **MEDA:** k:l mixing with k + l ≤ M, and partial takes of a droplet's volume.
- **Storage limit:** the droplets waiting on chip are bounded.

This conjunction is the gap no published method covers (BASELINES.md 1.2): no MEDA method takes a
demand vector, almost none has a storage limit, and none has all five.

### 1.3 Inputs the agent must take
**Inputs** vary per problem, are visible in the observation, are varied in training, and get a
held-out test axis. **Platform config** is fixed per chip and asked of the supervisor (section 15).
The design is for MEDA k:l mixing; M = 2 (DMFB, 1:1) is only a correctness check, and no range
or default is chosen with 1:1 mixing in mind.

| Parameter | Kind | Training range | Probed beyond training |
|---|---|---|---|
| Max mix volume M | input | 3–10 | 2 (sanity check), 12, 16 |
| Number of fluids F, and each fluid's kind (sample / reagent / buffer) | input | 2–6 (baseline comparisons at F ≤ 5) | 7, 8, 10 |
| Exact targets (ε = 0): accuracy D | input | D M-smooth (every prime factor ≤ M), up to 256 | 512 |
| Tolerant targets: any ratio within ε | input | ε ∈ {0.001, 0.005} (FacDA's setting), trained alongside exact targets | smaller ε |
| Target ratio vectors (support up to F fluids) | input | drawn by a weighted mix of samplers (8.1): exact grid, ε-tolerant, structured experiments | ratios never trained on (ratio-level holdout) |
| Number of targets T (all experiments merged) | input | 1–10 | 20 |
| Demand per target | input | 1–4 | 6 |
| Droplet volume per target | input | 1 to M (above M/2 one mix serves one delivery) | — |
| Storage capacity S | input | 2–16 and unbounded, with many draws where it binds | tighter or unseen values |
| Intermediate cap `max_denominator` | per problem (8.1) | set per problem, not a chip property | — |
| Min mix volume, min droplet volume, partial take allowed | platform config | min mix 2, min droplet 1, partial takes allowed | — |
| Cost weights (sample, reagent, buffer, mix, waste) | platform config, fixed | 64 / 8 / 1, mix 1, waste 1 (`costs/geometric.yaml`, 6.3) | — |

Notes on the table:
- **Storage** counts every non-reservoir droplet on chip, the mix result included. The
  literature's "k storage units" counts only parking cells; their k is roughly our k + 2, and
  comparisons state that mapping.
- **Target volume** is demand × droplet volume (each ≤ M), with the split fixed by the
  generator. A total-volume target that the agent splits itself (TVODA's setting) is a later
  extension that needs a core change.
- **Cost weights** stay fixed; weights as an input is later multi-objective work (6.5).
- **Mixes** take two inputs for the agent; more inputs are for baseline reproduction only.

Out of scope for this agent (later pipeline stages, section 14): routing, droplet positions,
contamination and washing, completion time, detection, split error (phase 2), mixes with more
than two inputs.

### 1.4 Design rules that follow
1. **Every count is a set, never a column.** Fluids, targets, droplets and candidate actions
   are rows (tokens); relations between them are pairwise features. One set of weights reads
   any F, T and number of droplets. Padding sizes are memory settings, never part of the
   network's weights.
2. **Every input of 1.3 the strategy depends on is explicit in the observation:** M, storage
   capacity, droplet volume, tolerance.
3. **Features mean the same thing at every size and on every chip:** relative quantities
   (nearest-target distance, fractions, counts satisfied) instead of indexed ones, and a volume
   scale that does not redefine "one unit" per chip (which scaling transfers best is to be
   measured).
4. **Size-free aggregation:** attention and mean/max pooling, LayerNorm; no sums over a count.
5. **The environment stays the referee** (rule 2.1): every plan is legal by construction on any
   input, so a weak decision costs more but never breaks a rule.

*Status (0.11):* observation v2 (9.2, 9.3) and network v2 (11.4) follow rules 1–4 and are
built; acceptance (d) of M8 tests them (16). The stage design draws every input of 1.3 (8.3).

### 1.5 The claim and how it is measured
> One policy, no retraining: valid plans on any input; near the best known cost across the
> trained ranges; measured behaviour beyond them.

- Reported as **curves of gap against each axis of 1.3** (M, F, T, demand, storage, accuracy),
  inside and beyond the trained range, with success rate and inference time, not only as one
  table per benchmark set.
- Gaps follow section 12.2: to a proved optimum where exact search finishes, otherwise to the
  interval [lower bound, best known].
- On each published method's own input class, the agent is compared with that method (tiers 2–3
  of 8.4). The full conjunction of 1.2 is the frontier (tier 5), where no published method
  applies directly.

### 1.6 Known practical limits (measured, not assumed away)
- **Training coverage:** reliability is earned only where training varied the inputs.
- **Action space growth:** candidates grow with M and with droplets on chip (pairs × volume
  splits), so `L_pad`, memory and decision difficulty grow at large M.
- **Grading weakens with size:** exact search proves optima only on small instances, and the
  [lower bound, best known] intervals widen on larger ones.
- **Exact search proves optima only on small problems:** measured in `notes/track2` (Track 2
  experiment E2), larger benchmarks are graded by [lower bound, best known] intervals.
- **Exact-integer bounds in the encoder:** denominators are kept below 2^20 (9.3); much finer
  accuracy would need that revisited.

### 1.7 Open decisions
Settled on 2026-10-20 (1.3): storage in droplets, exact and tolerant targets both trained,
`max_denominator` a per-problem bound, partial takes allowed, costs fixed at 64 / 8 / 1.
Still open:
1. The target platform's realistic M, mixer footprint and reservoir count, whether the router
   can serve as a cost oracle, and real assay cost weights (supervisor). The architecture is not
   limited by the answer; the headline results are centred on it.
2. Training regime: settled by the comparison experiment of 8.3 (stages, held-out cell and
   validation sets were chosen on 2026-10-22).
3. Whether limited storage (2–16, uniform) binds often enough: the naive and tree plans
   exceed the drawn capacity in 1 of 180 draws (8.3), so only plans that share would feel it.
   Measure on trained agents; a range skewed toward small capacities is the fallback.
4. Optional extensions: total-volume targets, non-pure stock fluids, a hard budget on sample
   volume, cost weights as an input (preference conditioning, 6.5).

### 1.8 Infrastructure goals
These carry the agent goal and are built (M0–M7):
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
9. **Permutation invariance.** Fluid order, droplet order and target order carry no meaning.
   Canonical ordering is used only for hashing and for stable candidate indexing. No network
   weight may belong to a particular fluid, droplet or target position (1.4).

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
│   ├── stages/                # stage designs over constraint cells (8.3)
│   ├── regime/                # training regimes: sequential, expanding, mixed (8.3)
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
│   │   ├── stages.py          # stage design over cells; MixtureSource
│   │   ├── regime.py          # training regimes: stage shares and promotion
│   │   ├── curriculum.py      # v1 ladder's staged difficulty schedule
│   │   └── benchmark_io.py    # read/write versioned benchmark files
│   ├── solvers/
│   │   ├── base.py            # Solver protocol
│   │   ├── exact_search.py    # uniform-cost / A* over canonical states
│   │   ├── fallback.py        # cheap cost-to-finish estimate (truncation penalty)
│   │   └── baselines/
│   │       ├── naive.py       # each target built independently
│   │       ├── tree.py        # split tree: every reachable target, any number of fluids
│   │       ├── share.py       # sharing pass: the constructions plus reuse across targets
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
│   │   │   ├── collector.py   # workers collector: one env and a policy copy per process
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
│   ├── make_benchmark.py      # build and freeze a benchmark set
│   └── regime_report.py       # compare regime runs, apply the decision rule (8.3)
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
  = `failure_penalty_factor × cost_to_finish(state, instance)` (section 10.3;
  `failure_penalty_factor` and the `cost_to_finish` kind in `configs/env`; factor an integer ≥ 1,
  default 2; kind `share`, default, or `construction`). **Since 0.25 (`share`)** the cost to
  finish is the cheaper of a completion from the current state that reuses the droplets on chip
  and the rebuild, so a half-built intermediate lowers the charge. The guarantee is then per
  state: being cut off costs the factor times a completion that really exists from there, so
  finishing yourself pays whenever your own finish is under factor × the planner's
  (`tests/envs/test_failure_penalty.py`). A whole failed episode can now cost less than the best
  known plan, when its moves were good (E-19: 247 against 475 on val_s1-008). **Under
  `construction`** (0.22–0.24) the rest of this item holds. With factor 1 an
  unfinished plan is never cheaper than finishing *by the construction*, but it can be cheaper
  than finishing with a worse plan: on the stage validation sets a deliberately failed episode
  cost only 1.03–1.33× the best known plan, and the 2026-10-22 pilot learned to finish only
  problems where it beat the construction. With factor 2, failing costs at least 1.8× the best
  known plan on every stage-1–3 validation problem (`tests/envs/test_failure_penalty.py`), so a
  finished plan is preferred unless it is roughly twice the construction's cost.
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
The MEDA stages use `configs/costs/geometric.yaml` (chosen 2026-10-20): sample 64, reagent 8,
buffer 1, mix 1, waste 1, e-MRCM's geometric scheme with the buffer kept above 0 (a free
fluid gives degenerate plans). `default.yaml` stays, because the stored optima of the existing
frozen sets use it.

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
- `target_structure` (0.18): one name or a list drawn from per instance (repeat to weight):
  `independent` (the default: each target from the composition sampler) or `gradient`, a
  linear dilution gradient: one exact denominator D the instance's M can reach (with at least
  T distinct shares), sample shares evenly spaced inside (0, D). Only for plain dilutions
  (`dirichlet_snapped`, support 2, one sample and one buffer). Recorded as
  `metadata["target_structure"]`; enters the config hash only when not `independent`.
- `support_size`: how many fluids are nonzero in a target (dilution = 2: one sample + buffer);
  a fixed value or a `[low, high]` range drawn per target (capped at the instance's F).
  `reachable_walk` takes a fixed value.
- `accuracy_denominator`: a fixed D, or a `[low, high]` range drawn per target (0.15). From a
  range, an exact target (tolerance 0) draws only D whose prime factors are at most the
  instance's M, so every M gets non-dyadic grids it can reach (12 = 2²·3 at M >= 3, 120 at
  M >= 5); a tolerant target may draw any D.
- `tolerance`: a value, or a list drawn from per target (e.g. `[0, "1/200", "1/1000"]`). A
  tolerant target need not be reachable; the witness builds the coarsest reachable composition
  within it (10.4).
- `max_denominator` (optional): the bound on any droplet's denominator for this problem (1.7:
  a property of the problem, not the chip), replacing the chip config's value. A tolerance of
  1/1000 needs grids finer than 256.
- `droplet_volume`: a fixed value or a `[low, high]` range, drawn per target; `[low, M]` ends at
  each instance's own M (1.3: 1 to M, with M drawn per instance). A draw above the
  instance's M raises (no droplet could be delivered); `tolerance`.
- `max_mix_volume`, `storage_capacity` (optional, `[low, high]`): chip inputs drawn per instance
  (1.3). The instance's chip is the configured chip with these two fields replaced and named
  `<chip>-M<m>-S<s>`; the platform constants (min mix and droplet volume, partial take,
  `max_denominator`) stay as configured. Unset, the chip's own values are used.
  A fixed value draws no randomness, and these keys enter the config hash only when set, so
  every config that does not use them samples and hashes exactly as before. An accuracy
  denominator must suit every M in the range (8.3); a draw the naive witness cannot build raises
  naming the instance.
- `step_limit`: an absolute value, or `null` to size it by `step_limit_rule`:
  - `depth` (the default): `ceil(step_limit_multiple × base)`, where base is, summed over every
    delivery, its target's **mix-depth bound** plus one output. The bound is the fewest mix
    levels any plan needs, `ceil(log_M D)` for reduced denominator D (a mix of at most M units
    multiplies a denominator by at most M), so the budget comes from the problem alone and not
    from how good a construction is. Default multiple 4: measured, the witness is at most 2.7 ×
    base (median 1.6–1.9) across the five stages and ranged and mixture settings, so 4 gives
    about the old budget. A limit below the witness length is raised to it and flagged
    (`step_limit_raised_to_witness`); `step_limit_base` and the multiple used are recorded too.
  - `depth_support` (`track2_v1`, 0.22): as `depth`, but each delivery counts at least
    (required fluids − 1) mixes, a fluid being required when its share exceeds the tolerance
    (each mix joins two droplets), and the limit is floored at `WITNESS_FLOOR` (2) × the witness
    length, flagged as above. `depth` under-budgets mixtures: deep mixture witnesses are far
    longer than any problem-only bound, and 13 of 40 stage-2 validation problems allowed exactly
    the witness plan (10 with the fluid term alone). With the floor, every problem allows at
    least twice its known plan.
  - `witness`: `step_limit_multiple` (default 2) times the witness plan length, the rule of
    generator version 1. The five existing stages set it explicitly, so their draws, hashes and
    frozen sets are unchanged.

  `step_limit_multiple` may be a `[low, high]` range, drawn per instance (a randomised budget,
  so the agent sees tight and loose budgets; it already observes steps left). A fixed value
  draws no randomness. A step limit below the witness length is an error.

**Every sampled instance is solvable by construction.** The sampler builds a *feasibility
witness* — a legal plan meeting all demand — before returning, and raises an error naming the
instance if it cannot. The witness comes from the walk's own recipes (`reachable_walk`, stored
in `metadata["recipes"]` so it stays re-derivable from the frozen instance), else from the
naive construction, else from the split tree (10.4); `metadata["witness_source"]` names which.
Naive goes first, so every instance it already witnessed is unchanged. `metadata` records the generator name and version, the config hash, the
composition sampler, `witness_actions`, `witness_source` and, when `weights` is given,
`known_upper_bound` (the witness cost as an exact `"a/b"` string).

**Tolerant targets in the depth rule.** A target's mix-depth bound uses the smallest
denominator of any composition within its tolerance (a plan may deliver that one), so a
tolerant target's budget is not sized by the fine grid it was written on.

**Training from a generator.** `GeneratorSource(config, chip, weights, exclude,
exclude_compositions)` is an env instance source that draws a fresh problem every episode
(picklable, for `AsyncVectorEnv`). A draw no construction can witness raises
`UnwitnessedError` from `sample_instance`; the source redraws it instead of ending a training
run, records the count in `metadata["unwitnessed_redraws"]` (measured: 2 in 150 draws over the
full 1.3 ranges, all at storage 2), and raises after `max_unwitnessed` (100) in a row.
`exclude_compositions` (also a `sample_instance` argument)
holds out target **ratios**: no target of a draw has one, which is what an unseen-ratio test set
needs (whole-problem hashes cannot express it).

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

### 8.3 Stages and training regime (`stages.py`, `regime.py`; `curriculum.py` for the v1 ladder)

*Chosen by the user on 2026-10-22 from the Track 2 constraint cells (8.4, Grading references).*

**Stage design** (`configs/stages/track2_v1.yaml`, `StageDesign.from_dict`). A **cell** fixes
one value on each axis: fluids (`dilution` / `mixture`) × targets (`one` / `many`) × demand
(`single` / `multi`) × storage (`loose` / `limited`). Its generator config (8.1) is the design's
`base` updated by each axis value's keys, then by every matching `overrides` entry; the cell's
name (`dilution-one-single-loose`) is the config name. Stage 1 is the `start` cell; each later
stage **adds one axis value**, and its cells are every combination of the values unlocked so
far that no earlier stage holds, minus the **held-out** cells, which are never drawn in
training. Ranges are those of 1.3.

| Stage | Adds | Cells (count) | Track 2 E2 headroom |
|---|---|---|---|
| `s1_dilution` | (start) | dilution · one · single · loose (1) | almost none: a warm-up |
| `s2_mixture` | fluids: mixture (F 3–6, support 3–6) | mixture · one · single · loose (1) | small |
| `s3_demand` | demand: 2–4 | {dilution, mixture} · one · multi · loose (2) | moderate |
| `s4_targets` | targets: 2–10 | {dil, mix} · many · {single, multi} · loose (4) | large (sharing) |
| `s5_storage` | storage: 2–16 | every cell above with limited storage (7) | large |

Every cell: one experiment, `dirichlet_snapped` targets, accuracy D from 4–256 (only D whose
prime factors are ≤ the instance's M for exact targets), tolerance drawn from
`[0, 0, 1/200, 1/1000]` per target (half exact), `max_denominator` 1024 (a 1/1000 tolerance
needs grids finer than 256), droplet volume 1 to M, M 3–10, depth step-limit rule. Loose
storage is 32 (env `N_pad`, standing in for unbounded). Dilution cells with many targets draw a
linear gradient for half their instances (`target_structure`, 8.1).

**Held out:** dilution · many · multi · limited, tested on its own tier-4 set
(`heldout_combo_v2`; `_v1` before 0.21): can the agent combine skills it learned in separate
cells? The headline cell (mixture · many · multi · limited) is deliberately *not* held out, so
the main agent trains on the problem the claim is about.

`final_weights` are the stage shares once every stage is in: 0.05, 0.05, 0.2, 0.35, 0.35 (most
draws where E2 found headroom). A stage's share is split evenly over its cells.

**Measured on 25 draws per cell:** every cell is witnessed (one unwitnessed redraw in 400, at
limited storage); the depth step limit is raised to the witness on about 30% of mixture draws
(flagged per instance, 8.1); step limits reach 600 on the largest cells. On the limited-storage
cells the naive and tree plans almost never exceed the drawn capacity (1 of 180 draws; they keep
2–4 droplets at a time), so whether a uniform 2–16 binds often enough for plans that share is
not yet known (1.7).

**Mixture source.** `MixtureSource(parts)` is the env instance source of a training run: each
episode picks a cell with probability proportional to its weight and draws a fresh problem from
that cell's `GeneratorSource` (never one of the frozen sets' hashes), recording
`metadata["cell"]`. `StageDesign.source(stage_weights, chip, weights, exclude)` builds it.
Picklable, so it works with `AsyncVectorEnv`; it draws only through the env's generator.

**Training regimes** (`Regime`, `configs/regime/*.yaml`). The regimes share the design, the
source and the validation sets, and differ only in the stage shares they ask for:

| Regime | Draws from | On promotion |
|---|---|---|
| `sequential` (E) | the newest stage only | the next stage replaces it |
| `expanding` (C) | stages 1..k: the newest gets `newest_share` (0.5), earlier stages split the rest evenly | the next stage is added; after the last stage passes, the final weights |
| `mixed` (D) | every stage at the final weights from step 0 | never promotes |

A stage **passes** at a validation round when its greedy success rate on its validation set is
at least `min_success` (0.95) and its mean `gap_to_best_known` has plateaued: the best gap of
the last `patience` (3) rounds is not `min_improvement` (0.01) below the best gap before them.
Promotion reads only validation results, never training episodes. A stage that has not passed
after `max_stage_fraction` (0.2) of the run's env steps is promoted anyway, and the event is
marked `forced`. A sequential or expanding regime with stage count × `max_stage_fraction` > 1
is refused at construction: forced promotions at every cap would bring the last stage in after
the budget (5 × 0.3 = 1.5), so it would never train. `num_stages` cuts the design to its first k stages for a pilot (final weights
renormalised). `state_dict` / `load_state_dict` carry the regime across checkpoints (13).

**Comparison experiment** (user-approved, 2026-10-22). Arms `expanding` (candidate), `mixed`
(challenger) and `sequential` (baseline); equal env-step budgets and seeds. Pilot: stages 1–3
(`regime.num_stages=3`), 2 seeds each; then the winner on all five stages with 3 seeds.
`scripts/regime_report.py` reads the run directories and applies the decision rule, fixed
before the runs:
1. Reject a regime whose mean final success over seeds is below 95% on any stage's set.
2. Among the rest, pick the lowest gap to best known, weighted by the final weights.
3. If two regimes' seed ranges of that gap overlap, prefer the simpler (mixed, then expanding,
   then sequential), then the one that forgets less (best validation success during the run
   minus final success, per stage).
"Final" is each run's end-of-run grading, or with `--at-step N` (or `common`, the latest round
every run has reached) its validation round at or before env step N: the pilot may stop once the
curves separate, and runs graded at different rounds are refused rather than compared.
A deferred arm, prioritised level replay over instances (D-PLR,
`notes/training_regime_research.md`), is tried only if C and D both stall.

**The version-1 ladder** (`curriculum.py`, `configs/generator/{tiny_dilution, dilution_single,
dilution_multi, mixture_single, mixture_multi}.yaml`) predates section 1: each stage fixes M and
D. It stays for the frozen sets built from it (`tiny_exact_v1`, `val_stage1_*`), the M5–M8
acceptance tests and `train.source=generator`. `Curriculum(stages, rng, weights, exclude)` is
its per-env promotion schedule on training windows; the regimes above replace it for training.

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
of targets and the chip limits (`problem_hash` ignores `max_denominator` and the chip's name,
below). It deliberately ignores the instance id, the metadata and the
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
| 1 In-distribution | Does the agent learn its training stage? | `val_s<k>_v1`, k = 1..5 (`track2_v1`); `val_stage1_v2` (v1 ladder) | `optimal`, else [bound, best known] | 0.18 |
| 2 Engine check (DMFB) | Does our core reproduce published 1:1 results? | `dmfb_sweep_v1` | exact search, bitscan, published DMFB numbers | M9 |
| 3 Literature | Do we reproduce, then beat, MEDA methods on their own problems? | `literature_<paper>_v1` | the paper's reported numbers + our reproduction | M9 |
| 4 Generalisation | Does the policy transfer to unseen and larger problems, and to an unseen combination of constraints? | `heldout_combo_v2` (0.21; `_v1` 0.18), `unseen_ratios_v1`, `larger_v1` | [bound, best known] | held-out cell: 0.18; others after training |
| 5 Frontier | How does the agent do where no published method applies? | `frontier_v1` | [bound, best known] (see Grading references) | with the stage design (the mixture fallback exists since 0.12) |
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
- Both prerequisites are met: env padding covers the 1.3 ranges (0.15; beyond them it raises,
  and with observation v2 padding is a memory setting, 9.2), and the cost-to-finish fallback
  supports mixtures (10.3, since 0.12).
- **Claim rule.** "No published method can attempt these" is false as written. Naive always
  produces a plan, and an exact solver can attempt any instance given unlimited time. Report
  what was measured instead: the instance lies outside every published method's input class,
  exact search did not finish within budget B, and the agent's cost is within x of the lower
  bound and y below the best adapted baseline, at inference time t.

**Grading references (Track 2 experiments E1–E3, `notes/track2/`, 2026-10-21).**

*Cells.* A problem's constraint cell is fluids (dilution / mixture) × targets (one / many) ×
demand (1 / more) × storage (unbounded / limited). Published MEDA heuristics cover only the cells
with demand 1 and unbounded storage (WSPM, FacDA, MRCM, e-MRCM, MTWM; the map is in
`notes/track2/e1_coverage.md`). The **frontier** is exactly: demand > 1 together with many targets
or a mixture, plus every cell with limited storage. In a covered cell the agent is compared with
every method that applies, on each method's own metric and on every proxy column of 12.1, not
only on the weighted cost, since each heuristic optimises a different objective.

*References, and what each proves:*
| Reference | Available | Proves |
|---|---|---|
| exact optimum (10.2) | small problems only (measured: 16% of small draws within 60 s; almost none with ≥ 4 deliveries) | the true gap |
| lower bound (10.2) | every size | `gap_to_lower_bound`: the agent is at most that far from optimal |
| naive and split tree (10.4) | every size | a floor baseline; they never share across targets, so beating them is a weak result |
| published methods | covered cells, as each is reproduced (tier 3) | "beats the state of the art" |
| adapted methods | frontier: 1:1 demand/storage planners (MTMD, DDMT, NFSP; a 1:1 plan is a legal MEDA plan) and MEDA methods run per droplet | labelled `adapted` (claim rule above) |

*Measured.* On the frontier, exact search never improves the construction within 300 s
(0 of 45 medium problems), and the interval [bound, best known] is 28–60% wide (median per cell).
Where the optimum is known, the bound is within 5% of it whenever the optimal plan wastes almost
nothing (14 of 18 multi-target problems). It is loose only when waste is forced, for example a
delivery smaller than the minimum mix volume. So on multi-target problems the width is mostly
the construction's excess, not the bound's (inferred from small problems), and
`gap_to_lower_bound` is a usable certificate at scale.

*Options to strengthen grading, none needed before training:*
1. **A sharing-aware search baseline** (beam or large-neighbourhood search over the engine,
   seeded by the split tree). It tightens the best-known end on the frontier and is the
   non-learned competitor. No new dependency.
2. **Faster exact search** (no re-check of generated legal moves, cached heuristic, dedupe of
   equivalent children). It enlarges the provable region for tier 0/1. No new dependency.
3. **A forced-waste term in the lower bound**, for deliveries a mix cannot produce without
   leftovers. It stays admissible. Cheap. (Whole-unit rounding, a waste term from volume
   conservation and a mix-count term are done in 0.21, 10.2.)
4. **An LP relaxation over fluid flows**, behind the same `lower_bound` field. It needs an LP
   solver (a new dependency, section 3).
5. **An exact ILP/SMT formulation** for small and medium problems. It needs a solver (a new
   dependency).
6. **Mixes with more than two inputs, for comparisons only.** A k-input mix equals a chain of
   k − 1 two-input mixes with identical fluid use and no extra waste (checked in the engine).
   Baseline plans with k-input mixes are expanded into chains and run through the engine.
   `dag_metrics` gains a contracted mix count, which merges a chain in which each intermediate
   is used whole, once and immediately into one mix, so mixes are reported on the papers'
   terms. The agent stays at two inputs.

*Reporting.* Under the fixed 64 / 8 / 1 weights the cost is mostly sample, so sample units
saved are reported next to cost.

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
- `val_s1_v2` … `val_s5_v2` and `heldout_combo_v2` (0.21, 0.22): the same problems and best
  known plans as the `_v1` sets below, with the 0.21 bound (10.2) as the lower end and the
  0.22 `depth_support` step limits (8.1; at least 2× the witness, 112 of 242 limits raised). Median bound /
  best known per set, v1 → v2: s1 0.38 → 0.66, s2 0.19 → 0.33, s3 0.35 → 0.43,
  s4 0.32 → 0.33, s5 0.33 → 0.39, held-out 0.50 → 0.51; 3 stage-1 instances are now proved
  optimal (bound = construction). Many-target sets barely move: rounding matters little when
  each fluid's total need is large, so their width is the construction's excess or the bound's
  missing terms (forced waste per mix, grading option 3). The design points at these sets.
- `val_s1_v1` … `val_s5_v1` (0.18) — tier 1 for the stages of `track2_v1` (8.3), and
  `heldout_combo_v1` — tier 4, the held-out cell (kept frozen; the first pilot ran on them).
  Built by
  `make_benchmark.py:build_design_set`: `val_size` (40) instances per set split evenly over its
  cells (stage 5 has 7 cells, so 42), seed `val_seed` + stage index, every draw a distinct
  problem; the sets are disjoint from each other and training excludes them all. Graded under
  `costs/geometric` (the design's `costs`; `train.source=regime` refuses other weights).
  Answer key: [lower bound, best known], best known the cheaper of the naive and split-tree
  plans, lower bound the fluid-conservation bound at the root. Exact search proved none of 24
  stage 1–3 draws within 1,000 nodes (1–100 s each), so a node budget buys nothing at these
  sizes. The root bound was loose here: the best known plan is 2.0–5.4 × the bound (median per
  set). Much of that is integer draws: a target holding 1/195 of sample still needs one whole
  unit drawn, which the 0.21 bound counts (`_v2` sets). Promotion (8.3) uses
  `gap_to_best_known`, which the bound does not affect.
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
there and in a stage-1 draw, and the chip's `name`, a label that a generator drawing M sets per
draw. `check_no_overlap` compares `problem_hash`. Hashes are computed, never stored in the
frozen files, so this changed no set.

---

## 9. Environment (`meda.envs.sample_prep_env.SamplePrepEnv`)

### 9.1 API (Gymnasium)
```python
env = SamplePrepEnv(env_config, instance_source, cost_weights, cost_to_finish, hints=None)
# instance_source(rng) -> Instance: sampler, curriculum stage, or FixedInstances(list)
# cost_to_finish(state, instance) -> Rational: truncation penalty (solvers.fallback, M5)
# hints: tables(instance, target_id) for the candidate hint columns (solvers.hints, 0.25)
obs, info = env.reset(seed=None, options=None) # options may pass a specific Instance
obs, reward, terminated, truncated, info = env.step(action_index)
mask = env.action_masks()                      # bool[L_pad]; same as obs["action_mask"]
```
- `action_space = Discrete(L_pad)`. The action is an **index into the current candidate list**,
  which is `rules.legal_actions(state)` in canonical order, padded to `L_pad`.
- If the number of legal actions exceeds `L_pad`, raise a clear error naming the instance (do
  not truncate silently). It is a config value and a memory setting only (9.2).
- Selecting an index that is not a legal candidate raises `IllegalActionError`.
- `reset` raises (naming the instance) if the instance does not fit the padding sizes, if a
  target's `droplet_volume` exceeds `max_mix_volume`, if no action is legal at the start, or if
  `cost_to_finish` exposes `supports(instance)` and it is false (a mixture the naive fallback
  cannot price, 10.3): an unpriceable instance is refused up front, not when it truncates.
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

### 9.2 Observation v2 (`gymnasium.spaces.Dict`, padded, float32 unless noted)

*Status: built (0.11, `envs/encoding.py`); v1 is removed. v2 follows the design rules of 1.4:
every count is a set of rows, never a feature column.*

The observation is a set of **tokens** (fluids, droplets, targets, candidates) plus **pair
matrices** that relate them. No feature column belongs to a particular fluid or target, so
no network weight does either: the same weights read an instance with any number of fluids,
targets, droplets or candidates, on a chip with any M or storage capacity.

| Key | Shape | Content |
| --- | --- | --- |
| `fluids` | [F_pad, D_fluid] | One row per fluid (= its reservoir), by fluid id |
| `fluid_mask` | [F_pad] int8 | Valid rows |
| `droplets` | [N_pad, D_drop] | One row per droplet on chip, in the state's canonical order |
| `droplet_mask` | [N_pad] int8 | Valid rows |
| `droplet_comp` | [N_pad, F_pad] | Composition of each droplet as fractions over the fluid rows |
| `targets` | [T_pad, D_tgt] | One row per target |
| `target_mask` | [T_pad] int8 | Valid rows |
| `target_comp` | [T_pad, F_pad] | Composition of each target as fractions over the fluid rows |
| `droplet_target` | [N_pad, T_pad, 2] | Per (droplet, target): L∞ distance, satisfies flag |
| `candidates` | [L_pad, D_cand] | Scalar features of each legal action (below) |
| `cand_comp` | [L_pad, F_pad] | Composition each candidate produces (its afterstate) |
| `cand_target` | [L_pad, T_pad, 2] | Per (candidate, target): L∞ distance of the result, satisfies flag |
| `cand_index` | [L_pad, 5] int32 | (op, a_type, a_row, b_type, b_row); op 0 = Mix, 1 = Output, 2 = Discard; type 0 = fluid, 1 = droplet, 2 = target; −1 if unused |
| `action_mask` | [L_pad] int8 | Valid candidates (`env.action_masks()` returns the same as bool) |
| `globals` | [D_glob] | Episode and chip quantities (below) |

**Padding sizes are memory settings, not limits of the agent.** `F_pad`, `T_pad`, `N_pad` and
`L_pad` (`configs/env`) size the arrays only. Raising them never invalidates trained weights;
an instance larger than the padding raises an error naming it (never truncated, section 18),
and the fix is a larger padding, not retraining. Config keys (`configs/env`): `F_pad`, `T_pad`,
`N_pad`, `L_pad`, `volume_ref`, `count_ref` (v1 used `F_max`, `T_max`,
`storage_capacity_max`, `L_max`). Batches are trimmed to their largest real row counts before
the network sees them (11.4).

**Scales.** Every volume appears on two scales, because both meanings matter and neither may
be redefined per chip:
- **relative**, `volume / M`: how much of a mixer it fills (what decides which mixes exist);
- **absolute**, `volume / volume_ref`: its size in units, where `volume_ref` is a fixed
  normalising constant in `configs/env` (default 16), the same on every chip.

Counts that matter in absolute terms (remaining demand, steps left, storage capacity) are divided
by `count_ref` (default 16). These normalisers are scales, not caps: values above 1 are legal,
and the observation space bounds them by [0, ∞). Fractions, flags and distances lie in [0, 1].

**Pair conventions.** For a met target, distance is 1 and satisfies is 0 (it is no longer
wanted); the target row's `met` flag says why. Padded pairs are 0 and masked.

### 9.3 Features (`encoding.py`) — the network never does arithmetic

Exactness is unchanged from v1: every result composition and distance is formed as exact
integers (numpy int64) and divided only at the end, so each float is the correctly rounded exact
value; "satisfies" flags are compared exactly in integers. To keep int64 exact,
`max_denominator` and every target composition/tolerance denominator must be ≤ 2^20 (checked at
reset), and tolerances ≤ 1. The denominator feature is `1 − 1/den` (0 for pure fluids).

Column order (as built) is the order of each list below.

**Fluid row (D_fluid = 3):** kind one-hot (sample, reagent, buffer). Kind is a category with a
fixed number of values, not a count. Reserved extensions, each a new column only when its
decision is taken (1.7): cost weight of the fluid (preference conditioning, 6.5), stock
concentration (non-pure stocks).

**Droplet row (D_drop = 6):** volume / M; volume / `volume_ref`; denominator feature; age
(actions since the droplet was made / step limit); distance to the nearest open target;
satisfies-any-open-target flag. The per-target detail is in `droplet_target`.

**Target row (D_tgt = 7):** droplet volume / M; droplet volume / `volume_ref`; tolerance;
remaining demand / demand; remaining demand / `count_ref`; met flag; denominator feature of the
target composition. The composition itself is in `target_comp`.

**Candidate row (D_cand = 27), the afterstate features:** op one-hot (3); k, l, k + l on both
scales (6: k/M, l/M, (k + l)/M, then the same over `volume_ref`; for Output k is the delivered
volume, for Discard the droplet volume, l = 0); the
result's denominator feature; distance from the result to the nearest open target; whether the
result satisfies any open target; this action's reagent units by fluid kind / `volume_ref` (3);
waste it creates / `volume_ref`; droplets on chip after the action / storage capacity. The
result composition is in `cand_comp` and its per-target detail in `cand_target`. Providing the
result of each candidate is the afterstate idea (TD-Gammon): the network judges outcomes instead
of predicting them.

**Hint columns (since 0.25; 10 of the 27).** For a Mix candidate, with the result's exact
reduced composition and volume, the fewest mixes to any open target: (a) **recipe**: along the
recipe for one droplet of the target by the split tree (for `tree.aim(target)`) or by the
sharing pass, one-hot 0, 1, 2, 3, ≥ 4, none; (b) **chain**: along a chain of mixes each adding
one pure fluid (k + l ≤ M), worked back from the aim with enough volume, one-hot 0, 1, 2, none
(depth 2). Output and Discard rows are 0. Why: the network cannot infer from compositions as
floats which intermediate leads to a target (33/40 S at M = 9 is one mix from 1/8 S, far by
distance); on stage 1, a crude version lifted greedy copy success from 25% to 67.5% (E-16).
They are information, not instructions: legality and costs are unchanged. Tables come from
`solvers.hints.CandidateHints(weights)`, injected into the env (`SamplePrepEnv(..., hints)`),
cached per (target, chip, fluid kinds); without it the columns are 0. With the env config's
`hint_source: problem` (0.28, an ablation), the recipe columns instead carry the fewest mixes
along a pure-fluid chain of depth ≤ 4 (one-hot 0..3, ≥ 4, none) and no planner is called. Cost: about 1–2 ms per
step; 5–17 ms per reset median on stages 1, 2, 4 (largest 0.6 s, stage 4).

**Globals (D_glob = 7):** step / step limit; steps left / `count_ref`; droplets on chip /
storage capacity; remaining demand / total demand; remaining demand / `count_ref`;
M / `volume_ref`; storage capacity / `count_ref`. The v1 globals F / F_max and T / T_max are
gone: they tie features to padding sizes, and the masks already say how many rows are real.
Preference weights are appended here when 6.5 is enabled.

**Operands.** `cand_index` names each operand by type and row: a Mix's operands are fluids
(reservoir takes) or droplets; an Output's are a fluid or droplet and the target; a Discard's is
a droplet. Rows index the corresponding token array, so no index depends on another array's
padding.

### 9.2a Observation v1 (removed in 0.11)
Keys `sources` (reservoirs, then droplets), `targets`, `candidates`, `cand_index` [L, 3] and
`globals` [5], with compositions as `F_max` columns and one distance column per target
(`T_max`), so F and T were hard limits of a trained network and volumes were divided by M only.
The M5–M8 acceptance runs before 0.11 used it; nothing trained on it was kept.

### 9.4 Vectorisation and speed
- Must work with `gymnasium.vector.AsyncVectorEnv` (picklable, no global state).
- `scripts/profile_env.py` reports steps/second for each generator config.
- Target: at least 10,000 env steps/second in total with 16–32 async workers on the
  workstation for curriculum stages 1–3. Optimise only after profiling (cache legal-action
  enumeration per canonical state; `numba` only if needed).
- Observation v2 adds pair matrices: `cand_target` is L_pad × T_pad × 2 floats per step (for
  example 2048 × 12 × 2, about 200 KB). The encoder computes them vectorised from the same exact
  integers as the candidate features; profile v2 against v1 before and after the switch.
- **Measured 2026-10-22 (0.18), stages 1–3 of `track2_v1`, random policy.** One env steps at
  about 190–230 steps/s (mean 4.3 ms, p90 8.7, p99 21: legal enumeration and encoding). At the
  default pads one observation is 4.7 MB, almost all candidate padding. The first workstation
  run (32 async workers, A5000) reached only about 280 steps/s with the cores mostly idle: the
  main process deep-copied every padded batch (about 150 MB a step) and the rollout kept views
  into those batches (21 GB). Training vector envs are now built with `copy=False` and the
  trainer keeps one trimmed copy per env before the next step (11.4); without the network,
  7 workers went from 129 to 313 steps/s on a laptop. Workers step in lockstep, so a vector
  step waits for its slowest env; that tail bounds 32 workers at roughly 1,500–2,000 steps/s
  under a random policy, short of the 10,000 target. Fewer droplets on chip (a trained policy)
  make steps cheaper.
- **Measured 2026-10-22 (0.20), same conditions.** Legal enumeration is about 65% of an env
  step and encoding about 30%. Two changes with bit-identical outputs (a hash of every legal
  list, observation and reward over 16,000 random steps across all five stages is unchanged)
  cut a step by about 20%: `legal_actions` skips the gcd when the unreduced mix denominator is
  already within `max_denominator` (the reduced one divides it), and the encoder builds the
  candidate columns from one list instead of per-element array writes. Lockstep efficiency
  (mean step over the slowest of 32) is about 21%: a step averages about 5 ms but its tail
  reaches 25–80 ms on a full chip. The workers collector (11.4) removes the per-step wait.

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
    def reuse(self, composition, need) -> Source | None: ...  # None; the sharing pass overrides
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

`heuristic` selects A*: `fluid_bound` (default) or `zero` (plain Dijkstra). The bound (0.21) is
a sum of three conservation terms, each a lower bound on its own part of the remaining cost:
1. **Fluid draws.** A droplet delivered to target t holds at least
   `droplet_volume * max(0, composition[f] - tolerance)` units of fluid f (L-inf tolerance), and
   fluid f comes only from what is on chip or from reservoir draws, which are whole units. So
   future draws of f are at least `D[f] = max(0, ceil(needed[f] - on_chip[f]))`, weighted by
   the fluid's cost.
2. **Waste.** Every unit on chip or drawn later is delivered or wasted (leftovers go to waste
   at the end): waste ≥ `on_chip_volume + sum(D) - volume still to deliver`.
3. **Mixes.** A unit delivered to a target no pure reservoir fluid satisfies comes from a droplet
   on chip now or from a future mix product, which holds at most M units: mixes ≥
   `ceil((that volume - on_chip_volume) / M)`.

The bound is consistent (each action lowers it by at most its own cost; checked along random
rollouts on every `track2_v1` cell and in the tests), and admissible (never above an optimum
proved by plain Dijkstra on small problems, nor above any construction; the wide check is
`scripts/check_bound.py`, 300 random small problems, run on a many-core machine). Before 0.21 it used
the exact target shares with fractional draws and no waste or mix terms: on tolerant targets it
could exceed the optimum (51% ± 2% at volume 2: 66.26 against 66). Re-opening is allowed (a pop
whose `g` is stale is skipped), so optimality would hold even for an inconsistent heuristic.

**Branch and bound.** The search starts from an incumbent plan (the cheaper of the naive and
split-tree constructions, 10.3) and prunes
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

### 10.3 Fallback cost-to-finish (`fallback.py`; `share.py` since 0.25)
**`share` (default since 0.25).** `SharingCostToFinish(weights)(state, instance)` = the cheaper
of `share.finish_from(state, instance, weights)` and the rebuild below. `finish_from` runs the
sharing pass (10.4) from `state` with every droplet on chip in its pool, over its target orders,
on a relaxed step limit, and returns the completion and its exact cost (final leftovers not
charged, as for a finished plan). Both are completions that really exist, so the minimum is one
too. Measured from states along the teacher's plans (E-18): always found; 4–9 ms median on
stages 1–3 and 5, 1.8 s on stage 4. Called once per failed episode, never per candidate.

**`construction`.**
`ConstructiveCostToFinish(weights)(state, instance)`: waste everything on chip, then build each
remaining target's droplets from the reservoirs, per target with the cheaper of the naive and
split-tree constructions (10.4; `cheapest_construction`). Because it is the cost of a completion
that really exists, it can never be below the optimal remaining cost, which is what the
truncation penalty needs — otherwise an unfinished plan could look cheaper than a finished one.
Taking the cheaper construction keeps it as close to the true cost of finishing as these
constructions allow. It is hypothetical, so it is computed on a relaxed instance and need not fit
the step limit that was just exceeded; the chip, storage included, is the instance's own. Both
constructions build each target on its own from an empty chip, so the remaining cost is a sum
over targets, cached per (target, remaining demand, fluid kinds, chip limits), so a generator
drawing a fresh problem per episode does not grow the cache per instance (it is cleared at
200,000 entries; values never depend on it); the object is picklable for `AsyncVectorEnv`.

**Scope.** Every reachable target: one whose reduced denominator has no prime factor above M
(a mix of k + l <= M units can introduce no larger prime), given storage for the tree's depth.
Outside that it raises `UnsupportedFallbackError` naming the instance and the target rather than
inventing a number (rule 9); `supports(instance)` checks before training, and the env refuses
such an instance at reset (9.1). Measured: storage capacity 3 suffices for every 3-fluid target
at denominator 64 on M = 8; at capacity 2, 120 of those 2,079 have no plan. The same estimate
serves as tabular Q's value prior (11.3).

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
- `tree.py` (0.12): the **split tree**, for any number of fluids. Factor the target's
  denominator D into mix sizes within `[min_mix_volume, M]`; at the top level with factor e the
  target is an equal-parts blend of e pieces of denominator D / e (consecutive chunks of the
  numerator vector, fluid by fluid, so most are pure); identical pieces merge, each distinct
  piece is a reservoir or is built recursively, and the pieces are folded into a running droplet
  one at a time (every mix at most e times a volume scale s, <= M; about two droplets per level
  on chip). A split's top mix is a multiple of a factor of D, so above M/2 it often cannot hold
  the droplet's volume (ninths at M = 8 come in mixes of 3 or 6). Then the droplet is made by a
  **dilute-up** (0.14): one mix of s units (volume <= s <= M), k units of a built droplet x_a with
  l = s - k units of a pure fluid, x_a = (s·x − l·e_fluid)/k; fewest mixes first, at most 4
  stacked. The split is always tried first, so every plan the split alone built is unchanged
  (checked on 71,250 single targets: 0 changed, 9,508 newly built). With it the tree builds every
  dilution n/D (D <= 64, M-smooth) at every volume up to M for M = 3–10.

  0.15: **storage-aware.** Each recipe's storage peak is computed exactly (the running droplet
  waits on chip while the next piece is built; leftovers go to waste after each mix), and a
  recipe must fit the chip's storage capacity; when the split does not, a dilute-up (at most
  two droplets on chip) or a **blend** does. A blend makes the droplet as one mix of two built
  droplets a and b (a is the target with one 1/D unit moved between two fluids; b is what makes
  the blend exact), needed for a many-fluid target at volume M where neither input can be pure
  (5/16, 5/16, 3/16, 3/16 at volume 3 on M = 3); its inputs use no blends. The search deepens
  one extra step at a time (so the shortest recipe is found first) and stops after
  `SEARCH_BUDGET` = 20,000 expansions per query: a target with no recipe under tight storage is
  "not found" instead of searched for minutes; the budget counts expansions, not seconds, and
  each query starts from an empty memo, so results are deterministic. Measured: at capacity 2
  every 3-fluid 1/64th on M = 8 in the test sample is built (the split alone needs 3 for some);
  6 fluids at M = 3 still need 3–4; the slowest query at capacity 2 takes about 2 s.

  0.15: **tolerant targets.** For a target with tolerance ε > 0 the tree builds the
  composition with the smallest reachable denominator (prime factors <= M, at most
  `max_denominator`) within ε, found exactly (`within_tolerance`; FacDA's ratio
  transformation, which FacDA applies to the nearest grid instead): 70/119 at M = 6 within
  0.005 becomes 7/12. One droplet serves as many deliveries of its target as its top mix allows. It
  builds **every reachable target** (tested exhaustively on 3- and 4-fluid targets at several
  denominators and M), shares nothing between targets, and returns `None` (not found, never
  "infeasible") when storage is too tight or `min_mix_volume` > 2 blocks a factor. Example:
  3/16 S + 5/16 R + 1/2 B at M = 8 is `Mix(S 3, R 5)` then 1 unit of that + 1 of B; naive
  cannot build it (it builds 35 of the 2,079 three-fluid targets at denominator 64 on M = 8).
  It is never cheaper than a proved optimum (tested on `tiny_exact_v1` and small mixtures).
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
- `share.py` (0.24): the **sharing pass**. The targets are built one delivery group at a time
  on one chip; every leftover stays in a pool of idle droplets instead of going to waste. Per
  group it prices, on a copy of the plan, (1) a pool droplet within tolerance, (2) one mix of
  a pool droplet with a reservoir or another pool droplet, (3) the cheapest **chain** of up to
  `CHAIN_DEPTH` = 3 mixes worked back from the target with `dilute_up_input`, each step adding
  a fluid or one of the `BLEND_PARTNERS` = 4 idle droplets nearest the target, down to a pure
  fluid or a pool droplet, and (4) the split tree from scratch, taking any piece already in the
  pool (`PlanBuilder.reuse`); it keeps the least marginal cost per delivery. When a mix needs
  room, the idle droplet furthest from every open target goes to waste first, so a recipe that
  fits an empty chip still fits. Four target orders (given, dearest first, cheapest first,
  nearest neighbour) are run and the cheapest plan is kept, never one dearer than
  `cheapest_construction`. `lookahead = k` rolls each group's k cheapest candidates out to the
  end (pilot method; about k × groups slower, a workstation option). The chain also finds
  cheaper single-target recipes under tight storage (3/4 at volume 2 on M = 6 through 1/2: 132
  against the tree's 196). Measured in paper log E-17: at the proved optimum on 51 of 65 small
  problems, within 1.02 (median) of the lower bound on the medium multi-target dilutions;
  median 0.04 s, at most 34 s per problem on one core.
- Further published methods (multi-target MEDA mixing, demand-driven mixture preparation) are
  added later behind the same interface, each with a reproduction test. The sharing pass is
  not one of them: "WSPM incremental" (one droplet per target, leftovers reused in sequence)
  is its nearest relative, but no published method is claimed for it.

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
  does the exploring. `fluid_bound` and `zero` are optimistic; the bound was loose (≈8.8 against
  a true 36.5 on 7/8, measured before 0.21), so every unvisited action outranks everything learned and the agent
  chases optimism instead of settling. With an optimistic prior the greedy policy also prefers
  visited actions, since the prior is then a guess rather than a value.

Truncation and dead ends are terminal with no bootstrap: the environment already adds
`cost_to_finish` to that reward, so bootstrapping would count the rest of the plan twice.

Acceptance: matches the exact optimum on every tiny instance it is trained on, **except 3/10 at
M = 8** (see 16).

### 11.4 PPO with a set encoder (`ppo/`)

**Network v2 (`networks.py`).** *Status: built (0.11); v1 is removed.* It reads observation v2
(9.2). Every small MLP is Linear → ReLU → Linear with hidden width d, as in v1. Write
`d` for `d_model`. No parameter's shape depends on F, T, the number of droplets or candidates,
M, storage capacity or any padding size (rule 1.4.1); every aggregation over a set is a masked
mean or a softmax attention, never a sum over a count (1.4.4).

1. **Fluid embeddings, informed by how the targets use each fluid.** For fluid f,
   `use_f = mean over valid targets t of MLP_u([target_comp[t, f]; targets[t]])`, then
   `x_f = MLP_f([fluids[f]; use_f])` → d. Two fluids of the same kind start from the same row;
   they become different through `use_f` whenever the targets use them differently.
2. **Composition embedding**, shared by droplets, targets and candidate results:
   `comp(c) = MLP_c(Σ_f c_f · x_f)`, the fraction-weighted sum of the fluid embeddings. It is
   the same operation as a linear layer on composition columns, except that the weight for
   fluid f is computed from fluid f's own embedding instead of belonging to column f, so it
   exists for any number of fluids.
3. **Tokens.** Fluid tokens `x_f`; droplet tokens `MLP_d([droplets[i]; comp(droplet_comp[i])])`;
   target tokens `MLP_t([targets[t]; comp(target_comp[t])])`; a global token = a learned vector
   + `MLP_g(globals)`.
4. **Set encoder.** A pre-norm Transformer encoder (2–4 layers, 4–8 heads, dropout 0, **no
   positional encoding**) over [global; fluids; droplets; targets], with key-padding masks and
   **relation biases**: for each related pair, every head's attention logit gets a learned
   **linear function of the pair fact with no constant term**, separately in each direction:
   droplet ↔ fluid from `droplet_comp` (the fraction; a 1 → heads map), target ↔ fluid from
   `target_comp` (1 → heads), droplet ↔ target from `droplet_target` (distance, satisfies;
   2 → heads). Each layer has its own maps. A zero fact (a fluid the droplet does not contain,
   a padded pair) and every unrelated pair get bias 0, so the pair facts shape attention
   without becoming columns. *As built:* the layer is written out (pre-norm attention over
   `torch.nn.functional.scaled_dot_product_attention` with the bias as an additive mask, then a
   4d ReLU feed-forward), not `nn.TransformerEncoderLayer`: in eval mode that class switches to
   a fused kernel that treats a per-head float mask differently from training mode (measured:
   logits 0.09 apart, against a spread of 0.05), which would make rollout log-probs disagree
   with those of the update.
5. **Candidate scorer.** For each candidate, concatenate: its scalar features; `comp` of its
   result (`cand_comp`); a target summary `r` = softmax attention over the contextual target
   embeddings, scored by a small MLP of `cand_target` (masked to valid targets); the contextual
   embedding of operand a and of operand b (a fluid, droplet or target by `cand_index` type, or
   a learned blank for Discard); the global token. MLP → one logit. Candidates do not attend to
   each other, so cost stays linear in their number. `comp` uses the fluid embeddings of step 1
   (before the Transformer), so compositions get context only through `use_f`: deliberate.
6. **Masked categorical** (`policy.py`): masked logits are set to the dtype minimum, so a masked
   candidate has probability exactly 0 (rule 2.4).
7. **Value head:** MLP on the global token → scalar. Returns grow with problem size (more
   targets cost more), so value targets span a wide range in mixed-size training. Diagnostic,
   logged from the start: value explained variance per size bucket. Remedy, added only when that
   diagnostic shows the need: per-task or running-scale value normalisation (PopArt; see
   `notes/training_regime_research.md` 3.4).

Known weak spot, to test: two fluids of the same kind that every target uses identically get
identical embeddings. That is right by symmetry for the problem as a whole, but a droplet
mixing them in unequal ratios is then embedded like its mirror image; the exact
`cand_target` relations still tell the outcomes apart. Ablation: a random per-episode identity
feature per fluid (`agent.fluid_identity`; reserved, not built: added with the ablation).

**Batch trimming.** Observations are trimmed to the batch's largest real row count per token
axis (fluids, droplets, targets, candidates; at least 1) before the network sees them, every
array along every axis it has (`policy.KEY_AXES`); the rollout stores each env's observation
trimmed to its own widths and pads each axis only to the widest observation of a minibatch
(`cand_index` with −1, the rest with 0). Candidate counts range from tens to thousands (random
play reaches about 9,300 at M = 10, F = 6, storage 32), so the stored rollout grows with the
mean width rather than the widest step.

**Chunked updates (0.18).** Padding a minibatch of 512 observations to its widest made one
update ask for gigabytes of GPU memory under a random policy on stages 1–3 (widths up to about
5,000; the first workstation run held 21 of 24 GB and hit an out-of-memory retry). The update
now sorts a minibatch by candidate width and splits it into chunks of at most `max_chunk_rows`
padded candidate rows (observations × the widest of them; one wider observation is a chunk of
its own). Each chunk's loss is weighted by its share of the minibatch and the gradients are
accumulated before the one clipped optimiser step, with advantages normalised over the whole
minibatch first, so the update equals the unchunked one (tested to 1e-5); only peak memory
changes. A chunk is built by one preallocated array per key and a slice copy per observation:
`np.pad` per observation and key was most of an update's time (measured on stage 1–3 random
play, 32 × 128 steps: 1.23 s an epoch against 0.17 s).

Measured on the workstation (0.18; 32 async envs, A5000, stages 1–3, random policy, before the
slice copy): an iteration of 4,096 steps took 5.8 s waiting for the env workers, 1.6 s in the
policy and copies and 7.6 s in the update, so about 270 steps/s end to end (the logged
`env_steps_per_second` counts collection only). GPU memory held at 2.7 GB with chunking.

**Network v1 (removed in 0.11).** MLPs on v1 source and target rows (widths fixed by `F_max`,
`T_max`), the same Transformer without relation biases, and the same scorer.

**Algorithm (`ppo.py`, CleanRL style):** vectorised rollouts; GAE(λ); clipped surrogate;
value loss (optionally clipped); entropy bonus; gradient-norm clipping; learning-rate
annealing; several epochs of minibatches per rollout. γ = 1.0 by default (finite episodes).
`reward_scale` (0.15, default 1) multiplies rewards before GAE: with γ = 1 a positive constant
leaves the optimal policy unchanged, and it keeps value targets near unit scale (under
`costs/geometric` an untrained episode returns about −4,750 and the value loss starts near
2×10⁵). Episode metrics stay in plan-cost units.

**As built (M8).**
- `networks.py`: network v2 above, `SetEncoderNet(NetConfig)`. It takes no padding size or
  env size at all, which is what lets one set of weights read any instance.
- `policy.py`: masked logits are set to the dtype minimum, so a masked candidate has
  probability exactly 0. Observations are trimmed per token axis before the network sees them.
  `PPOAgent` implements `Agent` (`save` / `load` keep the weights and `NetConfig` only).
- `rollout.py`: an episode that ended (success, truncation or dead end) is never bootstrapped,
  because the env already charged `cost_to_finish`. The only bootstrap is at the end of a
  rollout, for episodes it cut off. Observations are stored trimmed per env and step, as owned
  copies taken before the vector env steps (`split_obs`: a view would pin the whole padded
  batch, and a `copy=False` vector env reuses its buffer), and padded per minibatch
  (`minibatch_obs`).
- `ppo.py`: `PPOConfig.from_dict` (unknown keys raise; `clip_vloss` added to the config);
  advantages normalised per minibatch. `PPOTrainer` takes the vector env from the caller, so
  where problems come from is not its concern, and it refuses one that is not `SAME_STEP`.
  Its randomness is the torch RNG (sampling) and its own numpy generator (minibatch order).
- `utils/checkpoint.py`: weights, optimiser, counters, torch (and CUDA) RNG, the trainer's
  generator, one env snapshot per worker (9.1) and a free `extra` slot. Loading checks that
  config and observation sizes match, and rebuilds the current observations with `observe`.
- `utils/logging.py`: `metrics.jsonl` (one object per update, the record tests read),
  TensorBoard, and a console line.
- `PPOTrainer` still takes the env sizes, only so a checkpoint is resumed into the same
  observation layout it was saved from.
- Observed on `tiny-sharing-m2` (network v1): the greedy plan reaches the optimum by update 10 (8 envs × 64
  steps). After entropy collapses, the policy occasionally wobbles: with every advantage near 0,
  per-minibatch normalisation scales noise up. LR annealing damps it, and nothing is tuned for
  it yet.

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
clip_vloss: true
ent_coef: 0.01
vf_coef: 0.5
max_grad_norm: 0.5
lr: 3.0e-4
anneal_lr: true
# fluid_identity: false # reserved for the network v2 ablation (11.4); not built yet
update_epochs: 4
num_minibatches: 8
total_env_steps: 5_000_000
max_chunk_rows: 131072   # candidate rows per forward pass in an update (memory, not maths)
```

**Collectors** (`train.collector`). `lockstep` (default): a vector env (`train.vector`) steps
all envs together and the policy runs once per step on their batch, so each step waits for the
slowest env. `workers` (`ppo/collector.py`, `ActorPool`): one process per env, each holding a
copy of the policy, plays its whole rollout of `rollout_steps` actions alone; the learner then
builds the same `RolloutBuffer` and runs the same update. This is synchronous on-policy PPO with
decentralised actors (as DD-PPO), not an asynchronous method (IMPALA, APPO): no actor ever uses
weights older than the current update. Safeguards:
- every worker receives the same weights, stamped with the update number, and returns its
  rollout with that stamp; a different stamp or length is refused;
- workers share nothing mutable; the learner changes the weights only after every rollout of
  the update is back;
- before the first optimiser step of an update, the learner's log-probabilities and values on
  the first minibatch must equal the collected ones to 1e-3 (absolute log-prob, relative
  value; measured ~1e-6); a larger difference stops training (also checked under lockstep);
- each env samples with its own `torch.Generator` (`actor_seed(seed, i)`) from a batch-1
  forward on one CPU thread, so a run does not depend on timing; the generators are saved in
  checkpoints, and a checkpoint resumes only with the collector that wrote it;
- workers start with `spawn` (never fork a process that has initialised CUDA), ignore Ctrl-C,
  and a failed or dead worker stops the run with its traceback; only env methods that leave
  the episode alone (`set_instance_source`, `observe`) may be called through the pool.
The collected log-probabilities come from a CPU pass and the update's from the training
device, so runs are reproducible per collector but not bit-identical across collectors:
regimes that are compared must use the same collector. Measured on a laptop (6 envs, CPU):
collection took about 3 s per update against 7 s in lockstep.

**Logged every update:** mean episode return, success rate, dead-end rate, mean cost and each
cost component, policy entropy, approximate KL, clip fraction, value explained variance,
legal-candidate count, env steps/second, the largest first-minibatch differences from the
collected values (`collected_logprob_max_diff`, `collected_value_max_diff`), and where the
iteration's time went: `seconds_env` (waiting for the env workers; with `workers`, their whole
rollouts including their policy passes), `seconds_policy` (the rest of collection: policy
forward, copies) and `seconds_update` (the PPO update).

**Imitation warm start (0.23 planned; teacher and hints 0.25; built 0.26; paper log D-30–D-32,
E-15–E-20).** Random play almost never finishes a stage 1–2 problem (E-11), so PPO alone has no
successes to learn from. The teacher is the sharing pass (`share_plan`, 10.4; never dearer than
the construction; `imitation.TEACHERS`, `teacher=construction` keeps the old one). Demonstrations
play its plan through the env and record the teacher's candidate index per step, matching a
mix up to operand order (5.5) and an action on an unlisted identical droplet by canonical key,
with a uid map so later actions follow the droplet the env used (0.27, F-16). By D-31 the teacher is imitated on stages 1–2 only, and RL must
be able to leave it: it is not optimal everywhere, even on stage 1 (E-19).
1. **Copy test** (`scripts/imitation_check.py`): cross-entropy on the teacher's index on fresh
   generator problems (validation and held-out problems excluded); reports agreement (all
   steps and Mix steps alone) on the validation sets' teacher steps and greedy success and gap
   on `val_s1_v2`, `val_s2_v2`; saves the network (`imitation.pt`, a `PPOAgent` file). Passed
   in 0.25 with the hint columns: s1 95%, s2 72.5% (E-20).
2. **PPO from those weights** (0.26): `train.init_from=<imitation.pt>` loads the network into a
   fresh run (config must match; optimiser fresh; ignored on resume). With
   `agent.imitation_coef` w0 > 0, every minibatch adds `w × cross-entropy` on
   `agent.imitation_batch` demonstration steps sampled with the trainer's generator, w falling
   linearly from w0 to 0 over `agent.imitation_steps` env steps; demonstrations are
   `agent.imitation_problems` teacher episodes drawn from the training source before training
   (their own generator, so a resume rebuilds them). Logged: `imitation_coef`,
   `imitation_loss`, `imitation_agree`. Needs `train.source` generator or regime. Pass bar:
   s1 stable at ≥ 90%, s2 rising past the copy test, and more plans cheaper than the teacher's.
   The value head starts untrained while the encoder is shared, so use a `reward_scale` that
   keeps returns near unit (0.01 under `costs/geometric`).

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
`num_targets`, `total_demand`, the input axes of 1.3 (`num_fluids`, `max_mix_volume`,
`storage_capacity`, `max_droplet_volume`), wall time, nodes expanded, every metric of 12.1, and
gaps.
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
- For the agent goal (1.5), report gap and success rate as curves against each input axis of
  1.3, marking where the trained range ends.

---

## 13. Configuration, runs and reproducibility

- Hydra root `configs/config.yaml` with defaults: chip, costs, generator, env, agent, train, eval,
  stages (`track2_v1`) and regime (`expanding`); the last two are read only by
  `train.source=regime`.
- `meda.train` trains one agent. Tabular Q (M6) trains on a frozen benchmark set, one table per
  instance (`train.per_instance`), because it keys on exact states and the acceptance criterion
  is per instance; PPO (M8) will train from the sampler. Each run writes the resolved config,
  one checkpoint per table, and the greedy policy graded by `evaluate` (12.2):
  `results.parquet`, with the training statistics as extra columns, and `summary.md`.
- `meda.train agent=ppo` (M8, built so far) trains one network on the frozen instances that
  `train.benchmark` and `train.instances` select (acceptance (a)). It writes `metrics.jsonl`,
  TensorBoard logs, `checkpoints/last.pt` (every `train.checkpoint_every_updates`, resumable
  with `train.resume=<path>`) and `checkpoints/agent.pt`, and grades the greedy policy into
  `results.parquet` and `summary.md`. Every `train.eval_every_updates` it also grades the
  training instances (`eval/` in the logs). `train.collector` picks `lockstep` (with
  `train.vector`: `sync` or `async` workers) or `workers` (11.4);
  `train.device: auto` picks CUDA when available, else CPU (MPS is opt-in).
  `train.gpu_memory_fraction` (default 0.5) caps how much of the GPU's memory the process may
  allocate (`torch.cuda.set_per_process_memory_fraction`): the workstation is shared, so a run
  that needs more raises an out-of-memory error instead of crowding out other users.
- `train.source=generator` (0.15, PPO only): every episode draws a fresh problem from the
  `generator` config (`GeneratorSource`, 8.1), never one of `train.exclude` (default
  `tiny_exact_v1`, `val_stage1_v2`) or of `train.benchmark`, which then serves only to grade.
- `train.source=regime` (0.18, PPO only): draws from the cells of the `stages` design at the
  shares `regime` sets (`MixtureSource`, 8.3), on the design's own chip config, excluding every
  validation set of the design and `train.exclude`. Every `train.eval_every_updates` the greedy
  policy is graded on every stage's validation set (`eval/<stage>/...` in `metrics.jsonl`, the
  forgetting record; `eval/seconds` the time taken), then the regime may promote and the new
  shares go to every env worker (`call("set_instance_source")`), taking effect at its next
  reset; stage shares are logged under `regime/weight`. The round runs before the checkpoint, so
  `last.pt` holds the regime after it (`extra["regime"]`); a resume restores the regime and
  resends the shares. The run writes `regime.json` (state and promotion events) and grades the
  final policy on every stage's set and on the held-out set (`results.parquet` with a `stage`
  column, one `summary_<set>.md` each). Measured on a laptop: a round over the three pilot sets
  (120 instances) with an untrained policy takes about 10 s.
- Example commands:
  ```bash
  python -m meda.train agent=tabular_q
  python -m meda.train agent=tabular_q train.benchmark=tiny_exact_v1 agent.episodes=40000
  python -m meda.train agent=ppo "train.instances=[tiny-sharing-m2]" env.L_pad=128
  python -m meda.train agent=ppo train.resume=outputs/<run>/checkpoints/last.pt
  python -m meda.train -m agent.lr=1e-4,3e-4 seed=0,1,2        # sweep
  # regime pilot (8.3): 3 regimes x 2 seeds on stages 1-3
  python -m meda.train -m agent=ppo train.source=regime costs=geometric train.vector=async \
      agent.reward_scale=0.015625 agent.total_env_steps=<budget> \
      regime=sequential,expanding,mixed regime.num_stages=3 seed=0,1
  python scripts/regime_report.py multirun/<date>/<time> --out notes/track2/regime_pilot.md
  python scripts/regime_report.py outputs/pilot --at-step common   # runs still going or stopped
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
| Realistic M, fluid count and storage capacity of the target platform (centre of the training ranges of 1.3) | suggested ranges in 1.3 | generator configs | Supervisor |
| Exact targets (ε = 0) or a tolerance ε > 0; `max_denominator` as a chip input or only a search bound | ε = 0; chip config | targets; `chip.max_denominator` | Supervisor, assay |

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

(d) (added in 0.10, from the goal of section 1) one network, without retraining, reads
instances that differ in fluid count, target count, M and storage capacity, and its candidate
logits and value are unchanged when fluids, droplets or targets are reordered.

Progress. Before 0.11, on observation and network v1: (a) and (c) met. In 0.11, on observation
and network v2 (v1 removed):
- (a) met: `tiny-quarter-m2` and `tiny-sharing-m2` each reach their proved optimum (15 and 26)
  at 10,240 and 20,480 env steps (8 envs, d_model 64, 2 layers; `pytest -m slow`).
- (c) met: 2 updates, save, load into a fresh trainer and fresh envs, then 2 more updates gives
  the same metrics and weights as 4 straight updates, on CPU.
- (d) met: one network reads instances with F = 2, 3, 4, T = 1, 2, 3, M = 2, 4, 8 and storage
  capacity 3 and 10, alone or batched together, at two padding sizes with identical logits;
  logits and value are unchanged when fluids, droplets and targets are reordered.

Remaining, in order:
1. (b) after the stage design (8.3): measured on `val_s1_v2` (stage 1 of `track2_v1`), with
   `val_stage1_v2` kept as the M = 2 sanity check: success > 95% and mean `gap_to_best_known`
   < 5%, read from a regime run's `eval/s1_dilution` rounds. (The promotion rule asks for the
   same success but only a gap plateau, so promotion alone does not show (b).)
2. Done in 0.12: a general mixture cost-to-finish and witness, the split tree (10.3, 10.4).
   Every reachable target now has a truncation penalty and a feasibility witness.

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
- Observation v2: every composition row sums to 1; every pair distance and satisfies flag
  equals the exact value from `meda.core` (as v1's flags are checked against `apply`); operands
  in `cand_index` point at the rows the action names; one instance encodes identically at two
  different padding sizes, apart from the padding.

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
- Network v2: logits and value unchanged when fluids, droplets or targets are reordered (with
  `cand_index` remapped); one set of weights accepts instances with different F, T, M and
  storage capacity, and different padding sizes, in one test (M8 acceptance (d)).
- Evaluation: a hand-made results table gives the expected gaps.
- Stages and regimes (0.18): the design's stages add one axis value each and never hold the
  held-out cell; the mixture source draws only weighted cells, deterministically, never a
  frozen problem; gradients are evenly spaced on one reachable denominator; the frozen design
  sets rebuild identically and are disjoint; each regime's shares and promotion rule (plateau,
  success threshold, forced promotion) and its state round trip; a regime training run
  promotes, sends the new shares to the workers and resumes its regime (slow).

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
