# Paper log

The running record of everything this project could put in the paper: the claim, decisions and
why they were made, experiments with their numbers, failures and what they taught, and ideas for
figures. It is written for the person who will write the paper months from now and has
forgotten the details.

**How to maintain it**
- Add an entry the same day something happens: a decision, an experiment result, a bug that
  changed behaviour or numbers, a failed run, a literature finding.
- Give every entry a date, an ID (D-, E-, F-, L-), the evidence (commit, SPEC version, file
  path, run folder) and the numbers. Write the numbers down here; logs and scratch files get
  lost.
- Never rewrite history. If a later result overturns an entry, add a new entry and mark the old
  one *superseded by …*.
- Mark anything unverified as **unverified**. Mark example numbers that were made up for
  illustration as **illustrative**.
- SPEC.md stays the source of truth for the design. This file records *why* and *what
  happened*, which SPEC does not keep.

*Started 2026-10-25 from every project chat (2026-09-06 to 2026-10-25), SPEC 0.22, `notes/`
and git history. Last updated 2026-10-26 (E-24, SPEC 0.26).*

---

## 0. Status at a glance (2026-10-26)

- **Built:** exact rules engine, mixing-DAG validator, Gymnasium env with masking, generator
  with feasibility witnesses, exact search with a proved lower bound, split-tree construction,
  sharing pass (teacher and "finish from here" planner), candidate hint columns, evaluation
  harness, PPO with a size-free set encoder (obs/net v2), imitation warm start with a fading
  imitation term, 5-stage design `track2_v1`, three training regimes, `workers` collector.
  SPEC 0.26.
- **Trained:** two 6-run regime pilots (stages 1–3) that never learned mixtures (E-9, E-10).
  Then imitation of the sharing pass with hint columns (E-20, E-21) and a 3M-step warm-start
  PPO run on stages 1–2 (E-22): s1 100% and cheaper than the teacher on average, s2 85% after
  peaking at 97.5%.
- **Open:** why s2 decays after the imitation term ends (E-22); stages 3–5 (sharing-aware
  hints, certifying the teacher on mixtures); the regime question; paper benchmarks.

---

## 1. Paper framing

### 1.1 Problem
Sample preparation on MEDA (micro-electrode-dot-array) digital microfluidic biochips. Unlike
DMFBs, MEDA droplets have variable integer volumes, so mixing is k:l with k + l ≤ M, and a
droplet can be partly taken. The task: given fluids (sample, reagents, buffer), a set of target
ratios with per-target demand and droplet volume, and chip limits (M, storage), produce a mixing
DAG that meets every demand at low cost (fluid, mixes, waste).

### 1.2 The claim (fixed by the user, 2026-10-21)
> One policy produces near-best-known plans for MTMD mixture preparation on MEDA with a storage
> limit, a class no published method handles, while matching specialised methods on the
> sub-classes they do handle.

- MTMD = multi-target, multi-demand. Mixture = N ≥ 2 fluids.
- The user's framing: fill the MTMD gap; be competitive with every existing heuristic on the
  subsets of constraints it covers. The supervisor named no anchor baseline.
- Generalisation (one policy across M, F, T, storage, accuracy, no retraining) is a
  **requirement** of the agent, not the headline. Classical heuristics already work for any M
  and any ratio without training, so "it generalises" alone is weak against them (2026-10-21
  analysis). The headline is the frontier.
- How it is reported (SPEC 1.5): gap-vs-axis curves (M, F, T, demand, storage, accuracy), inside
  and beyond the trained ranges, plus success rate and inference time.

### 1.3 Literature positioning (narrowed wording, 2026-10-22)
Suggested sentence after reading the local PDFs (L-3):
> To our knowledge, no MEDA sample-preparation method accepts a per-target droplet demand vector
> over multiple targets or for multi-reagent mixtures, and none enforces an on-chip storage
> limit. The closest are TVODA, which takes a total output volume for a single dilution target,
> and multi-target methods (e-MRCM, MTWM, incremental WSPM), which produce one droplet per
> target. Demand-driven, storage-aware planning (DDMT) exists only for 1:1 DMFBs.

**Caveat until read:** the robust-aliquot ILP (Poddar et al., ACM TODAES 26(1), 2020) and Dong,
Chen, Chen 2024 (JETTA 40(1)) PDFs are not on disk. They are the remaining risk to this
sentence.

### 1.4 Why RL at all (argument to make in the paper)
The transition function is known and exact, so "the model is unknown" is not the reason (the
user's own commit d0da1f9, 2026-10-13: "we have the transition function, Q learning is
pointless"). The case for learning:
1. **Amortisation and tractability.** Exact search is intractable past tiny problems (E-4, E-6);
   a trained policy answers in milliseconds for any new target set.
2. **The learned value is the missing heuristic.** V* is the perfect A* heuristic, and no
   heuristic exists for MEDA k:l mixing with many targets, demand and storage.
3. **Later, the cost becomes genuinely unknown** (completion time from the router), which is the
   original project thesis (PROJECT_CONTEXT).

### 1.5 Candidate contributions (draft, revise as results come in)
1. An exact, masked RL formulation of MEDA sample preparation with afterstate candidate features
   (the network never does arithmetic) and a validator for every plan.
2. A size-free policy network (set encoder with fluid tokens, compositions as fraction-weighted
   fluid embeddings, relation-biased attention, per-candidate scorer) that reads any F, T, M,
   storage with one set of weights.
3. A split-tree construction that witnesses every exactly reachable target (any F, any M,
   storage-aware, droplets up to M, tolerant targets), used as feasibility witness, failure
   penalty and baseline.
4. A grading protocol for problems no exact method can solve: an admissible, consistent lower
   bound (tolerance-aware, whole-unit draws, waste and mix terms) as an optimality certificate.
5. A measured map of where headroom exists (E-5): sharing and reuse across targets and
   deliveries, and tight storage; single-target dilution is near-solved by construction.
6. Training lessons: the give-up pathology of construction-based failure penalties (F-11) and
   the exploration wall in exact-ratio planning (E-11). (Negative results are publishable;
   PROJECT_CONTEXT says so explicitly.)

---

## 2. Timeline

| Date | What happened | Evidence |
|---|---|---|
| 2026-09-06 | Rung-1 prototype in a separate repo (`RL-experiment`): dilution env, BFS oracle, tabular Q. Agent "beat the oracle" → env bug. | F-1 |
| 2026-10-05/06 | M0–M3: scaffold, exact core, DAG + validator, Gym env. | `c6ec142`, SPEC 0.2–0.4 |
| 2026-10-10 | M4: sampler with witness plans, naive baseline, frozen sets. | `fa5a788`–`9b47627`, SPEC 0.5 |
| 2026-10-13 | M5 exact search, M6 tabular Q: 9/10 tiny optima; table breaks at M = 8. | E-1, E-2, `6c223f7`–`6c49d56`, SPEC 0.6 |
| 2026-10-15 | Train/val leakage fixed; benchmark tiers drafted. | F-5, `1fea2bc`, `eaff56a`, SPEC 0.7 |
| 2026-10-15/16 | M7 evaluation harness; exact search returns a proved lower bound; `val_stage1_v2` with intervals. | E-3, `460decf`–`8ee518a`, SPEC 0.8 |
| 2026-10-16 | Literature review on training regimes (`notes/training_regime_research.md`). | L-1 |
| 2026-10-16/18 | M8 v1: env snapshot/restore, PPO + set encoder; acceptance (a) and (c). | `2a4e15a`, `c24d953`–`0222ac7`, SPEC 0.9 |
| 2026-10-18/19 | Goals redefined: one general agent, no baked-in limits; obs/net v2 specified. | D-8, D-9, `1899bac`, SPEC 0.10 |
| 2026-10-20 | Obs/net v2 built; acceptance (d); split tree; depth step limit; MEDA-first; input ranges fixed. | F-6, D-10–D-15, `4571fd2`–`a669171`, SPEC 0.11–0.13 |
| 2026-10-20/21 | Capability-ceiling audit: droplets up to M, storage-aware tree, blends, tolerant targets, training from a generator. | F-7, `d74334b`–`b83e548`, SPEC 0.14–0.15 |
| 2026-10-21 | Claim fixed; experiments E1–E3 (coverage, headroom, grading). | E-4–E-6, `d2c1478`, `4999101`, `6bbc2fb`, SPEC 0.16–0.17 |
| 2026-10-22 | Stage design `track2_v1`, three regimes, val sets; throughput engineering; pilot 1 launched; paper checks; `workers` collector. | D-18–D-22, F-8–F-10, `8c431bf`–`8f20015`, SPEC 0.18–0.20 |
| 2026-10-23 | Pilot 1 fails (agent learned to give up); bound inadmissible on tolerant targets; both fixed. | F-11–F-13, `2593e3b`, `0680c62`, SPEC 0.21–0.22 |
| 2026-10-24 | Workstation checks pass (619 tests, bound check PASS); pilot 2 launched. | E-8 |
| 2026-10-25 | Pilot 2 plateaus; random-play diagnosis → exploration, not reward. User chooses imitation warm start. | E-10, E-11, D-30 |
| 2026-10-25 | Greedy "get closer" policy solves 20% of s1, 0% of s2: the agent sees the distance but lacks the algorithm. | E-14 |
| 2026-10-25 | Teacher used only where near-optimal (stages 1–2); imitation fades so RL can override; copy test first. | D-31, SPEC 0.23 |
| 2026-10-25 | Imitation copy test: agreement plateaus at 64% (train), s1 greedy 22–35%, s2 ~0%; misses are mix ratios (number theory the network cannot see). | E-15, `903802c` |
| 2026-10-25 | Pilot 2 final (9.83M): expanding best (s1 38–42%), s2 0% in every run; no regime chosen. | E-10 |
| 2026-10-25 | Network gets the first mix right 24% (s1) / 6% (s2) even on trained problems; an exact 'mixes left' feature lifts s1 greedy 25% → 67.5%. | E-16 |
| 2026-10-25 | Teacher study: sharing pass reaches the proved optimum on 51 of 65 (construction 34), never dearer; much closer to the bound on stages 3–5. | E-17, SPEC 0.24 |
| 2026-10-25 | Sharing pass from mid-episode states: always finishes, 4–9 ms (s4 1.8 s); the current penalty charges 2–25× the real cost to finish (no credit for intermediates). | E-18 |
| 2026-10-26 | Finish-from-here planner as penalty, hint columns and teacher (SPEC 0.25). Cheap moves + finish beat the teacher on 7/40 s1 problems (168 vs 475). | D-32, E-19 |
| 2026-10-26 | Copy test with hints + sharing teacher: s1 greedy 95% (70% before the last epoch), s2 72.5% (was 0%), 12 / 6 plans cheaper than best known. Step 1 passes. | E-20 |
| 2026-10-26 | Workstation rerun and ablation: hints are the cause (old teacher does as well); s1 copies at ~70–77% (E-20's 95% was noise), s2 ~65%. Step 2 launched from this network. | E-21 |
| 2026-10-26 | Warm start + PPO, stages 1–2, 3M steps: s1 100% (cheaper than the teacher on average), s2 peaks 97.5% just after imitation ends, settles at 85%; held-out cell 50% zero-shot. | E-22 |
| 2026-10-26 | s2 failures are exactly the 6 longest mixture plans (23–41 teacher steps); 11% of training s2 problems are that long. Plan rebuilt in `notes/plan.md`. | E-23 |
| 2026-10-26 | s2 replay: the warm-start agent does not loop; it hoards droplets (10–25 on chip, 0–3 discards vs the teacher's 7–18) and dithers with low confidence until the step limit. | E-24 |

---

## 3. Decisions

Format: date · decision · alternatives considered · why · evidence.

### Formulation and environment
- **D-1 (2026-09-06) Unrepresentable mixes are illegal, never rounded.** Alt: round to the
  nearest grid value. Why: rounding makes the arithmetic inexact, breaks the oracle (the optimum
  of a lie), and errors compound. Every k/16 stayed reachable; the constraint only orders the
  work.
- **D-2 (2026-09-06) Reward −cost per step, γ = 1, no terminal bonus.** Why: V*(s) then equals
  minus the oracle's optimal cost, so Q-values can be checked entry by entry against brute
  force. "Choose the reward so the optimal value equals something you can verify."
- **D-3 (2026-10-05) Exact integer arithmetic in the core; cost weights as exact fractions**
  (0.5 read as 1/2). Why: targets have tolerance 0, and exact search must compare costs with no
  rounding. SPEC 0.2.
- **D-4 (2026-10-05) Action = index into the legal list (masking), with afterstate candidate
  features** computed exactly by the env (TD-Gammon idea). Why: the raw action space is huge and
  mostly illegal; the network should judge outcomes, not compute ratios (rule 1).
- **D-5 (2026-10-05) Unfinished episodes are charged a real cost-to-finish** (build the rest with
  a construction and price it), with no bootstrap after the episode ends. Alt: a constant
  penalty. Why: a constant is either too small (agent stalls) or the wrong scale across problem
  sizes. *Partly superseded by D-26 (factor 2).* Idea borrowed from AlphaTensor.
- **D-6 (2026-10-10) Every sampled instance carries a replayed witness plan,** so "solvable" is
  proved, not assumed; the witness cost is a known upper bound.
- **D-7 (2026-10-13) Tabular Q is a test of the env formulation, not a method** (user commit
  `d0da1f9`). It stays as the cheapest end-to-end canary in the slow tests.

### Goal and architecture
- **D-8 (2026-10-19) One general agent, no limit baked into the weights** (user: "one-stop
  solution", nothing that works only for M = 4 or F ≤ 8). Alt: fixed F_max = 4, T_max = 12
  columns (v1). Why: v1's per-fluid and per-target columns were hard ceilings; M is an open
  physical choice (literature 4–10); amortisation needs one policy across problems and chips.
  SPEC 0.10 §1.
- **D-9 (2026-10-19) Observation and network v2:** every count is a set of rows; compositions
  are fraction matrices embedded as Σ_f c_f·x_f over contextual fluid embeddings ("the weight
  matrix became data"); pair facts become per-head attention biases; volumes given as vol/M and
  vol/16; M and storage in globals; per-candidate scorer; size-free aggregation and LayerNorm.
  Precedents for pair-bias attention (cited from memory, **verify**): AlphaFold2, Graphormer,
  ALiBi, T5, Shaw et al. 2018, RAT-SQL. Built 2026-10-20 (`4571fd2`–`2105c55`), SPEC 0.11.
- **D-10 (2026-10-20) MEDA first; 1:1 (M = 2) is a sanity check only** (user: "MEDA ain't
  confined to 1:1"). Consequences: dyadic accuracy grids withdrawn; M trained 3–10; M8 (b)
  re-anchored off the M = 2 set; step-limit multiple re-measured on k:l problems.

### Problem distribution
- **D-11 (2026-10-20) Input ranges** (widest the literature supports, because generalisation is
  the argument; source PLATFORM_PARAMS.md). Train M 3–10 (probe 2, 12, 16); F 2–6 (probe 7, 8,
  10; baselines at ≤ 5); exact targets with M-smooth D ≤ 256 (probe 512); tolerant targets ε ∈
  {0.001, 0.005}; T 1–10, demand 1–4 (probe T 20, demand 6); droplet volume 1–M; storage 2–16 +
  unbounded (our count ≈ literature's k + 2). SPEC 0.16 §1.3.
- **D-12 (2026-10-20) Cost weights fixed at 64 / 8 / 1, mix 1, waste 1** (e-MRCM/CoDOS geometric
  scheme; buffer 1 not 0 because a free buffer gives degenerate plans). The user first chose
  weights-as-input, then reversed it: preference conditioning is later multi-objective work.
- **D-13 (2026-10-20) Target volume = demand × droplet volume, split fixed by the generator.**
  Total-volume input (TVODA style) deferred; it needs a core change.
- **D-14 (2026-10-20) Failure penalty uses the cheaper of naive and split tree per target**
  (user: "the more accurate the penalty, the better").
- **D-15 (2026-10-20) Step limit from the problem itself ("depth rule")**: limit = ⌈4 × Σ over
  deliveries (⌈log_M D⌉ + 1)⌉, raised to fit the witness if needed. Alts: multiple of the
  witness, fixed per stage, randomised, multiple of the best known. Why: independent of any
  construction's quality. *Amended by D-27.*
- **D-16 (2026-10-20) Droplet volume up to M, not M/2.** M/2 had no recorded reason; it was the
  ceiling of our constructions (F-7).
- **D-17 (2026-10-20) Generator = a weighted combination of samplers** (exact grid, ε-tolerant,
  structured experiments); random walk retired as a training sampler. User: "avoid both
  extremes, where the agent sees everything and where it sees nothing".

### Claim, grading, stages, regimes
- **D-18 (2026-10-21) Frontier grading does not block training.** User: a cell is frontier
  because no one has tried it; where possible we grade, where not, the agent is first. Grade
  with the lower-bound certificate, exact optima on small problems, and naive/tree. Stronger
  references (sharing search baseline, faster exact search, LP/ILP, adapted literature methods)
  recorded as options, not prerequisites. SPEC 0.17 §8.4.
- **D-19 (2026-10-22) Stages = constraint cells, one constraint added per stage:** s1 dilution →
  s2 + mixture → s3 + demand → s4 + many targets → s5 + limited storage (15 training cells).
  Gradient targets for half the multi-target dilution draws (most headroom in E-5).
- **D-20 (2026-10-22) Held-out cell: dilution + many targets + demand + limited storage**
  (`heldout_combo_v1/v2`). Alt: hold out the headline mixture cell. Why not: the headline
  result would then come from an agent that never saw the headline problem.
- **D-21 (2026-10-16 and 2026-10-22) Settle the training regime by experiment, with the decision
  rule fixed before running.** User (2026-10-16): "no point making assumptions… let the results
  answer". Regimes: `sequential` (baseline, expected to forget), `expanding` (candidate:
  newest stage 50%, earlier stages share 50%, headroom weights 10/20/70 at the end), `mixed`
  (challenger: all stages from step 0). Rule: reject any regime under 95% success on any stage;
  then lowest gap to best known; if close, the simpler one, then less forgetting. Promotion on
  **validation** success (≥ 95% and gap plateau over 3 rounds), forced after a budget share and
  logged as forced. Earlier, the user also rejected the automatic ladder: stages are a
  diagnostic, not the goal (2026-10-16).
- **D-22 (2026-10-22) Collectors.** `workers` (synchronous per-worker policy copies, still
  exactly on-policy, DD-PPO style; not IMPALA/V-trace) for speed. Rule: never mix collectors
  inside one comparison; runs are reproducible within a collector, not bit-identical across.
- **D-23 (2026-10-15) Grade unprovable problems against an interval [lower bound, best known];
  M8 (b) uses gap to best known.** `gap_to_optimal` is only ever filled from a proved optimum.
- **D-24 (2026-10-15) Training never draws benchmark problems** (`exclude` wired; chip-agnostic
  `problem_hash`). F-5.
- **D-25 (2026-10-21) Report sample units saved next to cost.** Under 64/8/1, one sample unit
  dominates small plans.

### Fixes that changed the training signal
- **D-26 (2026-10-23) `failure_penalty_factor` = 2** (failing costs twice the construction's cost
  to finish). From F-11. A slow test requires a deliberate failure to cost ≥ 1.8× the best known
  plan on every validation problem.
- **D-27 (2026-10-23) Step limit `depth_support`:** counts (fluids − 1) mixes per delivery and
  is floored at 2× the witness. From F-12.
- **D-28 (2026-10-23) `max_stage_fraction` 0.2, and configs that cannot fit every stage are
  refused.** From F-13.
- **D-29 (2026-10-21) n-input mixes for baselines need no core change:** a chain of 2-input mixes
  uses identical fluid and zero extra waste, costing n − 2 extra mixes (checked in the engine).
  Plan: translate baseline plans to chains and report a merged mix count.

### Current
- **D-30 (2026-10-25) Next training approach: imitation warm start from the construction, then
  PPO.** Alts: potential-based reward shaping; easier mixture sub-stages (random play failed even
  on 3-fluid mixtures); MCTS-guided exploration (AlphaGo Zero style, heavier). Why: E-11 shows
  successes are too rare for RL to start from; we have a teacher that solves every problem and
  can generate unlimited demonstrations. Analogy: AlphaGo (supervised on expert games, then RL).
  Difference: our teacher is not optimal, so the RL phase must improve on it. Plan: short test
  on stages 1–2, one seed, ~2M steps; pass bar = stage 2 above 0% and stage 1 past 40%; only
  then re-run the regime pilot. *Plan and pass bar amended by D-31.*
- **D-31 (2026-10-25) Imitate the teacher only where it is near-optimal; RL must be able to
  override it.** The user: AlphaGo learnt from experts, not average players, so the teacher
  must be good on the problems it teaches; and the agent should not start with habits it does
  not need. Evidence on teacher quality (`cheapest_construction`): single target with loose
  storage, median headroom 0% (≤ 1.4% on dilution, E-5), so it is an expert on stages 1–2;
  tight storage, medians 1–82% per cell (up to 149%); multi-target, 50–135% above the proved
  optimum, because it never shares droplets (E-5); medium problems unprovable, interval 28–60%
  wide (E-6). So on stages 3–5 it is an average player. Decisions: (1) demonstrations come
  from stages 1–2 only; (2) the imitation term fades to 0 during PPO, so RL can move away from
  the teacher; (3) before any PPO run, an **imitation-only copy test**: can the network copy the
  teacher's moves on unseen stage 1–2 problems (move agreement and greedy success on
  `val_s1_v2`, `val_s2_v2`)? If it cannot, the warm start is pointless and the problem is the
  network, not exploration; (4) PPO test pass bar: s1 ≥ 90% success, s2 well above 0%, and
  some plans cheaper than the teacher's; (5) a separate teacher study measures a sharing pass
  and published sharing methods (e-MRCM, MTWM, WSPM) against the proved optima, the medium
  sets and the large `val_s4`/`val_s5` problems; (6) stages 3–5 are then run two ways, pure RL
  from the stage 1–2 skills and imitation of the best teacher found; if both stall near the
  teacher's cost, try self-imitation (Oh et al. 2018), then MCTS (M9). Alts: imitate the
  construction on every stage (rejected: teaches the no-sharing habit the agent must later
  unlearn); exact optima as demonstrations (only small problems are provable, E-5). SPEC 0.23
  §11.4.

- **D-32 (2026-10-26) Finish from here: one planner for the penalty, the hints and the
  teacher** (user, after E-16–E-18; SPEC 0.25). The missing fact in E-15/E-16 and in the
  penalty is the same: from this state, using the droplets on chip, what does finishing cost?
  (1) **Penalty** `cost_to_finish: share` = the cheaper of the sharing pass started from the
  state (`finish_from`) and the rebuild; factor 2 kept. (2) **Hint columns** (10 of D_cand
  27): fewest mixes from each Mix result to an open target along the tree's and the sharing
  pass's recipe, and along a depth-2 pure-fluid chain. Not the whole planner per candidate:
  4 ms × hundreds of candidates per step is too slow (E-18). (3) **Teacher** = the sharing pass
  (never dearer than the construction, E-17). Alts: MCTS straight away (needs the same
  leaf value and a prior that the network does not have yet, so it comes after); more data or
  a bigger network (E-15: 3.3× data, +9 points). **Changed guarantee:** the D-26 test bar
  ("failing costs ≥ 1.8× best known") assumed a penalty that ignores progress; it is kept for
  `construction` and replaced for `share` by a per-state one (the charge is the factor times a
  completion that replays from the final state), because an accurate penalty rightly charges
  little for a failed episode whose moves were good (E-19).

- **D-33 (2026-10-26) Fix the cause, not the stage 2 score; then move to stages 3–5** (user,
  after E-22–E-24). Stage 1 counts as learned (comparison with published heuristics still
  open). Polishing s2 from 85% to 95% is not a goal in itself, but the failure mode (hoarding
  and dithering on long plans, E-24) will get worse with more targets and demand, so its cause
  is fixed first. Proposed fix, **not yet adopted**: charge waste up front (every unit is
  charged as waste when it lands on the chip and refunded when delivered), a potential-based
  shaping (Ng et al. 1999) that leaves every finished plan's cost unchanged, makes discarding
  neutral, and closes the gap where the `share` failure penalty does not charge leftovers. The
  user asked for more conclusive evidence before this change: experiments D1–D4 (discard
  behaviour, discard probability on the teacher's own states for the copy network vs the RL
  agent, a "janitor" intervention, and the size of the uncharged leftovers). Dropped for now:
  the constant imitation anchor (run B), a crutch that may stop the agent beating the teacher.
  In parallel: the zero-shot check on stages 3–5 (plan 3.1). The user also noted that "keep a
  droplet only if it is useful later" is a lookahead question, which points toward MCTS
  (plan 4.2).

---

## 4. Experiments and results

### E-1 (2026-10-13) Exact search on the hand-checked cases (M5)
- All 10 `tiny_exact_v1` optima proved in 4.7 s total, after matching each instance's accuracy
  to its targets (at `max_denominator` 256, four instances timed out).
- Two SPEC hand cases were stated in mix counts, not cost. 3/10 at M = 8: SPEC's 2-mix plan
  costs 29; the cost optimum is 41/2 with 3 mixes, drawing 1 sample unit instead of 2 (verified
  by hand). 1/4 at M = 4: the 2-mix plan (15) beats the 1-mix plan (31/2).
- Sharing case (targets 3/4 and 1/4 at M = 2): optimum 3 mixes vs naive 4.

### E-2 (2026-10-13) Tabular Q on tiny instances (M6)
- **9 of 10** tiny instances reach the proved optimum, including the sharing case (it learns to
  reuse the 1/2 droplet).
- Three settings decided everything (F-2, F-3, F-4): the value prior, step-count aliasing, and
  the step size.
- **The table breaks at M = 8 (3/10):** ~80 candidates per step vs ~4 under 1:1; a random walk
  meets the 4-action optimum with probability ~10⁻⁶. Optimistic prior finds 20.5 but settles at
  34.5; pessimistic prior settles but its best is 29.0 after 60k episodes (180,674 table
  entries). This is the expected rung failure that motivates function approximation.

### E-3 (2026-10-14/16) Grading stage-1 validation problems
- Exact search (60 s) proved 1/16 = 20, 9/32 = 39/2 (naive 32), 7/32 = 39/2 (naive 83/2, more
  than 2×). 25/32, 17/32, 19/32, 15/16 timed out even at matched accuracy.
- Neither a tighter admissible heuristic (root bound 1.2 → 15.5) nor branch-and-bound helped:
  60–80k expansions at ~800 nodes/s. The search space, not the bound, is the limit.
- `val_stage1_v2` (2026-10-16): 3 proved optima; intervals 11/16 [23, 53/2] (1.15×), 17/32 and
  19/32 [23, 29] (1.26×), 25/32 [23, 83/2] (1.80×), 15/16 [47/2, 97/2] (2.06×). Searching at the
  targets' precision found plans at 29, below naive's 32 and 41.5.
- **Figure idea:** 7/32 on a 1:1 chip. Naive: S+B → 1/2 → 3/4 → 7/8 → 7/16 → 7/32 (3 sample
  units, cost 41.5). Optimum: S+B → 1/2 → 1/4 → 3/8 → 3/16 → 7/32, reusing the leftover 1/4
  (1 sample unit, cost 19.5).

### E-4 (2026-10-21) E1: coverage map
`notes/track2/e1_coverage.md`. 16 cells (dilution/mixture × 1/many targets × demand 1/>1 ×
storage loose/limited).
- **Covered:** no demand and no storage limit (WSPM, FacDA, MRCM, e-MRCM, MTWM).
- **Frontier:** demand > 1 together with many targets or a mixture, plus every cell with a
  storage limit.
- **Adapted references for the frontier:** DMFB demand/storage planners (MTMD, DDMT, NFSP; 1:1
  plans are legal on MEDA) as a floor, and MEDA methods run once per droplet and merged.
- Corrections from L-3 applied 2026-10-22.

### E-5 (2026-10-21) E2: headroom on small provable problems
`scripts/headroom.py`, `notes/track2/results.md`, `e2_headroom.jsonl`. 400 problems, 20 cells ×
20, costs 64/8/1, exact search 60 s each. Headroom = best construction ÷ proved optimum − 1.
- Exact search proved **65 of 399 (16%)**; by deliveries 1/2/3/4/5/6+: 30/80, 18/113, 11/86,
  1/15, 4/29, 1/76. Median 112 nodes/s; ~90 candidates per state at M = 7; ~0.4 ms per child.
  **0 of 334 timeouts** improved on the construction.
- **Single target, loose storage:** median headroom 0% (≤ 1.4% on dilution). Near-solved by the
  tree; sanity check only.
- **Single target, tight storage:** headroom in 12 of 28 proved; median 1–82% per cell, up to
  149%. The optimum builds a smaller intermediate and takes part of it.
- **Multi-target:** headroom in **14 of 18** proved; mostly 50–135% (i.e. the optimum is
  roughly 33–57% cheaper). Saves 1–4 sample units, by sharing (+1 to +3 shared nodes) and less
  waste. Gradients with demand show the most (135%).
- **Worked examples (figure candidates):** (1) M = 3, 3 units of 1/6 + 2 units of 3/4: tree 202
  (3 sample, 2 waste) vs optimum 134 (2 sample, 0 waste) by sharing one 1/2 droplet. (2) M = 3,
  storage 2, 3 × 2 units of 1/8 + 2 × 3 units of 3/8: tree 496 (7 sample, 19 mixes, 12 waste) vs
  optimum 211 (3 sample, 10 mixes, 0 waste); every leftover starts another delivery.

### E-6 (2026-10-21) E3: grading medium problems
45 medium problems (9–22 deliveries), exact search at 10, 60, 300 s.
- 0 solved. Never beat the construction. Interval [LB, best known] 28–60% wide; the bound rose
  ≤ 0.44% of plan cost from 10 s to 300 s (7k–240k nodes in 300 s).
- **Bound tightness on the 65 proved problems:** within 5% of the optimum whenever the optimum
  wastes almost nothing (14 of 18 multi-target); loose only when waste is forced (a single small
  delivery, since a mix needs ≥ 2 units). **Inference (unproved):** the medium-problem gap is
  mostly construction excess, i.e. headroom, not a loose bound.

### E-7 (2026-10-20/22) Capability and distribution measurements
- Naive builds only **21 of 1,953** three-fluid targets at denominator 64 (M = 8); the split tree
  builds all of them. Reachable iff every prime factor of D ≤ M (and D within the cap).
- Droplets up to M: nearly every dilution ratio can be made at any volume ≤ M (93% at M = 3,
  v = 3); constructions managed v = M for only 59/84/74/43/39% at M = 3/4/6/8/10 before the
  dilute-up step.
- After the 0.15 tree work: all 71,250 grid targets built (9,522 new, 0 changed); storage-2
  search from 36,000 s worst case to ~2 s with a budget; FacDA's 70/119 at M = 6 → 7/12.
- Step-limit multiple 4 had to be raised to the witness on 5–8% of draws; storage 2 left ~7% of
  many-fluid draws unwitnessed; episodes up to ~500 steps (T ≤ 10), ~1,000 (T = 20).
- **Limited storage rarely binds for constructions:** 1 of 180 problems (open decision; measure
  on trained agents).
- Lower bound before the 0.21 fix was 2–5× below best known on the stage sets.

### E-8 (2026-10-23/24) Lower bound v2 checks
- Consistency: 0 violations over 14,910 random moves across 16 cells; never above a construction
  on 242 validation problems; hand case 1/7 → 67.
- Median LB / best known: s1 0.38 → **0.66**, s2 0.19 → 0.33, s3 0.35 → 0.43, s4 0.32 → 0.33,
  s5 0.33 → 0.39. Three s1 problems became proved optimal.
- Workstation `scripts/check_bound.py` (2026-10-24): 300 problems, optimum proved on 49, **0
  failures**, bound/optimum median 1.000, min 0.508. **PASS.**

### E-9 (2026-10-22/23) Regime pilot 1
6 runs (3 regimes × seeds 0, 1), stages 1–3, 10M steps, lockstep collector, failure factor 1,
`max_stage_fraction` 0.3. Runs in `outputs/pilot_old_penalty/` on the workstation.
- 1.2M steps: s1 validation success 20–28% (expanding/sequential), 2–8% (mixed); gaps −0.08 to
  −0.20 on solved problems.
- 7.37M steps: s1 plateaued at ~20–35% after ~3M; **s2 mixtures 0% in every run**, including
  `mixed`; s3 8–22%; **every promotion forced** (3.07M, 6.14M); solved plans 10–34% cheaper than
  the best construction. Decision: none chosen.
- Cause found: F-11. The negative gaps were a symptom (selection: it only finished problems it
  could beat the construction on), not a win.

### E-10 (2026-10-24/25) Regime pilot 2 (after the fixes)
Same 6 runs with D-26–D-28 and `_v2` sets; launched 2026-10-24 ~22:47 IST.
- Early: returns more negative (−800 vs −600, the doubled penalty); positive s3 gaps appeared
  (0.57, 0.05), so the agent now finishes plans worse than the construction instead of giving up.
- 5.9M: expanding s1 **42%** (both seeds), s2 0%, s3 22%.
- 8.19M: expanding s1 38–42% (flat), **s2 0%**, s3 18–25%; all promotions forced (2.048M,
  4.096M, 6.144M); gaps still −0.08 to −0.17; max forgetting 5–9%. Decision: none chosen.
- Reading: s3 ≈ 40–50% on its dilution half + 0% on its mixture half. Regime order does not
  matter yet; both walls are about learnability.
- **To do:** final 10M numbers and per-instance `results.parquet` not yet inspected.
- **Final (added 2026-10-25), last common validation round 9,830,400 steps** (`regime_report.py
  outputs/pilot`; all runs ended, tmux gone). Success s1 / s2 / s3 per seed (s0, s1):
  expanding 38/0/18 and 42/0/25; mixed 0/0/0 and 18/0/8; sequential 15/0/2 and 32/0/18.
  Mean gap on solved problems −0.07 to −0.24 (cheaper than the construction where it
  finishes). Max forgetting: expanding 8.75%, mixed 3.75%, sequential 5%. Decision rule: no
  regime reaches 95% on every stage, none chosen. Expanding is the best of the three on both
  seeds, but no stage-2 problem was ever solved by any run; mixed seed 0 learnt nothing in 10M
  steps. The regime question stays open until stages 1–2 are learnable (E-11, E-14, E-15).
  Per-instance `results.parquet` still not inspected.

### E-11 (2026-10-25) Random-play diagnosis
30 random episodes × 40 validation problems per stage, one core.
- **Reward is now sound:** failing costs a median **6.0×** (s1) / **5.1×** (s2) the best known
  plan; a sloppy random plan that finishes costs **3.4×** (s1). Finishing always pays.
- **Exploration is the wall:** random success **2 / 1,200** (s1, 0.2%) and **0 / 1,200** (s2).
  ~200 legal actions per step (s1), ~800 and up to 3,400 (s2); correct plans are ~6 (s1) and ~12
  (s2) moves. Every failure was a step-limit truncation; no dead ends.
- Ranked reasons the current setup fails: (1) successes too rare; (2) wide branching × long
  horizon; (3) all-or-nothing reward, no partial credit; (4) hitting an exact ratio is an
  arithmetic algorithm the network must discover from pass/fail; (5) a fresh problem every
  episode; (6) promotions forced before the foundation was solid; (7) entropy fell from ~4 to
  ~1.5–2 while success was low; (8) untested whether the observation makes "distance to target"
  easy to read.

### E-12 (2026-10-22) Throughput engineering (methods / appendix material)
Workstation: 40 cores, 252 GB, RTX A5000. The env is CPU-bound (GPU 2–13%).
| Change | Effect |
|---|---|
| Copy only each env's real rows (obs padded to 16,384 candidates was ~150 MB per step) | 280 → ~540 shown sps; RAM 21 GB → a few GB (`7fb0f50`) |
| Chunked minibatch updates sorted by width (131,072-row budget), GPU memory cap | GPU 21 GB → 2.7 GB; identical weights within 1e-5 (`1ed5157`) |
| Phase timing showed `sps` excluded the update (true ~270 sps); slice-copy minibatches | update 7.6 s → 2.7 s; ~390 sps end to end (`fcbddb2`) |
| gcd skip + one-pass encode (output hash identical over 16,000 steps) | ~20% cheaper env step (`2d925de`) |
| `workers` collector | ~2× collection on the Mac; lockstep was ~21% worker-efficient (mean step 5 ms, tail 25–80 ms) (`8f20015`) |
Six concurrent runs gave ~800 sps total (GPU time-slicing). Env profile under a random policy:
legal-action enumeration ~65%, encoding ~30%, resets ~5%.

### E-13 (2026-10-20) Network facts
- 882,787 parameters (measured 2026-10-25). d = 128, 3 layers, 4 heads; untuned starting points.
- One network reads F 2–4, T 1–3, M 2/4/8, storage 3/10 and is invariant to reordering fluids,
  droplets and targets (acceptance (d)). Env ~15% slower from the pair matrices.

### E-14 (2026-10-25) Diagnosis, part 2: random-play detail and a greedy "get closer" policy
Adds detail to E-11; same sets (`val_s1_v2`, `val_s2_v2`, 40 problems each), one core, scripts
in a scratch folder (method below is enough to redo it).
- **Random play (E-11), full numbers.** Uniform choice over the legal candidates, 30 episodes
  per problem: s1 2 / 1,200 successes (on `val_s1-011`, D = 59 with tolerance 1/200, and
  `val_s1-019`, D = 2), s2 0 / 1,200. All 2,398 failures are step-limit truncations, no dead
  ends. Step limit / construction length: median 2.58, min 2.00 (s1); median and min 2.00
  (s2). Construction length median 5.5 moves (s1), 12 (s2). Legal candidates per step: median
  197, max 510 (s1); median 801, max 3,389 (s2). Failing costs median 6.02× (min 3.69×) the
  best known plan in s1 and 5.09× (min 3.68×) in s2. The "random finished plan costs 3.4×"
  figure rests on only the **2** random successes, so it is weak evidence; the penalty
  argument itself (failing ≥ 2× the construction, D-26) does not depend on it.
- **s2 composition.** F = 3 (13 problems), 4 (11), 5 (5), 6 (11); 20 exact, 20 tolerant; D from
  5 to 243. Random play solved none, **including the 13 three-fluid problems**.
- **Greedy "get closer" policy.** Each step: deliver if an Output is legal, else take the
  candidate whose result is nearest an open target (the observation's own candidate feature
  "distance to nearest open target", SPEC 9.3), ties broken by step cost. s1: **8 / 40** solved,
  all at gap 0 (they equal the construction); on the rest it gets very close (median nearest
  distance 0.0044) but never exact. s2: **0 / 40**, median nearest distance 0.071.
- **Reading.** The observation already tells the agent how close it is, and following that cue
  alone solves 20% of dilutions, about half of what PPO reached (~40%, E-10). Closer is not
  enough: an exact (or 1/200) ratio needs a planned sequence of mix ratios (number theory), and
  a near miss earns nothing. This supports reasons (3) and (4) of E-11 over (8): the agent can
  see the distance; what it lacks is the algorithm. It strengthens the case for D-30: the
  construction *is* that algorithm, and imitation hands it over.
- **Still open** (need the workstation runs): entropy and value explained-variance traces
  against success (reason 7, critic quality); per-instance s1 results (which D, M, tolerance it
  solves); a single-problem overfit test on an s1/s2 problem (reason 5: can it learn one
  mixture when it doesn't have to generalise?).

### E-15 (2026-10-25) Imitation copy test, first runs (stages 1–2, Mac, one core)
Commit `903802c`, SPEC 0.23 §11.4, `scripts/imitation_check.py`. Behaviour cloning of
`cheapest_construction` on fresh stage 1–2 problems (half s1, half s2; validation and held-out
problems excluded); default network (d 128, 3 layers, 4 heads), Adam 3e-4, batch 64, grad
clip 0.5, seed 0. "Agreement" = the network's top candidate is the teacher's, on the teacher
steps of `val_s1_v2` / `val_s2_v2` (states never trained on). Greedy success and gap on the
same sets; best known there = the teacher's cost, so gap < 0 means cheaper than the teacher.
- **Setup check.** All 80 validation plans replay through the env to success at exactly the
  teacher's cost, once a mix is matched up to operand order (without that, 27 of 40 s1 and
  37 of 40 s2 plans hit a "missing" move: the env lists `Mix(droplet, reservoir)`, the
  constructions write `Mix(reservoir, droplet)`).
- **Overfit check** (`tests/agents/test_imitation.py`, 3 s1 problems, 18 steps, small net):
  100% agreement and the teacher's plan, at lr 1e-3 with clipping. At lr 3e-3 without
  clipping the loss oscillated (agreement 78–94%).
- **Run 1: 300 problems (2,969 steps), 12 epochs, 190 s.** Train agreement 0.32 → 0.55; val
  agreement s1 0.49 → 0.66, s2 0.38 → 0.50; still rising. Greedy success s1 0.15 (epoch 3) →
  0.30 (epoch 12), with 3 plans cheaper than the teacher (mean gap −1.1% on successes); s2 0.
  Reading: the network is learning the teacher and does not yet copy it (underfit: train
  agreement 55%). A 12-step plan needs per-step agreement near 0.99 to finish, which is why s2
  stays at 0 while agreement is 50%.
- **Run 2: 1,000 problems (9,916 steps), 30 epochs, 23 min.** Agreement plateaus: train 0.40 →
  0.64 (+0.02 over the last 10 epochs), val s1 0.61 → 0.70, s2 0.42 → 0.62. Greedy success s1
  0.275 / 0.35 / 0.325 / 0.30 / 0.25 / 0.225 at epochs 5–30 (no better than PPO's ~40%, E-10;
  mean gap on successes +10% to +31%, 0–2 plans cheaper than the teacher); s2 0 except one
  success at epoch 20 (gap −43%). 3.3× the data of run 1 moved train agreement only from 0.55 to
  0.64, so more data alone is not the fix.
- **Where it disagrees** (run 2 network, teacher steps of the val sets): 65 of 65 s1 and 214 of
  215 s2 disagreements are Mix vs Mix, almost always the **ratio** k:l, most often on the first
  mix, with the probability spread thin (0.02–0.05 on both the teacher's and the network's
  choice is common). Example `val_s1-001`: target 33/40 at M = 9; the teacher starts 1:7 (1/8),
  because 40 = 8 × 5 (a mixed-radix digit); the network prefers 3:1. Order ambiguity is not the
  main cause: the misses are about which ratio, not which part first.
- **Reading.** The network cannot infer the construction's number theory (denominator
  factorisation, digit expansion) from compositions given as embedded floats; this is the
  "lacks the algorithm" wall of E-14, now seen under supervision. Imitation of moves alone
  does not hand the algorithm over. Candidate remedies (not decided): exact per-candidate
  features the env computes (rule 1), such as whether the result's denominator divides each
  target's, or the construction's remaining mix count from the result to each target; a
  larger network or longer training (unlikely to suffice given the plateau).

### E-16 (2026-10-25) Why the copy test fails: checks A and B (stage 1–2, Mac, one core)
Scratch probes (not product code): `check_a.py`, `mixes_left.py`, `check_b_measure.py`,
`check_b_train.py` in the session scratchpad; network of E-15 run 2 for check A.
- **Check A, agreement by move kind** (top-1; unseen val / trained-on problems): s1 first mix
  **0.23 / 0.24**, later mixes 0.57 / 0.53; s2 first mix **0.10 / 0.06**, later mixes 0.37 /
  0.43; Output and Discard 0.99–1.00. The 64% of E-15 is flattered by trivial moves, and
  trained-on problems score no better than unseen ones: the network cannot learn the first
  intermediate even from examples it has seen. Top-3 first mix: s1 0.40, s2 0.23.
- **Check A, greedy episodes vs the teacher:** s1: copied exactly 6, diverged and failed 31,
  diverged and finished dearer 3, ties 0, cheaper 0; s2: diverged and failed 40. The first
  divergence is at step 0 in 31/40 (s1) and 36/40 (s2). Disagreements are mistakes, not ties.
- **Check B, the feature.** "Mixes left" = the fewest further mixes, each adding one pure
  fluid (k + l ≤ M), from a candidate's result to the teacher's aim for an open target
  (`tree.aim`), exact rationals, depth ≤ 5, volumes ignored. On the val teacher steps: s1, the
  teacher's mix has the smallest mixes-left in 96/119 states (unique in 40; median 2 tied);
  the network's wrong picks are mostly 2–4 mixes left. s2: the teacher's mix is not a chain
  (> 5 or a blend of pieces) in 176/325 states, so this feature cannot describe mixtures.
- **Check B, stage 1 copy test with vs without the feature** (same 1,000 problems, 5,900
  steps, seed, net and schedule; 20 epochs; feature as a 7-column one-hot appended to the
  candidate row):

  | | train agree | first mix | later mix | greedy s1 success | cheaper / same cost as teacher |
  |---|---|---|---|---|---|
  | without | 0.68 | 0.20 | 0.51 | **25%** | 0 / 5 |
  | with | 0.74 | 0.45 | 0.63 | **67.5%** | 7 / 13 |

  Success with the feature is 65% already at epoch 5. 7–10 of its plans are cheaper than the
  teacher's (it is not just copying).
- **Reading.** Missing exact information is the main cause: one crude, chain-only,
  volume-blind number takes s1 greedy success from 25% to 67.5%, more than PPO ever reached
  (42%, E-10). It is not the whole fix: first-mix agreement is still 45% and success is below
  the 90% bar; the feature ignores volumes and blends, and many candidates tie. Next: a
  feature that is volume-aware and covers blends (mixtures), e.g. the construction's own
  remaining mixes or cost from each candidate's result.

### E-17 (2026-10-25) Teacher study: the construction vs a sharing pass, stages 1–5
Branch `claude/teacher-study-aa8c19`, SPEC 0.24, `scripts/teacher_study.py`,
`src/meda/solvers/baselines/share.py`, records `notes/teacher_study/records_share.jsonl`. Mac,
one core, `nice -n 19`, 4 min 50 s for all 352 problems. Costs 64/8/1 (geometric). Every plan
replayed through the engine and its DAG validated. Ratio = plan cost ÷ reference.
- **Evidence and references.** `e2`: the 65 E-5 problems with a proved optimum, regenerated
  from their seeds (all 65 match the record; the construction's costs equal the recorded ones).
  `e3`: the 45 E-6 medium problems, against max(bound exact search proved there, today's 0.21
  root bound). `val`: `val_s1_v2` … `val_s5_v2`, `heldout_combo_v2`, against their stored
  root bound. E2 cells map to stages as in SPEC 8.3 (tight storage → s5; dilution many-target
  multi-demand tight, and dilution gradients with tight storage → held-out cell).
- **Teacher = `cheapest_construction`; share = the sharing pass (no lookahead).** Median / p90
  / max of the ratio:

  | Set | Stage | n | Reference | teacher | share |
  |---|---|---|---|---|---|
  | e2 | s1 | 10 | optimum | 1.00 / 1.00 / 1.01 | 1.00 / 1.00 / 1.00 |
  | e2 | s2 | 2 | optimum | 1.00 / 1.00 / 1.00 | 1.00 / 1.00 / 1.00 |
  | e2 | s3 | 7 | optimum | 1.00 / 1.20 / 1.38 | 1.00 / 1.00 / 1.01 |
  | e2 | s4 | 6 | optimum | 1.34 / 1.52 / 1.52 | 1.00 / 1.00 / 1.00 |
  | e2 | s5 | 33 | optimum | 1.00 / 1.88 / 2.49 | 1.00 / 1.01 / 1.89 |
  | e2 | held-out | 7 | optimum | 1.34 / 2.11 / 2.35 | 1.01 / 1.33 / 1.34 |
  | e3 | s4 | 24 | bound | 1.80 / 2.47 / 3.12 | 1.02 / 1.18 / 1.25 |
  | e3 | s5 | 5 | bound | 2.27 / 3.33 / 3.62 | 1.82 / 2.01 / 2.03 |
  | e3 | held-out | 16 | bound | 1.44 / 2.04 / 2.26 | 1.05 / 1.17 / 1.21 |
  | val_s1_v2 | s1 | 40 | bound | 1.52 / 3.45 / 7.96 | 1.17 / 3.12 / 7.09 |
  | val_s2_v2 | s2 | 40 | bound | 3.06 / 6.00 / 8.61 | 2.94 / 6.00 / 8.61 |
  | val_s3_v2 | s3 | 40 | bound | 2.32 / 6.24 / 9.17 | 1.25 / 2.99 / 8.90 |
  | val_s4_v2 | s4 | 40 | bound | 3.05 / 6.41 / 9.25 | 1.46 / 3.46 / 5.96 |
  | val_s5_v2 | s5 | 42 | bound | 2.60 / 4.60 / 7.48 | 1.78 / 3.58 / 7.48 |
  | heldout_combo_v2 | held-out | 40 | bound | 1.97 / 3.01 / 3.99 | 1.09 / 1.35 / 2.19 |

  Per cell (median / p90 / max; teacher → share): e3 dil-many-dN-loose 2.03/2.53/2.61 →
  1.04/1.12/1.20; dil-many-dN-tight 1.54/1.99/2.26 → 1.10/1.19/1.21; dilgrad-many-dN-loose
  1.42/1.93/2.29 → 1.01/1.02/1.02; dilgrad-many-dN-tight 1.33/2.04/2.04 → 1.01/1.08/1.09;
  mix-many-dN-loose 1.89/2.62/3.12 → 1.12/1.21/1.25; mix-many-dN-tight (5) 2.27/3.33/3.62 →
  1.82/2.01/2.03. val_s3 dilution-one-multi 1.83/4.96/8.88 → 1.02/1.74/3.12; mixture-one-multi
  2.62/6.47/9.17 → 1.64/3.41/8.90. val_s4 dilution-many-multi 1.85/2.43/2.50 → 1.04/1.12/1.18;
  dilution-many-single 2.60/4.74/9.25 → 1.31/2.04/2.69; mixture-many-multi 3.91/6.72/7.12 →
  1.65/2.93/3.51; mixture-many-single 4.57/5.81/7.51 → 3.26/5.66/5.96. val_s5 (6 each)
  dilution-many-single 2.53 → 1.45; dilution-one-multi 1.37 → 1.05; dilution-one-single
  1.87 → 1.56; mixture-many-multi 2.94 → 1.94; mixture-many-single 3.81 → 3.58;
  mixture-one-multi 2.40 → 1.73; mixture-one-single 2.40 → 2.40 (medians).
- **At the proved optimum (e2):** teacher 34 of 65, share 51 of 65 (s1 9→10, s3 4→6, s4 2→6,
  s5 16→24, held-out 1→3). Share is cheaper than the teacher on 237 of 352 problems and
  dearer on none (it falls back to the construction). Sample units drawn, teacher → share:
  `val_s3` 423 → 256, `val_s4` 1,826 → 1,008, `val_s5` 839 → 565, held-out 2,533 → 1,455,
  e3 1,692 → 1,114.
- **How to read the bound ratios.** On stage 1, where the teacher is at the optimum on 9 of
  10 small problems, its ratio to the `val_s1_v2` bound is still 1.52 (median), and on stage 2
  3.06. So a val-set ratio is mostly the bound's looseness on single targets and mixtures, not
  plan excess; it certifies the plan only where it is near 1 (the e3 dilution cells, `val_s4`
  dilution-many-multi, held-out). Mixture many-target cells stay uncertified: share at 1.65–3.6
  × the bound there could be near-optimal or not.
- **Share also beats the teacher on stage 1:** cheaper on 21 of 40 `val_s1_v2` problems
  (median ratio 1.52 → 1.17), mostly by the dilute-up chain (a cheaper single-target recipe the
  tree misses). Stage 2: 5 of 40.
- **Remaining misses (e2, worked by hand):** the optimum builds two intermediates ahead and
  blends them (1/4 at volume 3 on M = 3 via 3/8 = blend of 1/2 and 1/4: 72 vs 136); the
  greedy prices one delivery group at a time and does not find these. Rollout lookahead (k = 3)
  improved 4 of the 14 misses on e2 by 1–2 cost units (2 of them to the optimum), and costs ~15× the time (up to 48 s
  per medium problem), so it was not run on the full sets here (workstation command below).
- **Speed.** Share: median 0.04 s, p90 2.6 s, max 34 s per problem (one core); teacher: ms.
- **Task 3 (published sharing methods) not done.** The repo holds only one-line summaries
  (`notes/track2/e1_coverage.md`): MTWM needs an SMT solver (a new dependency) and its local
  PDF; e-MRCM's PDF is not on disk; WSPM incremental (sequence, leftovers reused) is close to
  what the sharing pass does. Writing them from memory would be the unchecked approximation
  SPEC 10.4 forbids, so none is claimed.
- **Recommendation (for the user to decide, D-31 (5)–(6)).** The sharing pass is
  expert-quality where it can be checked: at the proved optimum on every s3/s4 small problem
  but one, s5 median 1.00 (p90 1.01), held-out 1.01, and within 1–10% (median) of the bound on the
  medium dilution and held-out problems. Use it as the teacher on stages 3–5 for **dilution**
  cells and the held-out cell's skills; on **mixture many-target** cells it halves the
  teacher's excess but its quality is not certified (bound too loose), so either imitate it
  there with the imitation term fading as planned, or leave those cells to RL. It is also a
  better stage-1 teacher than the construction (cheaper on 21/40); the copy test on stages
  1–2 could switch to it at no cost in coverage. The construction should not be imitated on
  stages 3–5 (median 1.8–3.1× the bound on the medium/val many-target sets).
- **Workstation follow-up (not run):**
  `OMP_NUM_THREADS=1 uv run python scripts/teacher_study.py which=e2,e3,val
  producers=teacher,share,share_look3 workers=16 out=notes/teacher_study/records_look3.jsonl`

### E-18 (2026-10-25) "Finish from here": the sharing pass started mid-episode (probe)
Scratch probe `finish_probe.py` (session scratchpad), commit `b650384` code. The sharing pass
(`share._run`, no lookahead, all 4 target orders) started from a mid-episode state with every
droplet on chip in its pool, on a relaxed step limit. States: every state with droplets on chip
along the teacher's plan, first 15 problems of each val set. Mac, one core, `nice -n 19`.

| Set | states | ms per call median / p90 / max | no plan | finish-from-here ÷ current penalty, median |
|---|---|---|---|---|
| val_s1_v2 | 68 | 3.5 / 61 / 550 | 0 | 0.04 |
| val_s2_v2 | 185 | 5.0 / 119 / 443 | 0 | 0.13 |
| val_s3_v2 | 229 | 3.8 / 227 / 1,828 | 0 | 0.29 |
| val_s4_v2 | 743 | 1,833 / 5,211 / 12,019 | 0 | 0.54 |
| val_s5_v2 | 237 | 8.8 / 424 / 6,704 | 0 | 0.42 |

- **The current penalty ignores progress.** Along the teacher's own plan, the cost the pass
  needs to finish is a median 4% (s1) to 54% (s4) of `ConstructiveCostToFinish` (before the
  ×2 factor of D-26), which wastes every droplet on chip and rebuilds. So an episode truncated
  one mix short of the target is charged as if nothing had been built: building an
  intermediate earns no credit at truncation. This is a likely contributor to E-10/E-11 and
  is fixable: the pass always found a plan (0 failures in 1,462 states).
- **Speed.** Once per failed episode is affordable on s1–s3 and s5 (median 4–9 ms) and
  tolerable on s4 (1.8 s median, many targets). Once per candidate (200–800 per step) is not:
  even 4 ms × 200 is 0.8 s per step against ~200 steps/s in training. A per-candidate feature
  must use the pass's cheap parts (direct, one mix, chain), not the whole pass.

### E-19 (2026-10-26) Cheap moves plus a finish beat the teacher on some stage 1 problems
Scratch probes `penalty_factor.py`, `penalty_debug.py` (session scratchpad), SPEC 0.25 code. The
"cheapest failure" of `tests/envs/test_failure_penalty.py` (every step the cheapest legal move
that does not finish, until truncation), then the planner's finish from the final state
(`finish_from`), is a real plan. Its cost against the sharing pass's plan: **val_s1_v2 cheaper on
7/40** (ratio min 0.35, median 1.07); val_s2_v2 0/40 (min 1.04); val_s3_v2 3/40 (min 0.59).
Example `val_s1-008` (101/240 S, M = 6, step limit 22): 89 for the cheap moves (ending with 1/4
S ×2, 23/96 ×1, 25/96 ×1 on chip) + 79 for a 4-move finish = **168**, against 475 for the
sharing pass, the construction and the best known.
- **Reading.** (1) The teacher is not an expert on every stage 1 problem: on large
  denominators it spends sample units a cheap, buffer-heavy start avoids (sample costs 64 vs
  buffer 1). E-5's "~0% headroom on stage 1" holds on small provable problems, not on all of
  `val_s1_v2` (bound ratio median 1.52, E-17). Imitation must stay a warm start that RL can
  leave (D-31 (2)). (2) Under the `share` penalty a failed episode can cost less than the best
  known plan (247 = 89 + 2 × 79 here), so the D-26 test bar no longer applies; replaced per
  D-32. (3) A better best-known for `val_s1_v2` is cheap to get (greedy start + finish); the
  answer keys are unchanged for now.

### E-20 (2026-10-26) Copy test with hints and the sharing teacher: passes the bar
Commits `333de6b`, `b2bb9ea` (SPEC 0.25), `scripts/imitation_check.py problems=1000 epochs=20
eval_every=5`, seed 0, Mac one core, 17 min. Same setup as E-15 run 2 except: hint columns
(D_cand 27), teacher = sharing pass (mean gap to best known −13.3% on val_s1, −2.9% on val_s2),
`share` penalty in the eval env. "Mix agreement" = agreement on the teacher's Mix steps only.

| | E-15 run 2 (no hints, construction teacher) | E-20 |
|---|---|---|
| train agreement | 0.64 (30 epochs) | **0.93** |
| s1 mix agreement (unseen) | ~0.20–0.23 first mix (E-16) | 0.63–0.74 (all mixes) |
| s2 mix agreement (unseen) | ~0.06–0.10 first mix | **0.87–0.92** |
| s1 greedy success | 22–35% | 70% / 70% / 67.5% / **95%** (epochs 5/10/15/20) |
| s2 greedy success | 0–2.5% | 37.5% / 67.5% / 67.5% / **72.5%** |
| plans cheaper than best known (s1 / s2, epoch 20) | 0 / 0–1 | **12 / 6** |

Mean gap on successes at epoch 20: s1 +4.6%, s2 −3.4% (to the old best known).
- **Reading.** The hints were the missing information: on mixtures, agreement went from
  about 1 in 10 to 9 in 10 and greedy success from 0 to 72.5%. The D-31 pass bar (s1 ≥ 90%, s2
  well above 0) is met at epoch 20. **Caveat:** s1 success sat at 67.5–70% for epochs 5–15 and
  jumped to 95% only at the last grading, with s1 mix agreement flat at ~0.70; on 40 problems
  one grading is noisy, so s1 is borderline, not settled. The s1 misses are likely where the
  teacher itself is not unique or not best (E-19). Next: warm start + PPO (step 2), which can
  also fix what imitation leaves.
- *s1 result corrected by E-21: the 95% was noise; s1 copies at about 70–77%.*

### E-21 (2026-10-26) Copy test repeated on the workstation, and the teacher ablation
Same script and seed as E-20 (`scripts/imitation_check.py problems=1000 epochs=20 eval_every=5`),
commit `bc93a7b`. (a) **Workstation rerun** (CPU, 8 threads, `outputs/imitation/s12`, the network
step 2 starts from): identical demonstrations (teacher gaps −0.133 / −0.029, 9,297 steps). (b)
**Ablation** on the Mac (background priority): hints on, **old teacher** (`teacher=construction`).

| Greedy success, epochs 5 / 10 / 15 / 20 | s1 | s2 | cheaper than best known at 20 (s1 / s2) |
|---|---|---|---|
| E-20, Mac, sharing teacher | 70 / 70 / 67.5 / 95% | 37.5 / 67.5 / 67.5 / 72.5% | 12 / 6 |
| (a) workstation, sharing teacher | 72.5 / 67.5 / 70 / 77.5% | 55 / 60 / 67.5 / 65% | 12 / 6 |
| (b) Mac, construction teacher | 70 / 65 / 70 / 72.5% | 47.5 / 70 / 72.5 / 62.5% | 8 / 3 |
| E-15, no hints, construction teacher | 22–35% | 0–2.5% | 0 / ≤1 |

Mix agreement at epoch 20: (a) s1 0.71, s2 0.90; (b) s1 0.66, s2 0.90. Mean gap on successes at
epoch 20 (to the old best known): (b) s1 −3.3%, s2 +1.9%.
- **Reading.** (1) **The hint columns are the cause** of the jump from E-15: with the old
  teacher they give the same success as with the sharing teacher, on both stages. The sharing
  teacher's contribution is cost: more plans cheaper than best known (12 / 6 vs 8 / 3). (2)
  **E-20's 95% on s1 was noise** in one 40-problem grading: imitation alone copies s1 at about
  70–77% and s2 at about 60–70%. So the D-31 copy bar is met on s2 and missed on s1 (≥ 90%);
  s1 still rose from 25–35%. Step 2 (PPO from the network of (a)) is where s1 must clear 90%;
  the remaining s1 misses are likely where the teacher's move is not unique or not best (E-19).

### E-22 (2026-10-26) Warm start + PPO on stages 1–2 (step 2)
Commit `bc93a7b` (SPEC 0.26), workstation, `outputs/warm/mixed_s0`, seed 0. PPO started from the
E-21 (a) imitation network (`train.init_from=outputs/imitation/s12/imitation.pt`). Imitation term
on sharing-pass demonstrations, weight 1.0 falling linearly to 0 at 1M steps (update ~245), then
pure PPO to 3M (732 updates). `regime=mixed regime.num_stages=2` (s1 and s2 half and half, no
promotions; not a regime test), `costs=geometric`, `reward_scale=0.01`, share penalty and hint
columns (SPEC 0.25), `workers` collector, ~700 env steps/s, about 75 min. Greedy grading every 25
updates on `val_s1_v2` and `val_s2_v2` (40 unseen problems each); final grading adds
`heldout_combo_v2`.

| Update (imitation weight) | 25 (0.9) | 100 (0.59) | 200 (0.18) | 250 (0) | 325 | 375 | 425 | 550 | 725 |
|---|---|---|---|---|---|---|---|---|---|
| s1 success | 97.5 | 97.5 | 97.5 | 100 | 100 | 100 | 100 | 100 | 100 |
| s1 gap to old best known | −1.8 | +1.4 | −5.2 | −7.3 | −10.1 | −10.3 | −9.3 | −16.8 | −16.0 |
| s2 success | 82.5 | 85 | 92.5 | **97.5** | 92.5 | 77.5 | 72.5 | 75 | 85 |
| s2 gap to old best known | +0.9 | +1.2 | +1.2 | +10.7 | +8.5 | +3.6 | +0.2 | +2.5 | +0.1 |

(%; gap is the mean over solved problems. The teacher's gap is −13.3% on s1 and −2.9% on s2.)
Training at the end: success 91–94%, approx_kl 0.001–0.005 (learning rate annealed to 0),
explained variance 0.88–0.89.

**Final grading, agent vs the sharing-pass teacher per problem** (`outputs/vs_teacher.py`, not
committed; checked on the Mac by grading the teacher against itself):

| Set | Solved | Agent / teacher on solved: cheaper / equal / dearer | median, mean, min, max |
|---|---|---|---|
| s1 | 40/40 | 7 / 19 / 14 | 1.000, 0.978, 0.36, 1.53 |
| s2 | 34/40 | 8 / 12 / 14 | 1.000, 1.035, 0.81, 1.58 |
| held-out cell | 20/40 (50%) | not compared | |

- Largest wins: s1 `val_s1-024` 77 vs 213, `-003` 209 vs 402, `-007` 77 vs 144; s2 `val_s2-007`
  373 vs 458, `-039` 293 vs 358. Largest losses: s1 `-018` 205 vs 134, `-016` 455 vs 325; s2
  `-023` 459 vs 291, `-028` 845 vs 635.
- s2 failures (status `failed`): `val_s2-000, -004, -016, -019, -021, -037`, teacher costs
  481–564, all among the set's dearest problems.
- **Reading.** (1) **The warm start unblocks learning.** Pilot 2 reached s1 38–42% and s2 0% in
  10M steps; this run is at s1 97.5% and s2 82.5% from the first grading (100k steps). (2) **The
  D-31 pass bar is met:** s1 ≥ 90%, s2 well above 0, and plans cheaper than the teacher on
  some problems (7 s1, 8 s2). (3) **On s1, RL improved on the teacher after imitation ended:**
  the mean gap moved from about −5% to −15/−17%, past the teacher's −13.3%. It is a mean, not
  a uniform win: the agent is dearer on 14 of 40 problems and much cheaper on a few. (4) **s2
  decays after imitation ends.** It peaks at 97.5% at update 250, just after the weight reaches
  0, and stays at 92.5% to update 325; from update 375 it falls to 72.5–85% and does not
  recover. Its cost does not improve on the teacher (mean ratio 1.035 on solved problems). One
  40-problem grading has a noise of about ±6 points, but the drop is sustained over 15
  gradings. (5) **Zero-shot held-out cell: 50%** (dilution + many targets + demand + storage,
  none of stages 3–5 seen in training). Its cost is not yet compared.
- **Open (not tested):** why s2 decays. Candidate causes: s1 dominating the shared update once
  imitation stops; cost-cutting moves learned on s1 that break long mixture plans near the step
  limit; ordinary PPO drift without a teacher anchor. Status of the 6 failures (truncated vs
  dead end) and their plans not yet inspected. One seed only.

### E-23 (2026-10-26) Teacher against optima and bounds on s1/s2; what the s2 failures share
Scratch probes `teacher_vs_bound.py`, `train_lengths.py` (session scratchpad), SPEC 0.26, Mac
background priority. Teacher = the sharing pass.
- **Stage 1 references.** `val_s1_v2` has 3 proved optima and 37 intervals [lower bound, best
  known]. The teacher equals the optimum on all 3. Against the bound: teacher median 1.17, p90
  3.12, max 7.09, within 1.10× on 16/40 (old best known: 1.52 / 3.45 / 7.96, 13/40). Since the
  teacher is optimal on all 10 small proved s1 problems (E-17), most of the excess over the
  bound is bound looseness; but E-19 and E-22 found plans 2–3× cheaper than the teacher on a
  few large-denominator problems (`val_s1-024` 77 vs 213), so the teacher is far from optimal
  there. Best known is our own construction; no published heuristic is implemented yet (tier
  3).
- **Stage 2 references.** No proved optima; teacher / bound median 2.94, within 1.10× on 3/40.
- **The 6 s2 failures of E-22 are the 6 longest teacher plans:** 41, 38, 24, 24, 24, 23 steps
  (step limits 82, 76, 48, 48, 48, 46); every other s2 problem's teacher plan is ≤ 21 steps
  (median 10). They span F = 4–6 and M = 3–8, so length, not fluid count, is what they share.
- **Training exposure:** of 300 freshly drawn s2 training problems, teacher plans have median
  12 steps, p90 24, max 49; 10.7% are ≥ 23 steps (about 5% of all training episodes, since s2
  is half the mix).
- **Reading (not yet tested):** the decay after imitation ends hits long-horizon mixtures.
  Whether they are truncated or reach dead ends, and where the agent leaves the teacher's
  route, needs the final agent (`agent.pt`) replayed on them.

### E-24 (2026-10-26) Replaying the stage 2 failures: hoarding and dithering
Scratch probe `s2_replay.py` (session scratchpad; run on the workstation as
`outputs/s2_replay.py`, traces in `outputs/s2_replay.json`), SPEC 0.26. Greedy replay on
`val_s2_v2`; at every step the agent's move and probability, the probability it gave the
teacher's move while still on the teacher's route, and the projected total = cost so far + the
sharing pass's finish from the state reached. Teacher move counts from `share_plan`.
- **Warm-start agent (E-22, final weights): 34/40, the same 6 failures as E-22.** All 6 are
  truncated at the step limit (48–82 steps, twice the teacher's plan), with one droplet of
  demand left; none is a dead end.

  | Problem | Teacher steps | Agent moves: res-res mix / mixes with a droplet / discards | Teacher: res-res / with a droplet / discards | Droplets on chip at the end | Mean p of chosen move | Projected start → end |
  |---|---|---|---|---|---|---|
  | val_s2-000 | 24 | 7 / 41 / 0 | 3 / 12 / 8 | 25 | 0.31 | 481 → 1010 |
  | val_s2-004 | 24 | 7 / 39 / 2 | 3 / 10 / 10 | 14 | 0.46 | 505 → 842 |
  | val_s2-016 | 38 | 8 / 67 / 1 | 3 / 16 / 18 | 11 | 0.61 | 524 → 682 |
  | val_s2-019 | 23 | 8 / 38 / 0 | 5 / 10 / 7 | 19 | 0.32 | 520 → 1069 |
  | val_s2-021 | 41 | 6 / 76 / 0 | 4 / 19 / 17 | 11 | 0.64 | 564 → 696 |
  | val_s2-037 | 24 | 4 / 41 / 3 | 5 / 9 / 9 | 10 | 0.34 | 544 → 585 |

  Solved problems, for comparison: steps / limit median 0.37 (max 0.75), repeated states 0,
  mean p of the chosen move median 0.84. Both solved and failed episodes leave the teacher's
  route early (solved: median step 2, max 9; failed: step 0–2), so leaving the route is not
  the failure.
- **Step by step:** after leaving the route the agent makes mix after mix with droplets on
  chip, almost never discards, and the projected total rises by about 1 per step (one mix's
  cost; the planner's finish from the state barely moves). Its moves neither build toward the
  target nor destroy what is there; it never commits to finishing.
- **Copy-test network (E-20 weights) on the same set, for comparison:** 29/40; all 11 failures
  truncated with one droplet left, but by **looping** (mix, discard, the same state again; 7–35
  repeated states, about half the moves discards). RL replaced looping with hoarding.
- **A reward inconsistency found while reading this:** on success the engine charges every
  droplet left on chip as waste (`rules.apply`), but the `share` failure penalty uses
  `finish_from`, which does not charge the leftovers of its completion (SPEC 10.3). So in a
  failed episode the droplets the agent hoarded are never charged, while discarding them costs
  waste at once. Waste weighs only 1 per unit, so the size of the effect is unknown; it points
  the same way as the behaviour seen.
- **Reading (hypotheses, not tested):** (1) RL learned to avoid discards (they cost now);
  on long mixtures droplets pile up, the candidate set grows with every droplet pair, and the
  policy spreads its probability thin (mean p 0.31–0.64) and drifts; (2) the uncharged
  leftovers in the failure penalty make hoarding free in episodes that fail; (3) without the
  imitation anchor nothing pulls the policy back toward the teacher's discard-heavy plans, which
  fits s2 holding up to update 325 and decaying after.

---

## 5. Failures, bugs and lessons

Each one changed behaviour or numbers. The recurring theme: **RL fails silently; only an
independent reference (oracle, optimum, bound, validator) catches it.**

- **F-1 (2026-09-06) Agent "beat the oracle" (−2.09 vs optimal −4).** A dead-end state ended the
  episode cheaply, so the agent learned to fill storage and "commit suicide" to stop the clock.
  Fix: price dead ends in the referee. Lesson: an impossible score is the loudest alarm. **Same
  family as F-11.**
- **F-2 (2026-10-13) Q initialised at 0 is optimistic** when all true values are negative, so the
  greedy policy chases unvisited actions. Fix: an informed prior.
- **F-3 (2026-10-13) State aliasing over the step count:** keys without `step` merged states with
  different remaining budgets; `quarter-m4` failed until `include_step`.
- **F-4 (2026-10-13) α = 1/N averages drifting bootstrap targets:** Q(1:1) converged to −16.02
  where the truth is −15.0 after 1,248 visits. Constant α fixed 3 instances at once.
- **F-5 (2026-10-15) Benchmark leakage, twice.** The curriculum never passed `exclude` (about 1 in
  4 stage-1 draws would have been validation problems); and `instance_hash` included
  `max_denominator`, so tiny-set problems were not recognised in stage 1. Fix: wired exclusion,
  chip-agnostic `problem_hash`, a test that fails if unwired.
- **F-6 (2026-10-20) PyTorch's `TransformerEncoderLayer` gave different outputs in eval and train
  mode once per-head additive biases were added** (0.09 apart, larger than the spread between
  candidates), so PPO's log-probabilities would be inconsistent. Fix: a hand-written attention
  layer.
- **F-7 (2026-10-20) Hidden capability ceilings set by our own code:** droplet volume ≤ M/2 had no
  physical reason; it was where the constructions stopped. A full audit followed (user: "I don't
  want to get stuck on a capability that is easily improvable").
- **F-8 (2026-10-22) Logged `sps` left out the PPO update,** overstating speed by ~2×. Lesson:
  time every phase.
- **F-9 (2026-10-22) Padding dominated memory and speed** (16,384-row observations copied ×32 per
  step; minibatches padded to the widest). See E-12.
- **F-10 (2026-10-22) Small infrastructure bugs found on real hardware:** `device: auto` gave
  `cuda` with no index for the memory cap; resuming on GPU loaded RNG state onto the GPU;
  `evaluate_cli` crashed with the `tree` producer (since 0.12).
- **F-11 (2026-10-23) The give-up pathology (main pilot-1 failure).** The failure penalty was the
  construction's cost from scratch, so a cheap deliberate failure cost only ×1.33 (s1, min 1.10),
  ×1.09 (s2, min 1.03), ×1.16 (s3, min 1.05) of the best known plan. Any finished plan more than
  9% (s2), 16% (s3) or 33% (s1) worse than best known earned less than giving up. The agent
  finished only problems where it could beat the construction: low success, negative gaps, 0%
  mixtures. SPEC 5.8's "an unfinished plan is never cheaper than a finished one" held only
  against finishing *with the construction*. Fix D-26. **Paper-worthy general lesson:** a
  construction-based cost-to-go penalty must exceed the cost of any plausible finished plan, not
  just the construction's.
- **F-12 (2026-10-23) Step limits ignored the fluid count:** 13 of 40 `val_s2` problems allowed
  exactly the witness plan (zero slack; median 1.33×). Fix D-27.
- **F-13 (2026-10-23) Forced-promotion cap 0.3 × 5 stages > 1:** stage 5 would never train. Fix
  D-28.
- **F-14 (2026-10-22/23) The lower bound was inadmissible on tolerant targets** (51% ± 2% at
  volume 2: bound 66.26 vs optimum 66), so exact search's "proved optima" on tolerant targets
  could also be wrong. Fix in 0.21 (E-8).
- **F-15 (2026-10-15/16) Overclaims caught:** exact search's docstring promised branch-and-bound
  the code did not do; 11/16 was "proved" only at a matched accuracy, not on the real instance.
- **Ops lessons (not for the paper):** Python stdout buffering hid a 10 h background check;
  pytest thrashed at 1,144% CPU without `OMP_NUM_THREADS=1`; parallel checks on the Mac
  overheated it (heavy jobs now run only on the workstation).

---

## 6. Literature notes

- **L-1 (2026-10-16) Training-regime review** (`notes/training_regime_research.md`, ~30 sources).
  Sequential fine-tuning forgets (Lisicki 2020 on neural CO size curricula; CLEAR; Wołczyk
  2024); rehearsal ~50/50 fixes most of it (CLEAR); prioritized level replay beats uniform on
  held-out Procgen levels; training on mixed sizes holds the trained range but does not
  extrapolate (Joshi 2022); Procgen agents overfit at a few hundred levels. No RL work on biochip
  sample preparation was found; RL on these chips is about droplet routing.
- **L-2 (2026-10-19) Size generalisation:** sum aggregation and BatchNorm broke on larger TSP
  graphs; mean/max and LayerNorm held up (from the review; **verify the source** before citing).
- **L-3 (2026-10-22) Paper checks on local PDFs:** TVODA is in the FacDA paper (Kundu et al.,
  "Mixing Models as Integer Factorization", IEEE TCAD 41(3), 2022) and takes a total volume for
  one dilution target. e-MRCM is a greedy heuristic, not RL (per MTWM: Kundu & Roy, ACM TODAES
  28(6), 2023, §7); the 2026 Das review never calls it RL. WSPM incremental (Li et al., IEEE
  TBioCAS 11(6), 2017) and Fukui & Yamashita (ISOCC 2024) are multi-target with one droplet each.
  DDMT (Poddar et al., ACM TODAES 27(1), 2021) has demand and storage but only on 1:1 DMFBs.
  FloSPA-D needs ~216,000 s at accuracy level 6 (from BASELINES.md).
- **L-4 Still to read:** robust-aliquot ILP (TODAES 26(1), 2020) — main risk; Dong 2024 (JETTA
  40(1)); e-MRCM itself (Liang et al., IEEE TCAD 39(10), 2020). A paused side session
  (`paper-checks` branch) waits for these PDFs.
- **L-5 Methods to cite for D-30:** AlphaGo (supervised from expert games, then RL), AlphaGo Zero
  (search instead of demonstrations), AlphaTensor (synthetic demonstrations + search; our
  failure-penalty idea), DD-PPO (Wijmans et al. 2020) for the collector, PPG (Cobbe et al. 2021)
  and PopArt if value interference matters.

---

## 7. Figure and table ideas

1. 7/32: naive vs optimal plan as two small DAGs (E-3).
2. Sharing and reuse: the two E-5 worked examples, tree vs optimum, with sample/waste/mixes.
3. Headroom by cell (E-5 table) and the coverage map of the 16 cells (E-4).
4. Bound tightness: LB/optimum distribution, before and after 0.21 (E-6, E-8).
5. The give-up pathology: failure-cost multiples per stage before/after the fix, and pilot-1 vs
   pilot-2 s1 success curves (F-11, E-9, E-10).
6. Random-play success vs branching factor and plan length (E-11) as the motivation for the warm
   start, and then (once run) learning curves with vs without the warm start.
7. Architecture diagram: fluid tokens → composition embedding → relation-biased attention →
   per-candidate scorer; the c1/c4 example (two candidates that look identical locally; one uses
   up a droplet another target needs). The c1/c4 costs in chats are **illustrative**.
8. Throughput table (E-12) for the appendix.
9. Later: gap-vs-axis curves (SPEC 1.5), held-out cell result, baseline comparisons per cell.

---

## 8. Threats to validity and open items

- Two unread papers could narrow the claim (L-4).
- Frontier "best known" is still our own construction, which never shares; a sharing search
  baseline would make the gaps more convincing (D-18 options). *Partly addressed by E-17: the sharing pass tightens
  best known on many-target problems (not yet written into the answer keys).*
- The bound is still loose on stages 4–5 (median 0.33–0.39 of best known); a forced-waste term
  is the noted next step.
- Limited storage may rarely bind (E-7); measure on trained agents before claiming the storage
  gap.
- Negative gaps averaged over solved problems only are biased toward easy problems (F-11 showed
  how badly). Always report success rate next to gap.
- Pilots used 2 seeds; SPEC 12.3 asks for ≥ 3 for reported results.
- Cost weights are fixed (64/8/1); baselines optimise other metrics, so report each paper's own
  metric too.
- No agent has yet been evaluated on mixtures successfully; the generalisation and frontier
  claims are untested.
- Supervisor questions open: the platform's real M, mixer footprint, reservoir count, real cost
  weights, whether the router can serve as a cost oracle.

---

## 9. Where things live

- Design: `SPEC.md` (changelog at the top, 0.2–0.22). Project thesis: `PROJECT_CONTEXT.md`.
- Experiments: `notes/track2/` (`e1_coverage.md`, `results.md`, raw `.jsonl`),
  `notes/training_regime_research.md`, `scripts/headroom.py`, `scripts/check_bound.py`,
  `scripts/regime_report.py`.
- Pilot runs (workstation): `outputs/pilot_old_penalty/` (pilot 1), `outputs/pilot/` (pilot 2).
- Architecture explainer with a 27-entry decision log: "MEDA Agent Anatomy" artifact,
  https://claude.ai/artifact/A16FRKXRUpFSrXYU5JBv8U (private).
- Chat transcripts: `~/.claude/projects/-Users-adityatully-projects-btp/*.jsonl`.
