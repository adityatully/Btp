# Justification plan for method-v3 (2026-11-18)

The paper must give a reason for every choice in the method, down to the smallest. This file
lists each one (Part 1), with the evidence we have, and groups the missing evidence into a
small number of runs (Part 2). Registered 2026-11-18 as D-98 (paper log), plan step 6.9 and
Q-158 to Q-160. Runner for the search sweeps: `scripts/ablation.py`; subset
`notes/ablation/subset_v1.json`.

**What counts as a justification.** One of four kinds, named in the tables:
- **Proof / construction:** true by how the code is built (say where).
- **Ablation:** measured against at least one alternative on the dev (val) sets.
- **Sweep:** several values measured, and the chosen one sits on a plateau (its neighbours are
  level within run-to-run noise) or at the knee of a cost-vs-time curve.
- **Argument / citation:** a definition of the task, a physical fact, or a published convention.

No parameter can be proved optimal at full size (the optimum is unknown there). The realistic
claim for each number is "the result is flat around it" or "it is the knee".

**Status:** ✅ justified · 🟡 partly (one alternative, wrong setting, or in hindsight) · ❌ never
measured. **Run:** the Part 2 run that closes it.

Method-v3 (SPEC 0.44, D-79, D-83): a plain-pass plan first; then a PUCT search over the env's
legal moves, flat prior, 256 simulations per move, most-visited move played, 900 s CPU; every
leaf is valued by a planner finish (greedy MRCM pass, switching to the tiered finisher when
MRCM is slow); the answer is the cheapest whole plan seen.

---

## Part 1. Every decision

### 1. The search (S)

| ID | Decision | Value / where | Evidence today | Status | Run |
|---|---|---|---|---|---|
| S1 | Use tree search at all (vs other ways to spend the time) | PUCT MCTS | E-37: 900 s of random restarts of the planner gain 1-2%, the search 14%. Never compared with beam search, best-first or nested rollouts at equal time. | 🟡 | J4 |
| S2 | Simulations per move | 256 (`bench.METHOD`) | E-33: 64 vs 256 with the network prior only. E-36/37: 1 vs 256 with the network. E-95: more sims at the root after the pass ends, no gain. Never swept on the flat-prior search under 900 s. E-81: on s4-s5 the tree commits only 0-2 moves, so this setting may be what limits depth there. | 🟡 | J2 |
| S3 | Exploration constant | `C_PUCT = 1.5` (`search_check.py:172`) | Never varied since the first commit. | ❌ | J2 |
| S4 | Flat prior, no network | `search="uniform"` | E-33, E-36, E-39, E-51, E-54, E-82 | ✅ ablation | – |
| S5 | Leaf value from a planner finish, no value head | `planner_leaf=True` | E-35, E-40, E-43, E-45 to E-50 | ✅ ablation | – |
| S6 | Leaf value scale | `(c0 − cost)/c0`, `c0` = the root's own finish | Never compared with another scale (e.g. rank, or c0 from the first plan). The scale sets how much C_PUCT explores. | ❌ | J3 |
| S7 | Floor on the value | −1 | Never varied. Matters only for leaves over 2× the root's cost. | ❌ | J3 |
| S8 | Value of a dead end or a cut-off episode | −1 (the env's failure penalty is not used) | Never varied. | ❌ | J3 |
| S9 | Backup | mean of leaf values (`nd.total += end.value`) | The task is deterministic and single-player, and we want the minimum. A max (best-child) backup was never tried. | ❌ | J3 |
| S10 | Value of an unvisited move (first-play urgency) | the parent's value | Never varied. | ❌ | J3 |
| S11 | Root value | 0 | Only a convention; it affects nothing once the root is visited. | ✅ construction | – |
| S12 | Move played after each 256 sims | most-visited child, tree below it kept | Never compared with "the child with the cheapest plan seen". | ❌ | J3 |
| S13 | Answer = cheapest whole plan seen anywhere (leaf finishes included) | `_offer` | Anytime by construction; a plan is offered only if it fits the step limit. | ✅ construction | – |
| S14 | A leaf finish longer than the step limit is still used as the leaf's value but not offered as a plan | `leaf()` | How often this happens, and whether it misleads the tree, is not measured. | ❌ | J0 |
| S15 | Tie-breaking | candidate order (`run_task` passes no `rng`) | One run per problem; never tried other orders. | ❌ | J1 |
| S16 | Time budget | 900 s CPU | E-76 (method-v1 curve), E-101 (median 71 s to the final plan; s4 367 s), E-102 (57% of CPU after the final plan). No method-v3 cost-vs-budget curve written down. | 🟡 | J0 |
| S17 | CPU time, not wall time | `time.process_time` (D-78) | Argued (shared machine). But a faster CPU gets more leaves, so results depend on the machine, and run-to-run variance is not measured. | 🟡 | J1 |
| S18 | Deadline checked only between leaves | `_descend` | E-102: runs overshoot up to 2,442 s against 900 s. Not yet in the report as a fairness caveat. | 🟡 | J0 |
| S19 | Plain-pass plan before the root | `first_plan=True` | E-76: it removes the slow first plan. It does not change the search (by construction). | ✅ | – |
| S20 | Moves = every legal env action (mixes with any k:l, outputs, discards, partial takes); identical droplets listed once | SPEC 5.6 | Planner jumps as macro-moves were tried and dropped (E-70). Dropping Discard or partial takes from the search was never tried. | 🟡 | J3 |
| S21 | No transpositions (the same chip reached by two move orders is two nodes) | `Search` | Never measured how often states repeat. | ❌ | J3 |
| S22 | One search per problem, no restarts or portfolio | `portfolio=False` | E-75: the search alone carries every published verdict; the 4 runs matter only on s4, s5, held-out. | ✅ ablation | – |
| S23 | Leftover time not used | `grow=1` | E-95: using it gains nothing. | ✅ ablation | – |

### 2. The leaf finisher (F)

| ID | Decision | Value / where | Evidence today | Status | Run |
|---|---|---|---|---|---|
| F1 | Greedy MRCM pass as the default finisher | `share_mrcm`, D-65 | E-60, E-61: 10.0% below the new planner vs lookahead-3's 8.4%. | ✅ ablation | – |
| F2 | Switch to the tiered finisher when MRCM is slow | A3, D-75 | E-69, E-72, E-74 | ✅ ablation | – |
| F3 | Switch threshold | `SWITCH_MEAN = 20` s | E-73 swept 1, 5, 10, 20, 50 s, but offline on the whole run's average (hindsight); the real rule decides from its first leaves. E-74 ran the real rule at 20 s only. | 🟡 | J5 |
| F4 | Leaves counted before the switch can fire | `SWITCH_MIN_LEAVES = 3` | Fixed before the run, never varied. | ❌ | J5 |
| F5 | MRCM's share of time after the switch | `TIER_SHARE = 0.5` | Fixed before the run, never varied. | ❌ | J5 |
| F6 | Root's finish left out of the average | D-75 | Argued: the root's finish fills MRCM's cache and is always slow. | ✅ argument | – |
| F7 | Finisher runs on a relaxed step limit | `RELAXED_STEP_LIMIT = 100,000` | It answers "what would finishing cost". Linked to S14. | 🟡 | J0 |
| F8 | Finished plan's cost leaves out the final leftovers ("as for a finished plan") | `finish_from` docstring | E-103: the reported cost equals the engine's on 210 of 210 finishes (from the start and after random moves); regression test. | ✅ check | – |

### 3. The planner (sharing pass, `share.py`): the finisher and every planner baseline

| ID | Decision | Value / where | Evidence today | Status | Run |
|---|---|---|---|---|---|
| P1 | Several target orders, cheapest plan kept | 4 (`ORDER_RULES`) | E-28 (32 random orders, small gains), E-37 (random restarts, 1-2%). D-96 / Q-156 checks whether 1-2 suffice (coe9 run pending). | 🟡 running | J6, J7 |
| P2 | Which orders: given, dearest first, cheapest first, nearest neighbour from the dearest; "dear" = fluid cost per unit; "near" = L∞ distance | `named_orders` | Never argued or compared with other rules (e.g. by demand, by denominator). D-96 measures which rule wins. | ❌ | J6 |
| P3 | One target at a time (all its deliveries before the next) | `_run` | Never compared with interleaving delivery groups across targets. | ❌ | J6 |
| P4 | Recipe set: direct, one mix, chain, split tree, MRCM tree | `_candidates` | E-17 (chain + tree beat the construction), E-57/58 (MRCM tree). Dropping each recipe from the final planner never run. | 🟡 | J6 |
| P5 | Chain depth | `CHAIN_DEPTH = 3` | E-38: 2 vs 3 on the planner alone (3 is 4-18% cheaper, 5-18× slower). Depth 1 and 4 never tried, so no plateau shown. Inside the search, depth 2 would give 10-18× more leaves at the same time; never tested (parked, D-44). | 🟡 | J6, J7 |
| P6 | Blend partners per chain step | `BLEND_PARTNERS = 4`, nearest by L∞ | Set for speed only (code note). No cost measured. | ❌ | J6 |
| P7 | A chain's last input must be pure or already in the pool | `_chain` | A pruning rule, never ablated. | ❌ | J6 |
| P8 | Greedy rule | least marginal cost per delivery; ties: more deliveries, then fewer actions | Never compared with total cost or cost per unit volume. | ❌ | J6 |
| P9 | Deliveries one droplet serves | tries the most and 1 only; the one-mix recipe keeps only the largest count that works | Never tried the counts in between. | ❌ | J6 |
| P10 | One-mix recipe ranked by fluid drawn | `_one_mix` | Mixes and waste counted later in the marginal cost; never compared. | ❌ | J6 |
| P11 | Eviction when storage is full | idle droplet furthest (L∞) from every open target, then the smallest | Never compared with another rule. | ❌ | J6 |
| P12 | Lookahead k (each group's k cheapest recipes rolled out greedily) | 3, in the baselines `share_look3`, `share_mrcm_look3` (not in method-v3) | E-17: k = 3 fixed 4 of 14 small misses at ~15× time. Only k = 1 also measured. These are the baselines we beat, so a reviewer will ask if k = 3 is a strong enough baseline. | 🟡 | J6 |
| P13 | Tolerant target aims at the coarsest reachable composition within tolerance | `tree.aim` (FacDA's transform) | Argued (shallower recipe); never compared with the nearest composition. | 🟡 | J6 |
| P14 | Split tree: dilute-up or blend steps per recipe | `MAX_EXTRA_STEPS = 4` | Code note: 2 builds every dilution; 4 "leaves room". Never shown that 4 vs 6 changes nothing. | 🟡 | J6 |
| P15 | Split tree search budget | `tree.SEARCH_BUDGET = 20,000` | For speed and determinism; how often it binds is not measured. | ❌ | J0, J6 |
| P16 | MRCM recipe: search budget, aim, pool droplets offered | `mrcm.SEARCH_BUDGET = 2,000,000`; aim 1 or the most deliveries; the largest pool droplet per composition | Inherited from the MRCM baseline (D-61); how often the budget binds is not measured. | 🟡 | J0, J6 |
| P17 | Never dearer than `cheapest_construction` | `share_plan` | By construction. | ✅ | – |

### 4. The engine and the problem (E): fixed before the method, but the results depend on them

| ID | Decision | Value / where | Evidence today | Status | Run |
|---|---|---|---|---|---|
| E1 | Two-input k:l mixes, k + l ≤ M | SPEC 1.3 (user, 2026-10-20) | Task definition. n-input published mixes are replayed as chains (D-29), which costs them mixes; say so. | ✅ argument | – |
| E2 | Cost weights: sample 64, reagent 8, buffer 1, mix 1, waste 1 | `costs/geometric.yaml` | Fluid weights after e-MRCM (cited). Mix = 1 and waste = 1 are not argued. E-52 (small problems) and B1's three goals show the verdicts under other goals; no sweep of the weights themselves. | 🟡 | J8 |
| E3 | Step limit `depth_support`, floor 2× the witness | SPEC 8.1, 0.22 | Chosen for the RL agent (give-up failure). It caps how deep any plan, ours included, can go; never checked whether it cuts off cheaper plans. | 🟡 | J8 |
| E4 | No merging of two droplets of the same composition | `allow_same_composition_merge: false` | Default, never argued. | ❌ argument | – |
| E5 | Partial takes allowed; min mix and droplet volumes | chip config | Platform assumptions (SPEC 15). Cite MEDA papers. | 🟡 citation | – |
| E6 | Every droplet on chip counts toward storage | SPEC 5.6 | Task definition. | ✅ argument | – |

### 5. Evaluation (V)

| ID | Decision | Value / where | Evidence today | Status | Run |
|---|---|---|---|---|---|
| V1 | Set sizes | 40 per val set, 100 per test set | No power check. | 🟡 | J9 |
| V2 | Verdict rule | 95% bootstrap CI of the mean ratio, ≤ 10% losses, ≥ 10 problems (D-69) | Fixed in advance, which is good. The 10% loss threshold was never tested for sensitivity. | 🟡 | J9 |
| V3 | Each published method judged on its own goal | D-69 | Argued (fairness). | ✅ argument | – |
| V4 | Answer key: exact search 60 s, 1,000,000 nodes, counted only when proved | `bench.py` | Proved optima are exact whatever the budget; only coverage changes. | ✅ construction | – |
| V5 | NFSP 600 s per ILP solve | Q-118 | User decision. | ✅ decision | – |
| V6 | Open choices in each published method | `notes/lit_bench/selection.md` | Each reproduces a figure or table of its paper. | ✅ | – |
| V7 | One run per problem | `bench.py` | The search has no random seed, but its CPU deadline makes it slightly nondeterministic. `benchmark_plan.md` asks for 3 seeds where results can vary. | 🟡 | J1 |
| V8 | Time: ours at 900 s (and 60 s) vs each baseline's own time | E-101 | Reported, not a bar. | ✅ reported | – |

### 6. Not in method-v3

The PPO agent, hints, imitation, expert iteration and value heads were tried and dropped
(E-22 to E-54). The paper reports them as negative results with the settings they ran at; they
need no sweeps unless the paper claims something about them.

---

## Part 2. The runs

All runs use dev sets only (val s1-s5, `val_emrcm_v1`); the test sets stay untouched (SPEC
0.33). Method-v3 is frozen (D-83): if a sweep finds a clearly better setting, the paper reports
it as sensitivity, and changing the method (which means rerunning the finals) is the user's call.

**Shared subset (fixed before any run), 110 problems:** val s1 10, s2 10, s3 20, s4 20, s5 20, and
`val_emrcm_v1` 30 (10 each at N = 3, 4, 5), drawn with a written-down seed. Problems the switch
fires on (F) are mostly N 4-5 and s5, so J5 uses only those.

**Reading rule for every sweep, fixed now:** each setting is compared with method-v3 per
problem (cost ratio, 95% bootstrap CI of the mean). **Level** = the CI contains 1, or its mean
is inside J1's noise band. **Better or worse** = the CI excludes 1 and the mean is outside the
noise band. Method-v3's value is **justified** if its neighbours on each side are level or worse.
Each run also reports time to the final plan and, for search runs, how many moves the tree
committed (the depth of E-81).

**Cost:** one search configuration on the subset is about 110 × 900 s ≈ 27 CPU-h, about 45 min
wall on 36 coe9 workers (more with overshoot). Planner-only configurations take minutes, except
chain depth 4 and large k on s4-s5 (likely hours).

| Run | What | Closes | Configs | Compute | Needs |
|---|---|---|---|---|---|
| **J0** | **Offline, from the final B1 / B2b records and existing logs.** Method-v3's cost-vs-budget curve per stage (cost at 10, 30, 60, 120, 300, 600, 900 s from `ours.trace`). Overshoot of the cap. Switch times and finisher modes. Plus small instrumented reruns on the subset: how often a leaf's finish exceeds the step limit (S14), whether `finish_from`'s cost agrees with the engine's at success (F8), and how often the tree and MRCM search budgets bind (P15, P16). | S14, S16, S18, F7, F8, P15, P16 | – | none (records) + minutes | nothing |
| **J1** | **Noise floor.** Method-v3 three times on the subset, plus two runs with random tie-breaking (seeded `rng`). Gives the noise band every other run is read against. | S15, S17, V7 | 5 | ~4 h wall | run first |
| **J2** | **Sims and exploration**, a grid: sims per move {16, 64, 256, 1024} × C_PUCT {0.5, 1.5, 4}. Reports depth committed. Answers whether the s4-s5 depth limit is a property of the method or of 256. | S2, S3 | 11 (+ J1's v3) | ~8 h wall | J1 |
| **J3** | **Search mechanics**, one factor at a time at J2's chosen point: max backup; first-play value {0, −1}; value scale from the first plan / rank-based; floor off; dead-end value {−2, −0.5}; move played = cheapest plan through it; transpositions merged; moves without Discard. | S6-S10, S12, S20, S21 | ~9 | ~7 h wall | J2 |
| **J4** | **Other ways to spend the time**, same finisher, same 900 s: beam search over moves (width 4 and 16, leaves scored by the finisher); nested rollout (NRPA, level 2); greedy one-move lookahead repeated to the end. E-37's random restarts are the existing fourth arm. | S1 | 4 | ~3 h wall | J1 (code for beam and NRPA) |
| **J5** | **Finisher switch**, only on the problems where it fires (N 4-5, s5): mean {5, 10, 40} s with the real rule, min leaves {1, 10}, tier share {0.25, 0.75}. | F3-F5 | 7 | ~3 h wall | J1 |
| **J6** | **Planner alone**, every val problem, cost and time per stage: chain depth {1, 2, 3, 4}; blend partners {0, 2, 4, 8, all}; last-input rule off; lookahead k {1, 2, 3, 5, 8}; drop each recipe in turn; eviction {furthest, smallest, oldest, random}; greedy rule {per delivery, total, per unit volume}; deliveries per droplet {most and 1, all counts}; tolerant aim {coarsest, nearest}; extra steps {2, 4, 6}; interleaved targets vs one at a time; order rules (with D-96). | P1-P16 | ~30 | ~1 day, mostly depth 4 and k ≥ 5 on s4-s5 | nothing (can start now, Mac for s1-s3) |
| **J7** | **The planner settings that matter inside the search** (from J6 and D-96): chain depth 2 vs 3 at the leaves under 900 s; orders {best 1-2, 4}; blend partners if J6 shows a speed-cost trade. | P1, P5, P6 | 3-5 | ~4 h wall | J1, J6, D-96 |
| **J8** | **Problem-side sensitivity**, method-v3 and the strongest baselines on the subset: fluid weights {16/4/1, 64/8/1, 256/16/1}; mix weight {0, 1, 4}; waste weight {0, 1, 4}; step-limit floor {1.5, 2, 4}× the witness. Question: does the ranking of ours against the best baseline hold? | E2, E3 | ~9 | ~8 h wall | J1 |
| **J9** | **Offline statistics from the final records:** power of 40 and 100 problems for the effect sizes we report; the verdicts re-read with the loss threshold at 5% and 20%. | V1, V2 | – | none | nothing |

**Built (E-103, E-104):** everything. Search runs (J1, J2, J3, J4, J5, J7, J8) in
`scripts/ablation.py`, which also counts S14, S21, P15 and P16 on every run; J6 in
`scripts/justify_planner.py`; J0 and J9 in `scripts/justify_offline.py`; J8's published-method
plans and its reading in `scripts/justify_problem.py`. J4 is beam search (width 1, the rollout
algorithm; 4; 16); nested rollouts (NRPA) are left out, since they need thousands of cheap
playouts and ours are planner finishes of seconds each. Every run starts with empty memo
caches, so times compare.

**Run order and what each run waits for:**

| Run | Needs before it can run | Needs before it can be read | Machine | About |
|---|---|---|---|---|
| J0, J9 | nothing (final records on coe9) | nothing | coe9, 1 core | minutes |
| J8 published plans (`justify_problem.py published=build`) | nothing | – | coe9, few cores | ~1-3 h (NFSP) |
| J6 | nothing | nothing (deterministic) | coe9 | ~150-200 CPU-h |
| **J1** | nothing | nothing | coe9 | 145 CPU-h |
| J2 | nothing | J1 (noise band) | coe9, whole machine | ~350 CPU-h |
| J4 | nothing | J1 | coe9 | ~110 CPU-h |
| J5 | nothing | J1 | coe9 | ~75 CPU-h (s5 and N 4-5 only) |
| J8 runs | nothing | J1, J8 published plans | coe9 | ~300 CPU-h (search + 4 planner runs, capped) |
| J3 | J2's chosen point (if not 256 / 1.5, join it with '+') | J1 | coe9 | ~250 CPU-h |
| J7 | J6 and D-96 (which planner settings and orders to try inside the search) | J1 | coe9 | ~190 CPU-h |

So: J0, J9, J8's published plans and J1 now; J6 beside them. J2, J4, J5 and the J8 runs once
the machine is free (all read against J1). J3 after J2 is read, J7 after J6 and D-96. Every
search run must use the same commit as J1 or later (the memo caches are cleared per run from
E-104 on; a J1 started earlier is rerun).

**What the paper gets from each:** a sensitivity table per layer (search, finisher, planner,
problem), one figure for J2 (cost and depth over sims × C_PUCT), one figure for J0 (cost vs
budget per stage), and one sentence per ✅ row citing its E- or D- entry.
