# PROJECT_CONTEXT.md — RL Design Automation for MEDA Biochips

Context for Claude Code. Compiled on 5 Oct 2026 from all chats of the `cse-btp` project
(1 Sep – 5 Oct 2026), including the mid-sem report, the presentation and the codebase spec.

**Order of authority when files disagree:**
1. `SPEC.md` (5 Oct 2026) — the codebase specification. Latest and most exact.
2. `CLAUDE.md` — short coding rules that point to `SPEC.md`.
3. This file — the research context, the reasons behind decisions, the plan, and the open items.

If `SPEC.md` is not in the repo yet, section 8 of this file gives its main points.

---

## 0. Read this first (the 12 rules that decide if the code is right)

1. **The environment does all arithmetic exactly. The network only scores choices.** No network ever computes a ratio.
2. **No floats in the rules engine.** Compositions are exact integer numerators over one integer denominator. Floats only appear in network observations.
3. **Illegal actions are masked, never penalised.** A penalty teaches the agent to break a rule "rarely". A mask makes it impossible.
4. **Hard-constrain only what is physically required** (ratio within tolerance, volume conservation, valid DAG, k + l ≤ M, storage limit). Do **not** copy design shortcuts from published algorithms (for example "depth = shortest factor list", or "a parent may take only 1 unit of an intermediate"). These shortcuts remove the plans that can beat the baselines.
5. **The state is a set of droplets, not a grid.** Sample preparation has no geometry. Use a permutation-invariant encoder (set encoder / transformer without positional encoding / GNN), not a CNN.
6. **Pure, deterministic transitions:** `apply(state, action) -> (next_state, step_cost)`. Immutable, hashable state. All randomness through a passed-in `numpy.random.Generator`. Needed for exact search, tabular Q, MCTS and caches.
7. **A brute-force exact optimum for small instances is mandatory.** RL code fails silently. Without the optimum you cannot tell "the agent is bad" from "no better plan exists".
8. **Every solver and agent outputs the same mixing-DAG format**, checked by one validator. Later stages (scheduler, router) read this format.
9. **No distance-to-target reward shaping.** It punishes building shared intermediates, which is the main source of savings. If shaping is added later, it must be potential-based.
10. **Every physical assumption is a config value** (M, partial take, storage limit, cost weights, tolerance). The supervisor has not confirmed these values yet.
11. **Never truncate or clip silently.** Raise an error that names the instance and the limit.
12. **Environment speed is the bottleneck, not the GPU.** Profile steps/second early.

---

## 1. The project in one paragraph

A MEDA biochip runs biochemical tests by moving droplets over a grid of micro-electrodes. Its main advantage is parallelism: many experiments can share one chip in one run. Published work does not reach that case. Learned droplet routing is shown with 1–2 droplets, and sample preparation is solved for one set of targets at a time (usually dilutions of one sample, not mixtures of many reagents). The design stages (sample preparation, scheduling/binding, routing, washing) are solved one after another, each scored by its own numbers, and nothing carries the real execution cost back to the stage that made the plan. **This project builds complete design automation for many concurrent, multi-reagent experiments on one MEDA chip. Each stage becomes a learned agent, and the cost measured when the plan runs on the chip is fed back as reward to the stage that produced the plan.**

**Aim (from the final report):** a design-automation flow for MEDA biochips in which every stage — sample preparation, scheduling and binding, routing, and washing — is run by a learned agent, and the cost measured when a plan is executed is fed back to train the stage that produced it, so that many multi-reagent experiments can share one chip in a single run.

**Current stage (Oct 2026): the sample-preparation agent.** It is the first stage to build. The rest of the pipeline is reserved for later.

---

## 2. People, setup, preferences

- **Context:** final-year B.Tech project (BTP), NSUT (Netaji Subhas University of Technology), New Delhi. Two students (the user and a teammate who knows less RL). Mid-semester evaluation was ~30 Sep 2026 (report and 26-slide deck done).
- **Supervisor:** works on MEDA routing. Author of a 2026 paper, *polynomial-time optimal routing (Dijkstra-based) for MEDA*: one droplet of volume 2 (shapes 2×1, 1×2, or split into two 1×1), tested up to 100×100 arrays. He has his own router code (possible cost oracle later). He moved the project scope (3 Sep) from routing only to full design automation, from single target to many targets. His advice: "make some things work first, then optimise the complete automation"; aim for an end-to-end, multi-droplet solution instead of comparing one small algorithm against ten others.
- **Senior:** Yash (PhD/senior). His paper: DRL error recovery + electrode health on MEDA (TGT + Swin + GNN-ODE, PPO, 4 recovery actions). Context only.
- **Hardware:** workstation with **NVIDIA RTX A5000 (24 GB)** and 256 GB RAM. An A1000 may come later (use it for evaluation or surrogate training, not main training). Older files say "A500" — that is wrong. Number of CPU cores matters more than the GPU for sample prep.
- **User's background and way of working:**
  - Strong probability and maths. Studied RL theory from first principles (return, value, Bellman, conditioning). Little or no RL code written yet.
  - Learns by building. Wants the **final research codebase from the start, not a toy/demo repo**. Will experiment and add to it while learning RL.
  - Works **top-down**: design the whole system first, then build each box. Good pattern here: every box first gets a simple classical stand-in so the pipeline runs end to end, then boxes are replaced by learned agents one at a time.
  - Wants decisions explained with reasons and math intuition, and wants wrong reasoning corrected directly.
  - Prefers simple, clear English.
- **Publication target:** first paper in the Feb–May 2027 window (NEWCAS, ISVLSI, GLSVLSI, ICCAD, MLCAD). Best-fit strong venues: DAC, ICCAD, DATE (ASP-DAC one step down); journals IEEE TCAD and ACM TODAES. ML venues (NeurIPS, ICLR) only if the RL method itself is new.

---

## 3. Domain facts the code must respect

### 3.1 MEDA vs DMFB

| Property | DMFB | MEDA |
|---|---|---|
| Droplet volume | Fixed (one electrode) | Variable, integer number of units |
| One mix | 1:1 only | **k:l with k + l ≤ M** (M = max mixing volume; literature reports 4–10 units) |
| Reachable concentrations | multiples of 1/2^d | fractions whose denominators are products of integers ≤ M (e.g. 3/10 = factors 2 and 5) |
| Splitting | equal halves | unequal parts allowed ("pseudo-dispense": take k units from a droplet) |
| Movement | 4 directions, uniform cost | 8 directions (incl. diagonal), speed depends on shape and direction |
| Sensing | few sensors | capacitive sensing in every micro-electrode (position, shape, volume — not chemistry) |

- Mix rule: k units of composition c1 + l units of c2 → (k+l) units of (k·c1 + l·c2)/(k+l).
- On MEDA the **choice of factorisation is a decision**. Example: ratio sum 30 with M = 8 → 8 ordered factorisation trees (2·3·5 in 6 orders, 5·6 in 2 orders); with M = 10 also 3·10 and 10·3.
- `d` (accuracy level) is a DMFB idea (targets k/2^d). On MEDA, define targets by **tolerance ε**, not 1/2^n. Keep precision as a problem parameter, not a chip fact.
- **Mixing does not copy fluid.** A droplet is used up when consumed. Reuse needs volume and splits; leftovers become waste. This "no free copy / volume conservation" property is the main thing that separates this problem from AlphaTensor, AlphaDev, proofs and retrosynthesis — and it is the novelty point.
- Split/merge change volume and composition. **Morphing (shape change) does not**; shape is a transport concern only (irrelevant to sample prep).
- Real MEDA splits are **not exact** (split error). This is phase 2 (noise), not now.

### 3.2 The chip as a system

- Active electrode array, plus **reservoirs at the edges** (one per reagent, one per sample, buffer, wash, waste; typically ~6–12 total). Each reservoir has a **dispensing port**; every droplet is born at a fixed port.
- **Fluid set is frozen for the whole run** (chip is sealed). It limits the number of distinct fluids, not the number of experiments.
- **Sample vs reagent:** sample (e.g. blood serum) is scarce and changes per run → weight it highest in cost. Reagents are known, cheaper. Buffer is cheapest.
- **Detection sites:** 1–4 at fixed coordinates; capacity 1, hold time, and a wash between different fluids (changeover).
- Experiments run **concurrently**: their DAGs are merged into one forest and synthesised once. Sample preparation is **joint across all experiments**: one mixing DAG with intermediates shared across experiments.
- **Stored droplets are obstacles for the router.** A stored intermediate blocks its cells plus a spacing ring (a 2×1 droplet + 1-cell ring = 12 cells), can merge by accident, and contaminates cells. Reagent-saving trees are deep/narrow → long storage. So measure **storage occupancy × duration**, not only storage count.
- **Contamination is per fluid class** (a cell dirty with plasma is fine for plasma). Washing = dispense a wash droplet, route it over dirty cells, then to waste; a wash droplet saturates after a limited number of cells.
- Classical benchmarks: multiplexed IVD (4 samples × 4 reagents = 16 assays; **no splitting and no storage, so not a sample-prep benchmark**); PCR mixing (7 mixes); colorimetric protein assay (deep serial dilution). Scaled protein family: Protein 1/4/5/7 = 36/316/636/2556 ops with 8/78/158/638 storage ops. **No MEDA benchmark suite exists.**

---

## 4. The full system and its end architecture

### 4.1 Pipeline (data flow)

```
Inputs (chip spec + experiment specs: target ratio vectors, demand, tolerance, deadlines)
   -> Sample preparation          -> one mixing DAG (shared intermediates)        [runs once, before the chip]
   -> Scheduling + binding        -> start time + resource per operation           [runs once]
   -> Placement (module-based) / merge points (routing-based)
   -> Executor advances time; at each slot boundary calls the ROUTER with
      a list of (droplet, source, destination) moves                             [runs many times]
   -> Wash decision + wash droplets (through the same router)
   -> Chip: actuation, detection, waste  -> realised cost (time, washes, contamination)
   -> (planned) realised cost fed back as reward to sample prep / planner
```

- The DAG gives a **partial order**, not times. Scheduling is resource-constrained DAG scheduling (NP-complete); classical default = list scheduling with critical-path priority.
- The **executor** calls the router, not the other agents. The router never chooses its (source, target) pairs.
- If routing fails, the schedule is stretched or redone (existing feedback loop).

### 4.2 Agreed end state: two agents (options / SMDP)

- **Planner** (one agent): decides **what** happens — sample prep (which droplets to mix, at which ratio), scheduling/binding (when, on which resource), and **wash decisions** (when to wash, how many wash droplets, which cells each covers). Acts a few dozen times per assay.
- **Router** (one agent, called like a function): decides **how** droplets move. Inside it: one shared network applied to every droplet (MARL, parameter sharing, CTDE), each droplet conditioned on its own goal and fluid class. A split spawns another actor; wash droplets use the same network with a role flag / "cover these cells" goal. Acts every time step.
- The planner works on a semi-MDP: each router call is a temporally extended action.
- **Router contract (define once, keep fixed; the classical router must use the same signature):**
  - In: chip occupancy, residue map by fluid class, list of (droplet, source, destination), deadline.
  - Out: paths, or failure.
  - Reported back to planner: time taken, cells newly contaminated, success/failure.
- Known risks: the planner state becomes mixed (droplet set + geometry) → feed a **summary** of contamination (per-class dirty-cell counts, coarse density), not the raw grid; planner cannot intervene mid-route (allow early return on failure); credit assignment across 3 responsibilities → add one responsibility at a time.
- **Rejected:** one flat agent over everything (timescales differ ~1000×, credit cannot travel ~40,000 steps); four peer agents that negotiate (non-stationary, adds no structure — the interface already exists).
- **Adopted first:** train one agent (sample prep); everything below it is classical code that returns a cost. This is a normal single-agent problem and already gives the end-to-end objective.

### 4.3 Why RL (claims the project uses — and does not use)

Use:
1. **Amortisation:** a solver pays per instance; a trained policy answers a new target set quickly.
2. **No heuristic exists for MEDA k:l mixing with many targets and reagents.** The learned policy/value is that heuristic (V* = perfect A* heuristic h*).
3. **The true cost is only samplable** (known only after scheduling/routing/washing run).
4. **Cost depends on the future** (contamination: a short dirty route now forces a wash later).
5. Phase 2: **split error** makes it stochastic → a closed-loop policy beats an open-loop plan.

Do not use: "the search space is large" (the supervisor's reformulation refutes this type of claim), and "electrode degradation" (known failure probability p → edge weight 1/p or −log p → Dijkstra solves it). Do not claim to beat an exact solver on its own objective for a fixed instance; claim speed, scale, a larger search space, multi-objective coverage, generalisation.

---

## 5. Plan of work

### 5.1 Objectives (final report, Section 5.1)

1. Build the sample-prep environment (many reagents/targets/experiments, k:l mixing, volume conservation, storage limit, masking) **and a brute-force exact solver** for small instances.
2. Train the first agent: joint sample preparation; compare with the optimum and with published MEDA methods.
3. Make the true cost measurable: a classical execution harness (scheduling, binding, placement, routing, washing). First result: do proxy scores (mix steps, reagent, waste) predict real completion time / wash count? (Correlation over many generated DAGs — a result that needs no RL.)
4. Put a learned agent on every stage, one at a time.
5. Couple the stages through a shared reward (train sample prep on realised cost).
6. (Extension, first to drop) one preference-conditioned multi-objective model.

Objectives 1–3 committed by end of Jan 2027; 4–5 by Apr 2027.

### 5.2 Timeline (Gantt from the report; confirm with semester calendar)

| Work item | Dates |
|---|---|
| Problem formulation | Sep – mid Oct 2026 |
| **Environment + exact solver** | **Oct 2026** |
| Classical baselines | mid Oct – mid Nov 2026 |
| Sample-prep RL agent | Nov – Dec 2026 |
| Classical execution pipeline | Dec 2026 – Jan 2027 |
| Proxy-cost correlation check | late Jan 2027 |
| Scheduling / routing agents | Jan – Feb 2027 |
| Washing agents | Feb – Mar 2027 |
| Reward sharing | Mar 2027 |
| Multi-objective RL | Mar – Apr 2027 |
| Report + ablations | Apr 2027 |

### 5.3 Sample-prep sub-problem order (curriculum and research ladder)

| Step | Problem | Baselines |
|---|---|---|
| 1 | Dilution (2 fluids), one target, M = 2 (DMFB-like, tabular) | brute force, bit-scanning |
| 2 | Dilution, one target, M = 8 | WSPM, FacDA, brute force |
| 3 | Dilution, many targets with demand (sharing becomes useful) | Liang 2020, Fukui 2024 |
| 4 | Mixture (3+ fluids), one target | Kundu & Roy 2022 |
| 5 | Mixture, many targets, many experiments | Kundu 2023; Demand-driven mixture prep (DMFB) |

Learning ladder (each rung fails on purpose and motivates the next tool): tabular Q on tiny cases (must match BFS) → larger M/precision breaks the table → function approximation (instability → replay/target nets) → variable droplet set and changing action set → masked PPO with set encoder → if PPO does not find sharing, MCTS guided by π and V (AlphaZero style) → slow true cost (cache + surrogate, AlphaDev style).

---

## 6. Sample preparation: the formulation

### 6.1 Formal problem

- **Given:** fluid set F (one per reservoir; stock may be non-pure later); targets T from all experiments (each: ratio vector over F, demand = number of droplets, droplet volume, tolerance ε); chip limits (M, min mix volume, storage capacity S, unit volume, min droplet volume, allowed partial take).
- **Find:** one mixing DAG (dispense / mix / output / discard; split and store are implicit in v1) such that every target gets at least `demand` droplets within ε.
- **Subject to:** volume conservation at every node; storage ≤ S at all times; k + l ≤ M; per-target error ≤ ε.
- **Minimise (phase 1):** one main objective with clear priority. Default: weighted reagent use (sample > reagent > buffer) + mixes + waste. Report the structural metrics too.

### 6.2 MDP (phase 1: deterministic, fully observed)

- **State:** multiset of droplets on chip, each (exact composition vector, integer volume); remaining demand per target; storage used; step count. Reservoirs are implicit sources with unlimited volume, not droplets.
- **Transition:** deterministic, `s' = f(s, a)`; P is 0/1. Randomness is only in the start state s₀ ~ p(s₀) (the problem generator) and in exploration.
- **Action:** autoregressive in concept: operation → droplet A → droplet B → ratio k:l. **In SPEC v1** this is implemented as `Discrete(L_max)` = index into the canonical list of currently legal candidate actions, with a mask and per-candidate features (afterstate features: the exact result of each candidate, computed by the core). Actions: `Mix(a, k, b, l)`, `Output(d, t)`, `Discard(d)`. Dispense is implicit (take from a reservoir); store is implicit (any droplet on chip is stored); explicit Split is not in v1.
- **Reward:** `reward_t = −cost_t`, γ = 1. Cost = Σ w_fluid[kind]·reagent units + w_mix·mixes + w_waste·waste. Default weights (placeholders): SAMPLE 10, REAGENT 2, BUFFER 0.5, mix 1, waste 1. Leftover droplets at the end count as waste.
- **Termination:** success (all demand met); truncation at step limit → add a penalty = cost to finish with a known fallback method (AlphaTensor idea), so an unfinished plan is never cheaper; dead end (no legal action) → same penalty, and track its rate.
- **Reward tiers:** Tier 0 validity (masking). Tier 1 free from the DAG: reagent per fluid, waste, mixes, **peak droplets alive, storage occupancy × duration, critical-path depth**, (fluid-class diversity at peak). Tier 2 expensive: realised completion time, wash count (needs the chip pipeline; use later with cache and surrogate).
- **Network:** set encoder over [global token; sources; targets] (transformer, no positional encoding), candidate scorer that gathers referenced embeddings + candidate features → logit; masked categorical; value head on global token. Give arithmetic features (denominators/log₂, residues mod M, prime factors, L∞ distances, "satisfies target" flags) so the network never has to discover divisibility.
- **Algorithm:** in-house PPO (CleanRL style) with masking and GAE; `sb3-contrib` MaskablePPO only as a cross-check. Search (MCTS/PUCT using π and V on the pure `apply`) is reserved for later. Note: MCTS for this problem has AND nodes (a mix needs both inputs) as well as OR nodes.

### 6.3 Reviewed fixes to the student's original formulation (26 Sep, all adopted)

1. Precision 1/2ⁿ is DMFB → use tolerance ε.
2. Add k + l ≤ M (main MEDA constraint; defines the action space).
3. Rename clashing symbols (earlier: M = number of targets and R = split ratios; now **M = max mixing volume**).
4. Seven weighted terms is too many → one objective + fixed priority order; report the rest as metrics.
5. Distance shaping can kill sharing → no shaping, or potential-based; or AlphaTensor-style step-limit penalty.
6. HER is overstated (off-policy, gives wasteful demos) → prefer **synthetic forward-generated instances** (random legal mixes from reservoirs; mark products as targets; plan cost = known upper bound).
7. "Sharing will emerge" is a hope → test on a small case where the optimum is known to share.
8. Brute-force optimum was missing → added.
9. Split noise → phase 2.

### 6.4 Phases of the sample-prep problem

| Phase | Transition | Observation | Class |
|---|---|---|---|
| 1 (now) | deterministic | full | known deterministic MDP |
| 2 | split error, c′ = c_planned + noise | real volume after split (capacitive sensing) | stochastic MDP, closed-loop replanning |
| 3 | stochastic | volume only; concentration hidden | POMDP (belief over concentration) |

Code hook: `core/noise.py` with `NoNoise` default; one place where the split/take result is computed. With noise on, set `info["stochastic"] = True` and disable exact search.

### 6.5 Alternative formulation (plan B, from PrefixRL)

Edit-based instead of build-from-empty: start from a valid DAG (e.g. FacDA's), action = local edit (change a factorisation, re-route an intermediate, change one k:l), a **legaliser** repairs volumes/demands so every state stays valid, reward = improvement each step. Pros: dense reward, never invalid, guaranteed floor (≥ baseline). Cons: local exploration only; needs a working baseline and a legaliser. Keep as plan B if build-from-empty cannot find valid plans at step 4–5 of the ladder.

### 6.6 Multi-objective (later)

Either one agent per weight vector (PrefixRL style, traces the Pareto front) or one preference-conditioned policy π(a | s, w) with w sampled per episode (Abels 2019 conditioned networks; Yang 2019 envelope Q-learning). Normalise objectives before weighting. Linear weights miss concave parts of the front. Compare against each baseline at its matched weights. SPEC has `env.preference_conditioning` (default false).

---

## 7. Worked example used across the report and slides

Fluids A, B, C. Targets: E1 = 0.5:0.3:0.2 (demand 2), E2 = 0.5:0.5:0 (demand 1), E3 = 0.25:0.15:0.6 (demand 4).
One joint plan: I = A+B 1:1 (= E2); J = A+C 1:1; E1 = I:J at 3:2 (needs M ≥ 5); E3 = E1 + C at 1:1.
Check: 3/5·(0.5,0.5,0) + 2/5·(0.5,0,0.5) = (0.5,0.3,0.2); ½·(0.5,0.3,0.2) + ½·(0,0,1) = (0.25,0.15,0.6).
0.3 = 3/10 is not dyadic; DMFB can only approximate it (e.g. 77/256 = 0.3008 at 8 levels).

Classic sharing example (good first test): targets 3/4 and 1/4 with M = 2 → optimum 3 mixes (shared 1/2), naive = 4.

---

## 8. Codebase decisions (summary of SPEC.md, 5 Oct 2026)

- **Repo:** `meda-da/`, package `src/meda/`. Python ≥ 3.11, `uv`, `pyproject.toml` + `uv.lock`.
- **Stack:** numpy, gymnasium 1.x, torch (CUDA for A5000), hydra-core/omegaconf, tensorboard (wandb optional), pandas + pyarrow, matplotlib, networkx (validation only, not hot path), rich/tqdm, pytest + hypothesis, ruff, mypy (strict on `meda.core`), pre-commit; optional sb3-contrib, numba (only after profiling). Exact arithmetic with Python `int`/`math.gcd` (no `fractions.Fraction` in hot paths; it may be used in tests to cross-check).
- **Layout:**
  - `core/` exact rules, no ML imports: `composition.py`, `fluids.py`, `chip.py`, `problem.py`, `state.py`, `actions.py`, `rules.py` (`legal_actions`, `apply`, `is_terminal`), `costs.py`, `dag.py`, `noise.py`.
  - `envs/` Gymnasium env (`sample_prep_env.py`), `encoding.py`, `wrappers.py`.
  - `generators/` sampler (p(s₀)), synthetic forward-built instances, curriculum, benchmark I/O.
  - `solvers/` exact search (uniform-cost/A* over canonical states), fallback cost-to-finish, baselines (`naive`, `bitscan`, `facda` marked unverified until it reproduces the paper).
  - `agents/` random, tabular Q (keys = canonical state/action keys, never candidate index), PPO (`networks`, `policy`, `rollout`, `ppo`), `mcts/` reserved.
  - `analysis/` DAG metrics (proxy + structural via abstract list schedule), evaluate, compare, plots.
  - `pipeline/interfaces.py` reserved: `Scheduler`, `Router`, `CostOracle` protocols only.
  - `configs/` (chip: `meda_m8.yaml`, `dmfb_m2.yaml`; costs; generator stages; env; agent; train; eval), `benchmarks/` (frozen versioned JSON sets), `scripts/` (`profile_env.py`, `make_benchmark.py`), `tests/`, `notebooks/` (never imported by src), `outputs/` (gitignored).
- **Chip defaults (meda_m8):** max_mix_volume 8, min_mix_volume 2, unit_volume 1, min_droplet_volume 1, allow_partial_take true, storage_capacity 10 (droplet count), max_denominator 256, allow_same_composition_merge false. `dmfb_m2`: mix volume exactly 2 (1:1 only).
- **Composition:** `num: tuple[int, ...]`, `den: int`, `sum(num) == den`, reduced by gcd; `mix(c1, k, c2, l)`; exact `linf_distance`; `to_float_array()` only for observations.
- **Canonical keys:** state key = multiset of (composition, volume) + remaining demand (uids dropped); action key uses referenced droplets' (composition, volume) + k, l, target. Mix(a,k,b,l) and Mix(b,l,a,k) are one action. Legal-action list order is canonical (Output, then Mix, then Discard).
- **Mixing-DAG format:** nodes `RESERVOIR | MIX | OUTPUT | WASTE` with composition, volume, step; edges `{src, dst, volume}`. Validator checks: acyclic, volume conservation, composition consistency, mix limits (≤ 2 parents in v1), outputs within tolerance with correct volume, demand met, reported totals match, storage replay ≤ capacity. JSON I/O with `instance_id`, `producer`, totals.
- **Env:** `action_space = Discrete(L_max)`; obs Dict with padded `sources`, `targets`, `candidates`, `cand_index`, masks, `globals`; error (not truncation) if legal actions exceed `L_max`; works with `AsyncVectorEnv`; target ≥ 10,000 steps/s total with 16–32 workers for curriculum stages 1–3.
- **Benchmarks:** `tiny_exact_v1` (optima stored), `val_<stage>_v1`, `unseen_ratios_v1`, `larger_v1`, `literature_v1` (hand-transcribed from WSPM, FacDA, Kundu papers with reported results). Never regenerate a published version. Check no train/test overlap by canonical instance hash.
- **Milestones (strict order):** M0 scaffold → M1 exact core → M2 DAG → M3 env + random agent (`check_env`, 10,000 random episodes, profile) → M4 generators + benchmarks → M5 solvers → M6 tabular Q (matches optimum on `tiny_exact_v1`) → M7 evaluation harness → M8 PPO (overfit one tiny instance; stage 1 success > 95% and gap < 5%; resume from checkpoint reproducibly) → M9 reserved (MCTS, literature baselines, preference conditioning, noise, pipeline).
- **Hand-checked solver cases (user must verify them; they become core tests):**
  1. Target 1/4, M = 2: 2 mixes (S1+B1 → ½ vol 2; 1 unit ½ + B1 → ¼ vol 2; output 1). Reagent S=1, B=2. End waste 2.
  2. Target 1/4, M = 4: one mix S1+B3 → ¼ vol 4; output 1; waste 3. Which of 1 vs 2 is cheaper depends on weights — test both weightings.
  3. Target 3/10, M = 8: two mixes (S2+B3 → 2/5 vol 5; 3 units 2/5 + B1 → 3/10 vol 4). Infeasible under `dmfb_m2` with tolerance 0.
  4. Sharing: targets 3/4 and 1/4, one droplet each, M = 2 → optimum 3 mixes; naive baseline 4.
- **Property tests:** volume conservation after every step; compositions sum to 1 exactly; every legal action applies; illegal ones raise `IllegalActionError`; determinism; uid permutation does not change canonical key; no float in `PrepState`/`StepCost`.
- **Workflow with Claude Code:** one milestone per session ("Implement M0 from SPEC.md"); tests, ruff, mypy before "done"; any change to a public interface, the DAG format or a default physical assumption updates `SPEC.md` in the same change; ask before adding a dependency.

---

## 9. Baselines, literature and reporting

### 9.1 Sample-prep baselines (to implement or reproduce)

| Method | Platform / scope | Notes for implementation |
|---|---|---|
| Brute force / exact search | any, small | ground truth; canonicalise states or it never ends |
| Naive (each target built alone) | any | always-available baseline and fallback |
| Bit-scanning (BS) | DMFB, 2 fluids, 1 target | binary expansion; valid only under `dmfb_m2` |
| **WSPM** (Li et al., IEEE TBioCAS 11(6) 2017) | MEDA, dilution, 1 target | BFS over primary droplets (PD = sample diluted with buffer only), state = (concentration, size) pair; Sc (size limit) sets per-step ratios; δ grid + **rounding** to grid (`||·||` in eq. 1); ECT heuristic for intermediate droplets (loose upper bound, first-found, used only for ordering); greedy top-down DG; PD sharing reuses leftover droplets. Priority: sample > operations > waste. Tested Sc = 2,3,4 (ops 8.03 → 4.07 at δ = 1/256). Note: WSPM rounds, our core is exact — reproduction needs a rounding mode or tolerance. |
| **FacDA** (ASP-DAC 2019) | MEDA, dilution | factorisation-based; reports ~38% fewer steps, 52% less sample, 23% less waste vs earlier MEDA methods; M 4–10. **Author list uncertain** (Kundu et al. vs Bhattacharjee et al.) — check the PDF. Mark unverified until it reproduces published numbers. |
| Kundu & Roy 2022 (DSD) | MEDA, mixture, 1 target | skeleton tree + SMT to minimise waste; levels = shortest factor list (a design shortcut — do not copy into the agent) |
| Kundu 2023 | MEDA, mixture, multi-target | |
| Liang et al., TCAD 39(10) 2020 | MEDA, multi-target | |
| Fukui 2024 | MEDA | reuse of intermediate droplets |
| Demand-Driven Single-/Multi-Target Mixture Prep (Bhattacharya group) | DMFB, N ≥ 3, many targets, demand | closest problem match ("direct baseline"); sharing by local subtree matching only |
| Others named | | WARA, REMIA, RSMA, MinMix, de Bruijn/ATSP (Mitra TCAD 2014; built on 1:1 mixing, no storage, high waste), DMRW, IDMA, GORMA, FloSPA-D (exact, ~216,000 s at accuracy 6), TVODA, MRCM (min most-costly reagent), Dong 2024 splitting trees, Saha 2026 prime factorisation |

- DFMM-style algorithms let a parent take only **1 unit** of an intermediate (rest is waste). MEDA allows k units (e.g. 7:7:6 tree: child 1:1:2 vol 4, parent takes 3 units + r1 + r2 → waste 1, reagents 6, mixes 2). Edge volume is a decision variable for the agent.
- Acceptance rule: **an implemented baseline must reproduce its published numbers inside our simulator before any agent is trained.** If not, the simulator is wrong.

### 9.2 Reporting conventions

- Sweep **all targets** at an accuracy level, or thousands of random target sets (η = 2…100 targets). Never a few hand-picked cases.
- Report mean / median / 90th percentile; gap to optimum (where stored) and to each baseline; ≥ 3 training seeds; generalisation on unseen and **larger** instances than training.
- Report solver/inference time (answers the TODAES 2025 objection that learned placers are too slow online).

### 9.3 Later-stage anchors (for reserved packages)

HSM (Li et al. DAC 2016; TBioCAS 11(3) 2017, includes velocity model and reservoir placement), WLSM (Zhong et al. TCAD 2021, wear levelling), Kundu et al. ACM TIST 17(1) 2025 (RL placement + heuristic routing; mixing graph is given), Online chamber-less synthesis (TODAES 2025), Liang ICML 2020/2021 (MARL routing, open-source MEDA-Env, 2 droplets), Elfar DATE 2022 / TCAD (health-aware DRL routing), target-guided shape-aware routing (Biosensors 2025, 1 droplet, ≤ 12×12), simultaneous wash routing (2024/2025), supervisor's 2026 Dijkstra router. Su & Chakrabarty JETC 2008 (HLS formulation), Grissom & Brisk (scheduling NP-complete), Maftei/Pop/Madsen (routing-based synthesis). Mixing-during-transport model exists only for DMFB (Paik et al.); hybrid recommended: mixing routing-based, incubation/detection placed.

### 9.4 Cross-field analogies (what each gives the design)

| System | Take | Where it breaks |
|---|---|---|
| AlphaZero | exact rules engine + network that only judges; search as policy improvement | separate network per game → one method per stage, not one agent for all |
| AlphaTensor | build step by step; gain from reusing intermediates; synthetic forward data; step-limit penalty; sampled actions + masking | liquid cannot be copied; no subtraction; storage limit; tolerance |
| AlphaDev | mask invalid steps; estimate a slow true cost with a learned predictor | its check is cheap; our tier 2 is a full chip run |
| PrefixRL | score upstream graph on measured downstream cost; legalisation; one agent per weight | needed ~32,000 GPU h for 64-bit → cache + surrogate |
| AlphaProof / HTPS | AND-OR search nodes | needs human data |
| Retrosynthesis (AiZynthFinder) | top-down decomposition to stock = WSPM/FacDA direction | ignores quantities; search capped by policy quality |
| MuZero | learned model for inexact chip (phase 2) | — |

---

## 10. Lessons and rules (from the chats)

- Environment design (state, action, mask, reward, p(s₀)) is most of the work and most of the research. If boxes "environment" and "evaluation" are right, the algorithm can be swapped in a day.
- **First real checkpoint:** random policy + masking must produce valid mixing DAGs on small instances. If not, no learning will help.
- Feasibility, not the objective, is the hard part (most action sequences give a wrong ratio). Masking + feasible-by-construction actions + curriculum.
- Optimal is relative to a formulation: SMT is optimal only inside its own restricted space. The agent's chance to win is the larger space.
- Generalisation is designed and measured, not guaranteed: relative state, randomised targets, set encoder, test on held-out and larger instances.
- Avoid correlated training samples from one plan (AlphaGo value-net lesson).
- Derive "layer 1" RL (Bellman, policy gradient, baselines); meet "layer 2" fixes (replay, target nets, clipping, GAE, entropy) as answers to failures seen in this code.
- Keep the step that computes the split/take result in one place (future noise hook).
- When the true cost (tier 2) comes: cache every (DAG hash → cost), train a surrogate, call the real oracle on a small fraction; watch for reward hacking of the surrogate.
- Do not use IVD to motivate sample prep (no splits, no storage).
- Speed math: a 10M-step run is ~17 min at 10,000 steps/s and ~28 h at 100 steps/s.

---

## 11. Verified numbers (safe to quote in docs/tests)

| Quantity | Value |
|---|---|
| Ratios k:l with k,l ≥ 1, k + l ≤ 8 | 28 |
| Pairs from 10 droplets | 45 → MEDA 1,260 choices/step vs DMFB 45 |
| 20-step plan | 1,260²⁰ ≈ 1.0 × 10⁶² sequences |
| Compositions, R fluids at 1/2^d | C(2^d + R − 1, R − 1): R=3,d=8 → 33,153; R=5,d=8 → ≈1.86 × 10⁸ |
| States: multisets of 10 from 33,153 | ≈ 4.4 × 10³⁸ |
| 16 independent assays orderings | 16! ≈ 2.1 × 10¹³ |
| Routing joint actions, 10 droplets × 63 moves | ≈ 10¹⁸ per step |
| Wash assignments, 50 cells to 3 droplets | 3⁵⁰ ≈ 7.2 × 10²³ |
| Supervisor's router, 100×100, 5% dead (9,500 usable cells) | **one droplet of volume 2**: C(9500,2) ≈ 45M states, ~1.6 h, ~7 GB, 64-core; next case C(9500,3) ≈ 1.4 × 10¹¹ (not "k = 2 droplets", and not 8.6 × 10¹¹) |

---

## 12. Open decisions (all must stay configurable)

| Decision | Current default | Ask |
|---|---|---|
| Max mixing volume M on target platform | 8 | supervisor (most important) |
| Partial take (pseudo-dispense) allowed; min droplet volume | yes, 1 | supervisor |
| Storage limit as droplet count or volume | count, 10 | supervisor |
| Accuracy / tolerance metric | max denominator 256, L∞ | assay needs |
| Cost weights per fluid kind | 10 / 2 / 0.5, mix 1, waste 1 | supervisor, assay |
| Dilution first or mixture first | dilution first (curriculum) | supervisor |
| Explicit Split action; mixes with > 2 inputs | no / no | later (changes action set) |
| Same-composition merge (volume consolidation) | off | test if needed |
| Split-error model | none | literature / hardware |
| MEDA dynamic mixing model changes the arithmetic? | assumed no | supervisor |
| Mixing-during-transport model for MEDA | none known | supervisor (decides routing-based vs module-based) |
| Reservoir and detector counts; single-use chip; shape-velocity table source | unknown | supervisor |
| Access to supervisor's router as cost oracle; FacDA code availability | unknown | supervisor |
| Has anyone coupled MEDA sample-prep volume choice to routing cost? | none found | supervisor |

---

## 13. Inconsistencies across the chats (resolved / still open)

- GPU: "A500" in older files and the report → actually **RTX A5000** (resolved).
- Action design: report/slides say a 4-step autoregressive head (op → A → B → ratio). SPEC v1 uses a masked index over enumerated legal candidates with afterstate features. Same idea, different implementation; SPEC is the decision for v1. If candidate counts get too large (`L_max` errors), move to the autoregressive/pointer head.
- Ops list: report includes dispense/split/store as actions; SPEC v1 makes dispense and store implicit and has no Split. SPEC wins for v1.
- Report says "reward shaping on simplex distance" and "HER" in the 26 Sep draft → both replaced (section 6.3).
- Routing numbers: report text says "k = 2 droplets"; correct is "one droplet of volume 2" (section 11).
- Washing: report table lists washing as its own agent; final end state puts wash **decisions** in the planner and wash **movement** in the router.
- FacDA author attribution unconfirmed.

---

## 14. Glossary

- **Composition / ratio vector:** shares of each fluid in a droplet, summing to 1 (a point on the simplex).
- **Demand:** number of droplets needed of a target. STSD/MTSD/MTMD (or SDST/SDMT/MDST/MDMT) = single/multi target × single/multi demand. Reagent count (2 = dilution, ≥ 3 = mixture) is a separate axis.
- **M:** max mixing volume (k + l ≤ M). **Sc** in WSPM = droplet size limit (same role). **δ** = precision grid. **ε** = tolerance.
- **Primary droplet (WSPM):** made from sample + buffer only. **Intermediate droplet:** anything else.
- **Pseudo-dispense / partial take:** take k units out of a larger droplet.
- **Proxy cost:** mixes, reagent, waste counted from the DAG. **True cost:** completion time, washes after scheduling/routing. **Cost oracle:** the chain that returns true cost.
- **Afterstate:** the state right after the agent's own action; scoring candidates by their exact result.
- **Amortisation:** pay training once, reuse on every new instance.
- **SMDP / option:** a temporally extended action (planner calls the router).
- **CTDE:** centralised training, decentralised execution (router MARL).
