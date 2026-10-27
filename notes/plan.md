# Plan

Where the project stands and what comes next, step by step. Rebuilt from scratch on 2026-10-26
from everything learned up to E-23; the plan from §4 on was rewritten on 2026-10-27 around
stages 3–5 (D-35). `SPEC.md` is the design; `notes/paper_log.md` holds the evidence (IDs like
E-22 point there). This file says **what we are doing and why, in order**.

**How to maintain it**
- Read it at the start of every session. Update the status of a step the day it changes, with
  the paper log ID that backs it.
- When a result changes the plan, edit the plan here and record the decision in the paper log
  (D-…). Keep finished steps, marked done, so the path stays visible.
- `notes/questions.md` lists every question with its status; each open question names the
  step here that answers it. When a step finishes, close its questions there.
- Each step names its pass bar and what happens on pass and on fail. No step starts before the
  step it depends on has passed.
- **Defaults** (marked *default*) are Claude's choices so work can go on; the user can change
  any of them. When a step reaches a real decision, bring the user that one question with a
  recommendation, not the whole list.
- Compute: the Mac runs only light jobs at background priority (`OMP_NUM_THREADS=1 taskpolicy
  -b nice -n 19`). Anything longer than ~10 minutes goes to the workstation (`~/code/Btp`,
  tmux; never reboot it during runs).

*Last updated 2026-10-27 (D-35, F-16 fixed, SPEC 0.27).*

---

## 1. Goal

One RL policy that plans MEDA sample preparation (k:l mixing, any M, F, targets, demand and
storage) and produces near-best-known plans for **MTMD mixture preparation with a storage
limit**, a class no published method handles, while matching specialised methods on the
sub-classes they handle (claim fixed 2026-10-21, paper log §1.2). Generalisation across sizes is
a requirement of the agent, not the headline.

**What the agent must show (D-35):** on stages 3–5 it makes cheaper plans than the **best
planner** we have (the cheapest of the sharing pass, its lookahead and its extra target
orders; the proved optimum where one exists), above all where that planner is weak: tight
storage and many targets (E-28). The agent may use planner-derived hints; the paper says so
plainly and reports how much the agent leans on them.

## 2. What we have learned (the reasons behind the plan)

| # | Lesson | Evidence |
|---|---|---|
| 1 | A failure penalty that is too small teaches the agent to give up. | F-11, D-26 |
| 2 | Pure PPO cannot start: random play solves 2/1,200 dilutions and 0/1,200 mixtures (200–800 legal moves per step, 6–12 correct moves in a row). The regime does not matter while nothing is learned. | E-9–E-11 |
| 3 | The network could not find the right intermediate droplet (first mix right 24% on s1, 6% on s2, even on training problems). It saw ratios as decimals; the arithmetic that makes 1/8 the way to 33/40 was invisible. | E-14–E-16 |
| 4 | Exact hint columns computed by the env (mixes left to a target along known recipes) fix that. They, not the choice of teacher, lift copying from ~30% / 0% to ~75% / 65%. | E-16, E-20, E-21 |
| 5 | The sharing pass (reuse leftovers across targets) is a far better teacher and reference than the old construction: optimal on 51/65 small proved problems vs 34, never dearer. Its quality on many-target **mixtures** is uncertified because the bound is loose there. | E-17 |
| 6 | The old failure penalty ignored droplets on the chip, so building good intermediates earned nothing. The penalty now finishes from the current state. | E-18, D-32 |
| 7 | Teachers are not experts everywhere: cheap moves plus a planner finish beat the teacher on 7/40 s1 problems, and RL found plans 2–3× cheaper on some. Imitation must stay a warm start. | E-19, E-22, D-31 |
| 8 | Warm start + PPO works: s1 100% and cheaper than the teacher on average; s2 82.5% from the first 100k steps. | E-22 |
| 9 | Without the imitation anchor, s2 decays (97.5% → ~80%). The failures are exactly the 6 longest mixture plans; there the agent hoards droplets and dithers until the step limit. | E-22–E-24 |
| 10 | Bound ratios on val sets mostly measure the bound's looseness. Only ratios near 1 certify a plan. | E-17, E-23 |
| 11 | Hoarding is a symptom: a clean chip and twice the steps rescue 0 of 6 long s2 failures. Either the agent will not pay for progress or it is blind off the hint map; not yet separated. | E-25, E-27, D-34 |
| 12 | Headroom above the sharing pass is sparse but large: optimal on 141 of 220 proved small problems, 2–90% off on 12%, mostly tight storage and many targets. Cheap planner variants take much of it; exact search and the agent find what they miss (agent 13–64% cheaper on some single-target problems). | E-28 |
| 13 | Every benchmark problem is one our planner can solve (the generator keeps only those; at our ranges it almost never discards one). At inference the planner enters only through the recipe hint, a fixed per-target map that ignores what is on the chip and reuse across targets. | E-28, D-34, D-35 |

## 3. Where we stand

| Piece | State |
|---|---|
| Env, core, validator, generator, exact search, lower bound | Done (SPEC ≤ 0.22) |
| Sharing pass: teacher, "finish from here" penalty | Done (SPEC 0.24–0.25) |
| Hint columns (10 per candidate) | Single-target recipes and pure-fluid chains; no chip awareness, no reuse across targets |
| Imitation + warm start in PPO | Done (SPEC 0.26) |
| Stages 1–2 agent | s1 100%; s2 85% (peak 97.5%); one seed (`outputs/warm/mixed_s0`) |
| Stages 3–5 | Not trained. Zero-shot: s3 32/40, s4 11/40, s5 16/42, held-out 20/40; many-target mixtures 0/32 (E-26) |
| Proofs | E2 small set 220/399 proved (E-28); val sets 3 proved (s1); medium none |
| Regime choice | Expanding as a stand-in (*default*); pilot 3 after step 3.5 |
| Answer keys | Still the old construction as best known; v3 from E-28 is step 2.1 |
| Paper benchmarks (tiers 2–6) | Not built |

## 4. The plan

### Phase 1. Stages 1–2 (done as far as learning goes; rest parked, D-34)

- **1.1 Diagnose the s2 decay.** *Done (E-23, E-24):* the 6 failures are the 6 longest plans;
  the agent hoards and dithers.
- **1.2 Fix the cause of the hoarding** (D-33). *1.2a done (E-25, E-27): hoarding is a
  symptom; 1.2b (waste charged up front) not adopted.* The open cause is handled in Phase 3
  (3.2) and, if it returns on stages 3–5, Phase 4.
- **1.3 Second seed, 1.4 read cost properly.** Parked; seeds are repeated on the full system
  in 5.1.

### Phase 2. Grading we can trust

**2.0 Headroom above the sharing pass.** *Done 2026-10-27 (E-28):* 155 new proofs (E2 220/399);
the sharing pass is optimal on 141 of them, 2–90% off on 12%; 8 cells meet the rule fixed in
advance, mostly tight storage and many targets; cheap planner variants take much of it, exact
search and the agent find the rest. Large many-target mixtures stay uncertified. Records:
`notes/headroom/records.jsonl`; script `scripts/headroom_share.py`.

**2.1 The bar and the answer keys (now, small, Mac).** (a) Define the **best planner**: per
problem, the cheapest of `share`, `share_look3` and `share_orders` (D-35), as one producer.
(b) Answer keys `_v3`: best known = the cheapest of every producer, the agent's plans and the
proved optima of E-28. Every later grading uses `_v3`, so a run is never compared with a weak
reference.

**2.2 More proofs** (workstation, background, whenever it is free): exact search seeded with
the best planner on the val sets and longer budgets on the E2/E3 timeouts.

**2.3 Certify mixtures.** The bound is loose on many-target mixtures; tighten it or prove more
small ones. Until then those results are intervals.

**2.4 Mini literature comparison on stages 1–2** (D-36; *started 2026-10-27 in its own
session*). Five or six published heuristics that claim single-target dilution or mixture
preparation (candidates on disk: FacDA, Dong 2024, dilution from arbitrary stocks 2023, MRCM,
Kundu 2022 mixture, the generic approach of Roy 2021), implemented from their PDFs and checked
against a worked example or table in each paper. They run on `val_s1_v2`, `val_s2_v2` and the
proved E2 s1/s2 problems, next to the sharing pass, the best planner and the agent (E-22). Each
plan is replayed through our engine. Report: cost under 64/8/1 with sample units, each paper's
own metric, and how many problems each method accepts. It also covers the papers' own example
problems and ~100 fresh s1 and s2 draws (Q-103). Run on the workstation with the E-22 agent,
and run again on the fully trained agent after 5.4 (user, 2026-10-27). It answers early where
we stand against the literature. It isn't the paper's benchmark (that's 6.3), but its implementations
carry over.

### Phase 3. Stages 3–5 (now)

Per stage: (a) is the teacher good there? (b) can the network copy it, with hints? (c) warm
start + PPO with the stages already learned. Never train a stage whose copy test fails.

**3.0 Fix F-16.** *Done 2026-10-27 (SPEC 0.27): `imitation.UidMap`; tests include
`val_s4-033`, which crashed before.* The env offers one move per set of identical droplets; the teacher's
plan names one particular droplet, so when it names the twin the env did not list, nothing
matches and the copy code crashes. Stages 3–5 make identical droplets often (demand). Fix:
match moves by what they do (canonical key), keep a map between the teacher's droplets and
the env's. Tests on identical-droplet cases.

**3.1 Zero-shot check.** *Done (E-26):* s3 32/40, s4 11/40, s5 16/42, held-out 20/40;
single-target dilutions transfer at the teacher's cost; many-target mixtures 0/32.

**3.2 Hint coverage** (*done 2026-10-27, E-30: not blind; the hint map covers the teacher's
route (87–100%) and the planner's next move in failures, but in failed episodes the agent
picks unhinted mixes (17–50% hinted vs 70–98% when solved). Held-out failures pick hinted
mixes and still fail: cross-target information is missing.*) Reading fixed in advance, on the planner's next move from the agent's state: blind
off the map if it is hinted on < 50% of the last-quarter steps of failed episodes (and less
than half as often as in solved ones); will not commit if it is hinted on ≥ 70% of failed
steps but gets median probability < 0.3. Along the s2 failures and along the
teacher's plans on s3–s5: how often the candidates carry any hint, whether the agent's move was
hinted, and the probability it gave the planner's next move. Answers two things at once:
whether the agent gets lost off the hint map (lesson 11), and where 3.3 must add hints.

**3.3 Better hints** (SPEC change; *design after 3.2*). Today's map is fixed per target, from
single-target recipes. Candidates, chosen by what 3.2 shows: recipes from the **best planner**
for many targets (reuse across targets), extending the map on the fly from compositions the
agent actually makes (cached), and the planner's next move from the current state (one
planner call per move: fine on s1–s3, too slow on s4 as it stands).

**3.4 Copy tests, stage by stage** (workstation, ~1 h each): s3, then s4, then s5.
- Teacher = the **best planner** (D-35); exact optima where proved.
- Pass bar (*default*, like stages 1–2 reached): move agreement ≥ 70% and greedy success
  ≥ 60% on the unseen val set. Fail: look at what is missed (hints, teacher, network) before
  any RL.
- Demonstrations are made once, before training, in one process (`train.imitation_demos`,
  1,000 problems). The best planner is up to ~15× slower than the plain pass, so for stages 4–5
  generate them in parallel or cache them before the run.

**3.5 Warm start + PPO** (workstation, ~75 min a run). Expanding regime (*default*, a stand-in
for the user's curriculum): s1–s3, then s1–s4, then s1–s5. One seed.
- Pass bar (*default*): success ≥ 90% on s3 and ≥ 80% on s4–s5; median cost within 5% of the
  best planner; cheaper than the best planner on a real share of the headroom cells (E-28).
- If long plans fail again as on s2 (lesson 11): the options, in order, are search at decision
  time (4.2), progress shaping, a small constant imitation weight; 3.2 says which fits.

**3.6 Regime pilot 3**, once 3.5 passes: sequential, expanding and mixed, all with the warm
start, stages 1–5, 2 seeds each, graded by `regime_report.py`. *The user brings the curriculum
design before it starts.*

### Phase 4. Beat the planner where it is weak

**4.1 Per-cell comparison** against the best planner and, where proved, the optimum; sample
units saved next to cost (D-25).
**4.2 Small MCTS test at decision time** (M9 early): the network's policy as prior, "finish from
here" as leaf value, on the headroom cells against the best planner. Lookahead targets the
planner's known miss (building two intermediates ahead and blending them, E-17).
**4.3 Self-imitation** if cost stalls near the teacher.

### Phase 5. Make it solid

- **5.1** Second and third seeds of the full system.
- **5.2** Generalisation probes (unseen sizes and constraints; benchmark tier 4).
- **5.3** Recipe-hint ablation: copy test and agent without the planner-derived hint (chain
  hint only), to report how much the agent leans on the planner at inference. *Brought
  forward 2026-10-27 (Q-105, E-31):* `hint_source: problem` (SPEC 0.28) is built. A
  planner-free hint covers dilutions but not mixtures. The stage 1–2 copy test, planner hints
  vs problem hints, is the next workstation run.
- **5.4** Full run, 5 stages × 3 seeds, with the regime chosen in 3.6.

### Phase 6. Paper benchmarks and baselines

- **6.1 Paper checks:** robust-aliquot (Poddar 2020), Dong 2024, e-MRCM → claim wording
  (session `paper-checks`).
- **6.2 Tier 2:** DMFB sanity sweep (`dmfb_sweep_v1`).
- **6.3 Tier 3:** published baselines, one set per paper (WSPM, FacDA, MRCM, e-MRCM, MTWM, and
  single-target MEDA dilution methods such as Dong 2024 for stage 1), plus the plan
  translator. Needs the PDFs; SPEC 10.4 forbids rebuilding methods from memory.
- **6.4 Tier 4:** generalisation probes (with 5.2).
- **6.5 Tier 5:** frontier set `frontier_v1`.
- **6.6 Tier 6:** assay ratios from real protocols.

### Phase 7. Paper

Written from `notes/paper_log.md`.

## 5. Waiting on the user (only when the step is reached)

- **3.4 / 3.5 pass bars:** the defaults above stand until changed.
- **3.6:** the curriculum design, before regime pilot 3.
- **Pure RL vs RL + teacher on stages 3–5** (was 3.5, D-31 (6)): still wanted?
- **PDFs** (6.1, 6.3): e-MRCM (Liang et al., TCAD 39(10) 2020; free author copy at
  par.nsf.gov/servlets/purl/10155987) and NFSP (Dinh et al., ASP-DAC 2014) still missing.

## 6. Parked (small, not blocking)

- Stage 2 polish (85% → 95%) and the up-front waste charge (D-33, E-27).
- `regime_report --at-step common` breaks when a run is mid-round (workaround: wait or pass a
  step).
- Tests should cap PyTorch threads themselves; the `fork()` warning in async vector-env tests.
- Attention interpretability, for the paper's analysis section only.
- PopArt or PPG if the shared actor-critic loss ever shows value interference.
