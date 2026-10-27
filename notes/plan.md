# Plan

Where the project stands and what comes next, step by step. Rebuilt from scratch on 2026-10-26
from everything learned up to E-23. `SPEC.md` is the design; `notes/paper_log.md` holds the
evidence (IDs like E-22 point there). This file says **what we are doing and why, in order**.

**How to maintain it**
- Read it at the start of every session. Update the status of a step the day it changes, with
  the paper log ID that backs it.
- When a result changes the plan, edit the plan here and record the decision in the paper log
  (D-…). Keep finished steps, marked done, so the path stays visible.
- Each step names its pass bar and what happens on pass and on fail. No step starts before the
  step it depends on has passed.
- Compute: the Mac runs only light jobs at background priority (`OMP_NUM_THREADS=1 taskpolicy
  -b nice -n 19`). Anything longer than ~10 minutes goes to the workstation (`~/code/Btp`,
  tmux; never reboot it during runs).

*Last updated 2026-10-27 (E-28, SPEC 0.26).*

---

## 1. Goal

One RL policy that plans MEDA sample preparation (k:l mixing, any M, F, targets, demand and
storage) and produces near-best-known plans for **MTMD mixture preparation with a storage
limit**, a class no published method handles, while matching specialised methods on the
sub-classes they handle (claim fixed 2026-10-21, paper log §1.2). Generalisation across sizes is
a requirement of the agent, not the headline.

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
| 9 | Without the imitation anchor, s2 decays (97.5% → ~80%). The failures are exactly the 6 longest mixture plans; there the agent hoards droplets instead of discarding and dithers until the step limit. | E-22–E-24 |
| 10 | Bound ratios on val sets mostly measure the bound's looseness (the teacher is at the optimum on every small proved s1 problem yet 1.17× the val_s1 bound). Only ratios near 1 certify a plan. | E-17, E-23 |

## 3. Where we stand

| Piece | State |
|---|---|
| Env, core, validator, generator, exact search, lower bound | Done (SPEC ≤ 0.22) |
| Sharing pass: teacher, "finish from here" penalty | Done (SPEC 0.24–0.25) |
| Hint columns (10 per candidate) | Done for single targets; cross-target sharing not covered |
| Imitation + warm start in PPO | Done (SPEC 0.26) |
| Stages 1–2 agent | s1 100%; s2 85% at the end (peak 97.5%); one seed (`outputs/warm/mixed_s0`) |
| Stages 3–5 | Not trained. Zero-shot on the held-out cell: 50% |
| Regime choice | Parked until every stage can be learned |
| Answer keys | Best known = old construction; 3/40 s1 problems proved optimal; no published baselines yet |
| Paper benchmarks (tiers 2–6) | Not built |

## 4. The plan

### Phase 1. Make stages 1–2 solid (paused for 2.0, D-34)

**1.1 Diagnose the s2 decay.** *Done (E-23, E-24).*
- The failures are the 6 longest plans; ~11% of training s2 problems are that long (E-23).
- Replay (E-24): all truncated with one droplet left, no loops; the agent hoards droplets
  (10–25 on chip, 0–3 discards where the teacher discards 7–18 times) and dithers with low
  confidence. The `share` failure penalty does not charge leftovers, so hoarding is free in
  failed episodes.

**1.2 Fix the cause of the hoarding** (D-33). Goal: not a higher s2 score for its own sake, but
removing a failure mode that will hit stages 3–5 harder.
- **1.2a Evidence first** (*done, E-25, E-27*). **Result: hoarding is a symptom.** A clean chip
  and twice the steps rescue 0 of 6; the agent does not take the costly steps that make
  progress on long mixtures (it ends having spent less than the teacher's whole plan). So 1.2b
  is **not** adopted as the fix; the next decision is how to make the agent commit on long
  horizons (see the open decision under §5).
  Earlier notes: RL un-learned discarding
  (the teacher's discards get probability 0.00–0.04 from the RL agent, 0.85–0.96 from the copy
  network); the uncharged leftovers are 2–12% of the penalty; the first janitor test was
  inconclusive, so a fair one (step limit ×2, janitor on/off) is next.
  Original design: experiments on the workstation with the trained agent:
  D1 discard behaviour (teacher vs copy network vs RL agent); D2 the probability each network
  gives the teacher's discards on the teacher's own states (did RL learn to avoid discarding?);
  D3 a "janitor" that clears unused droplets above the teacher's peak (does clearing the chip
  rescue the failures?); D4 how much leftover waste the failure penalty leaves uncharged.
  - If D2 shows RL learned to avoid discards and D3 rescues the failures: build 1.2b.
  - If D4 is large: try the smaller fix first (charge leftovers in the failure penalty only).
  - If D3 does not rescue them: hoarding is a symptom, not the cause; look at long-horizon
    credit and search (Phase 4).
- **1.2b Waste charged up front** (SPEC change, needs the user's OK after 1.2a): every unit is
  charged as waste when it lands on the chip and refunded when delivered (potential-based
  shaping; plan costs unchanged), and failure penalties charge leftovers.
- **1.2c Verify:** one stages 1–2 run (~75 min); the replay must show discards near the
  teacher's rate and no pile-up; s2 holding near its peak is a bonus.
- Fallbacks: constant imitation anchor (dropped for now, D-33); self-imitation.

**1.3 Second seed.** Repeat the fixed run with seed 1. Pass: the same within one grading's
noise (about ±6 points on 40 problems).

**1.4 Read cost properly.** Per problem against the teacher and against the bound, including
the problems where the agent is dearer (14/40 on s1 in E-22), to see what RL learned and what
it gave up.

### Phase 2. Grading we can trust (mostly workstation)

**2.0 Headroom: where can the sharing pass be beaten, and by how much?** (*done 2026-10-27,
E-28:* headroom is sparse but large. Share is optimal on 141 of 220 proved small problems; 12%
are 2–90% off, mostly tight storage and many targets; 8 cells meet the rule. Cheap planner
variants (more orders, lookahead) take much of it; exact search and the agent find what they
do not, the agent with 13–64% gains on single-target s1–s3. Large many-target mixtures stay
uncertified. Next: inspect the records, then choose the follow-up with the user. The
sharing pass already solves every problem of every stage and is at the optimum wherever we
have proofs (51/65). Learning or search is only worth building where it can be beaten, so this
comes before any further agent work.
- **H1. Proofs with a strong incumbent.** Exact search seeded with the cheaper of the sharing
  pass and its lookahead as the incumbent (branch and bound then prunes far more than with the
  constructions), 600 s per problem, on all 400 small E2 problems (65 proved so far;
  many-target mixtures only 3 of 80) and the 45 medium E3 problems. Each new proof gives the
  sharing pass's true gap; any cheaper plan the search finds is headroom outright.
- **H2. Stronger producers, without proofs.** On `val_s1`…`val_s5`, held-out and E3: the
  sharing pass with lookahead 3, with many target orders (its 4 plus 32 random ones), and the
  agent's own plans (E-22 on s1/s2, E-26 on s3–s5). Headroom found = sharing pass ÷ best − 1,
  also in sample units saved (D-25).
- **H3. Where nothing proves or improves:** the sharing pass's excess over the bound split into
  waste, mixes and fluid; reported as an interval, no claim.
- **Output:** one table per cell: n, proved, sharing pass at the optimum, its gap on proved
  problems (median, max), problems H2 improves (count, median, max, sample units), its ratio
  to the bound.
- **Reading, fixed in advance:** a cell has headroom if the sharing pass's gap on proved
  problems averages ≥ 5% or exceeds 2% on a quarter of them, or if H2 lowers its median cost
  by ≥ 5%. Learning and search are then aimed at those cells, starting with the small MCTS
  test (4.2). If no cell we can check has headroom, the paper's story changes (*user's
  decision*): the sharing pass with exact certificates becomes a main contribution, and the
  agent must justify itself by speed or generalisation, or be dropped.
- **Caveats:** proofs reach only small problems (selection bias); large many-target mixtures
  may stay uncertified (2.3).
- **Cost:** workstation, about 2–3 h on 30 cores; nothing heavy on the Mac. Script-side only:
  a subclass of `ExactSearch` with the sharing pass as incumbent and extra producers in
  `scripts/teacher_study.py`; no change to `meda.core` or solver interfaces.

**2.1 Best known = the best of every producer.** Old construction, sharing pass, its lookahead,
greedy start + finish (E-19), and the agent's own plans. Frozen as new answer-key versions
(`_v3`), so a run is never graded against a weak reference. Needs: the lookahead run from E-17
(workstation command there).

**2.2 More proved optima.** Longer exact search on the workstation for the small and val
problems; every proof turns an interval into a true gap. Stage 1 has only 3/40 now.

**2.3 Certify mixtures.** The bound is loose on mixtures and many targets; either tighten it or
prove optima on small many-target mixture problems. Until then, mixture results are reported
as intervals.

### Phase 3. Stages 3–5, one constraint at a time

The same recipe that worked for stages 1–2, per stage:
(a) is the teacher good there? (b) can the network copy it, with hints? (c) warm start + PPO
on that stage together with the stages already learned. Never train a stage whose copy test
fails.

**3.1 Zero-shot check.** *Done (E-26): s3 32/40, s4 11/40, s5 16/42, held-out 20/40.
Single-target dilutions transfer at the teacher's cost; every many-target mixture cell is 0/32;
many-target dilutions cost 1.4–1.9× the teacher (no sharing).* Grade the stage 1–2 agent on `val_s3`, `val_s4`, `val_s5` and the
held-out cell (cost included). Cheap; shows how much already transfers (held-out: 50%).

**3.0 Before any copy test on stages 3–5: fix F-16** (teacher moves matched by droplet uid).

**3.2 Stage 3 (demand).** Teacher near-optimal on small s3 (E-17). Copy test on s3; if it
passes, warm start on s1–s3.

**3.3 Stage 4 (many targets).** The hints do not cover reuse across targets (162 of 479
teacher mixes on s4 get no hint). Design sharing-aware hints (SPEC change), copy test, then
train. This is the stage where the paper's advantage lives.

**3.4 Stage 5 (storage).** Check hints and teacher under tight storage, copy test, train.

**3.5 Pure RL vs RL + teacher on 3–5** (D-31 (6)). The worry behind it was a mediocre
teacher; the sharing pass is now near-optimal wherever we can check. *User's decision whether
it is still needed.*

### Phase 4. Beat the teacher (when cost stalls near it)

**4.1 Self-imitation** (if not already used in 1.2).
**4.2 Small MCTS test at decision time** (M9 early): the network's policy as prior, "finish from
here" as leaf value, on the val sets against the teacher. If search finds cheaper plans,
AlphaZero-style training follows.

### Phase 5. Full training

**5.1 Curriculum and regime.** Re-ask the regime question only once every stage can be learned;
*the curriculum design is the user's.*
**5.2 Full run.** 5 stages × 3 seeds (workstation or A100).

### Phase 6. Paper benchmarks and baselines

- **6.1 Paper checks:** robust-aliquot (Poddar 2020), Dong 2024, e-MRCM PDFs →
  claim wording. Waiting on the PDFs (session `paper-checks`).
- **6.2 Tier 2:** DMFB sanity sweep (`dmfb_sweep_v1`).
- **6.3 Tier 3:** published baselines, one set per paper (WSPM, FacDA, MRCM, e-MRCM, MTWM, and
  single-target MEDA dilution methods such as Dong 2024 for stage 1), plus the plan
  translator. Needs the PDFs; SPEC 10.4 forbids rebuilding methods from memory.
- **6.4 Tier 4:** generalisation probes (unseen M, F, demand, storage).
- **6.5 Tier 5:** frontier set `frontier_v1`.
- **6.6 Tier 6:** assay ratios from real protocols.

### Phase 7. Paper

Written from `notes/paper_log.md`.

## 5. Waiting on the user

- **How to make the agent finish long mixtures (blocks 1.2 and Phase 3).** *Deferred until
  the headroom check (2.0, D-34).* Options: progress shaping, a small constant imitation
  weight, search at decision time (MCTS, 4.2). Two readings of the cause, not yet separated:
  the agent will not pay for progress (E-27), or it is blind off the hint map (D-34).

- PDFs (6.1, 6.3). **Added 2026-10-26:** robust aliquot (Poddar et al.), Dong et al. 2024,
  MRCM (Liang et al., ASP-DAC 2019), MTCS/MTMD (Kumar et al., TODAES 2018); the claim check is
  running in session `paper-checks`. **Still missing:** e-MRCM (Liang et al., TCAD 39(10) 2020;
  free author copy at par.nsf.gov/servlets/purl/10155987) and NFSP (Dinh et al., ASP-DAC
  2014).
- 3.5: is the pure-RL comparison on stages 3–5 still wanted?
- 5.1: curriculum and regime design.

## 6. Parked (small, not blocking)

- `regime_report --at-step common` breaks when a run is mid-round (workaround: wait or pass a
  step).
- Tests should cap PyTorch threads themselves; the `fork()` warning in async vector-env tests.
- Attention interpretability, for the paper's analysis section only.
- PopArt or PPG if the shared actor-critic loss ever shows value interference.
