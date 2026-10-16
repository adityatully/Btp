# Training regime across curriculum stages: literature review and experiment design

Date: 2026-10-16. Status: research note, not part of SPEC. Nothing in code, SPEC or configs was
changed. Decision it informs: how M8's PPO set-encoder agent (SPEC 11.4) should be trained across
the curriculum stages (SPEC 8.3) so that it generalises. That includes a later-stage agent still
solving earlier-stage problems.

Scope: the regimes are compared on stages 1–3 only. The mixture stages (4–5) cannot be trained
yet because `NaiveCostToFinish` raises `UnsupportedFallbackError` on mixtures (SPEC 10.3).

---

## 0. Summary

- **Recommendation: regime C, built as an *expanding mixture*.** Keep the current `Curriculum`'s
  promotion test, but a promotion adds the next stage to the sampling distribution instead of
  replacing the current one. Earlier stages keep a fixed floor of the draws (start at 50%, split
  evenly across earlier stages). The main challenger is **D-PLR**: one run over the stage 1–3
  mixture from step 0, with prioritized level replay over instances. The current regime **E**
  stays as the baseline to beat. **A** (one specialist per stage) is the per-stage reference.
- **Why.** The sources agree on three points.
  1. Training only on the newest stage causes forgetting of earlier ones. This was measured in
     neural combinatorial optimisation with curricula over problem size (Lisicki et al. 2020), in
     multi-task RL (Rolnick et al. 2019), and in RL fine-tuning (Wołczyk et al. 2024). E is that
     regime, and it never revisits stage 1.
  2. Rehearsing earlier tasks fixes most of the forgetting at little cost to the new task. CLEAR
     found 50–50 new/replay a good trade-off. Lisicki's adaptive staircase, which rehearses
     smaller sizes on half the trials, gave the best trade-off between forgetting and average
     performance.
  3. An easy-to-hard ordering helps when the hardest distribution is hard to learn directly
     (OpenAI ADR, TSCL, POET). Adaptive sampling within a distribution beats uniform sampling on
     held-out levels (PLR, robust PLR).

  An expanding mixture keeps both the easy-first ordering and the rehearsal. It is also the
  smallest change to the existing `Curriculum`.
- **On generalisation, the evidence is weaker than it looks.** Training on mixed sizes
  preserves performance across the trained range. It did **not** improve extrapolation to larger
  TSPs (Joshi et al. 2022). So no regime choice should be expected to buy out-of-distribution
  generalisation on its own: that has to be measured on separate sets.
- **Two repo facts dominate.** (a) Stages 1 and 2 are tiny finite pools: 31 and 63 distinct
  problems, with 16 left for stage-1 training. Procgen shows that PPO overfits strongly at a few
  hundred levels. (b) The observation divides every volume by M. M is therefore only implicitly
  observable, and the same physical action looks different on an M = 2 chip and an M = 8 chip.
- **On M as a feature: yes, add it (to `globals`), and test it.** The literature supports making
  the varying context observable. It also suggests that two training values of M (2 and 8) will
  not make the agent reliable at an unseen M, so a probe set at M = 4 is needed. This is a
  change to a public interface (the observation), so SPEC 9.2 must change with it.
- **Experiment.** Arms A, E, C-expanding and D-PLR (plus D-uniform as a cheap control), 3 seeds
  minimum and 5 preferred, with matched total environment steps. Two new held-out sets are
  required (`val_stage2_v1`, `val_stage3_v1`), plus a cross-M probe. Every checkpoint is
  evaluated on every set, giving a stage-by-checkpoint forgetting matrix.

---

## 1. What in this repo makes the question specific

Read from SPEC 1, 8.3, 8.4, 9.2–9.3, 10.3, 11.4 and `src/meda/generators/curriculum.py`,
`src/meda/envs/encoding.py`, `configs/generator/*.yaml`.

| Fact | Where | Consequence for the regime choice |
|---|---|---|
| Stage 1 (`tiny_dilution`, M = 2, accuracy 32) has 31 distinct problems: 8 in `val_stage1_v2`, 7 overlap `tiny_exact_v1`, leaving 16 for training. | 8.1, 8.4 | Any regime that spends a long time on stage 1 trains on 16 problems. Procgen-scale overfitting is the expected failure, and stage-1 training success says little about generalisation. |
| Stage 2 (`dilution_single`, M = 8, accuracy 64): one target n/64 with 0 < n < 64, so 63 distinct problems (`dilution_grid` gives the same set). | configs, 8.1 | Also a finite pool. `val_stage2_v1` and any `unseen_ratios` probe both take problems out of these 63. |
| Stage 3 (`dilution_multi`): 1–3 experiments × 1–3 targets, demand 1–3, accuracy 64. The pool is effectively large. | configs | This is the only stage among 1–3 with a large, procedurally generated distribution in the Procgen sense. Its returns can be several times larger than stages 1–2, because cost adds up over up to 9 targets with demand up to 3. |
| Stages 1 and 2 differ in **chip**, not only in problem: M = 2 (1:1 mixing only) vs M = 8. | 8.3, `dmfb_m2.yaml` | Going from stage 1 to stage 2 changes the action space's structure: the number of legal candidates jumps (about 4 vs 28–80; SPEC 16, M6 note). So it is not just a harder instance from the same MDP family. |
| The current `Curriculum` draws **only from the current stage** and promotes forward on a training-episode window. | `curriculum.py` | E is sequential fine-tuning in a single run, with no rehearsal. After promotion, stage-1 problems are never seen again. |
| Volumes are divided by M everywhere: source volume / M, k/M, l/M, (k+l)/M, target `droplet_volume` / M, reagent units / M, waste / M. `globals` = step fraction, storage fraction, remaining demand fraction, F/F_max, T/T_max. | 9.3, `encoding.py` | M is not a feature. It can currently be recovered by accident: every stage has `droplet_volume: 1`, so the target row carries exactly 1/M. If droplet volume ever varies, that stops being true. The same 1:1 mix of single units reads (0.5, 0.5, 1.0) at M = 2 and (0.125, 0.125, 0.25) at M = 8. |
| The candidate scorer scores each candidate from its own features plus the embeddings it references. Candidates do not attend to each other. | 11.4 | This matters for the action-set literature (section 4). |
| Reward is −cost with γ = 1. | 6.4 | Mixed-stage training mixes return scales. See PopArt (section 3.4). |

---

## 2. Comparison table

Evidence strength: **strong** = several peer-reviewed results in closely related settings;
**moderate** = one or two results, or results in a less related domain; **weak** = indirect or a
single small study.

| Regime | Final performance on the hardest stage | Forgetting of earlier stages | Generalisation (held-out / larger / OOD) | Evidence strength | Cost in this repo |
|---|---|---|---|---|---|
| **A** Train each stage from scratch | Fine if the stage is learnable directly. Harder distributions take longer or fail outright (ADR: training time grows with fixed randomisation entropy; TSCL and POET: some tasks unsolvable directly). | Not applicable: one model per stage, so there is no "later agent on earlier problems" at all. | Specialises to its trained size and degrades as the test size moves away (Kool et al.). Fixed-size TSP models generalise poorly outside their size, in both directions (Joshi et al.). | Strong (as a reference) | 3 runs, no code change. |
| **B** Sequential warm start (stage k initialised from k−1's weights) | Often faster to start, but the "transfer" can be lost during fine-tuning (Wołczyk et al.). Warm starting can give worse generalisation than fresh initialisation at similar training loss (Ash & Adams, supervised). It can also leave a lasting memory effect (Igl et al.) and lose plasticity (Nikishin et al.; Dohare et al., over long task sequences). | Severe without retention. Forgetting is the norm in sequential RL (CLEAR, Wołczyk). In NCO size curricula, models that do not revisit earlier sizes decay on them (Lisicki). | No evidence that it helps OOD. | Strong for forgetting; moderate for the plasticity and generalisation harms. | Checkpoint hand-off only. |
| **C** Staged with rehearsal (earlier stages keep being mixed in) | About equal to training on the new task alone. CLEAR at 50–50 shows "no appreciable decrease"; 100% replay costs a little. | Mostly eliminated: at 75–25 new/replay, most but not all forgetting is removed (CLEAR). Adaptive staircase with 50% probe rehearsal gave the best trade-off between forgetting and average performance (Lisicki). | Keeps the earlier-stage range. Extrapolation beyond the trained range is not shown to improve (Joshi). | Moderate to strong | Small change to `Curriculum.source()`. |
| **D** One run over a mixture of all stages, optionally adaptive (PLR, learning progress, PAIRED/ACCEL-style) | Uniform mixing works but can be beaten: it was worse than curricula across TSP sizes (Lisicki), while in TSCL "uniform sampling is surprisingly efficient" on one task. PLR beats uniform on Procgen test return and gets a curriculum for free. | None by construction; every stage is always sampled. | Best evidence of all regimes for held-out levels: PLR improves test return and narrows the generalisation gap. Robust PLR (PLR⊥) improves zero-shot OOD over DR and PLR. Mixed sizes preserve the trained range but do not extrapolate (Joshi). Multi-task training over attributes beats single-task models on unseen variants (Liu et al.). | Strong for held-out levels in the same family; weak for different-M transfer. | Uniform: small. PLR: a level buffer and value-loss scores per instance, which fits the finite pools of stages 1–2. PAIRED/POET: a generator to learn; not justified here (section 3.4). |
| **E** Automatic promotion on training success (current `Curriculum`) | Same as B within one run (shared optimiser), plus a promotion test. Promotion thresholds need prior knowledge of "acceptable performance" or plateau detection (TSCL). | Same as B: earlier stages are never revisited. | Promotion is decided on training draws. For stage 1 that is 16 problems, so promotion can happen by memorisation (Procgen). | Strong that it forgets; the promotion idea itself is sound (ADR uses thresholds). | Built. |

---

## 3. Evidence per regime

### 3.1 A — from scratch, one model per stage
- Kool et al. (2019) train one attention model per TSP/VRP size. On other sizes "the models
  generalize ... although quality degrades as the difference becomes bigger". They conclude the
  models "learn to specialize on the problem sizes trained for", and recommend picking the
  per-size model with the best validation score. That is regime A, and it is honest about being
  an ensemble of specialists.
- Joshi et al. (2022): a TSP20 model "is unable to generalize to large sizes", while "TSP100 models
  generalize poorly to trivially easy sizes". A specialist for a later stage is therefore not
  expected to solve earlier-stage problems. That is exactly the user's "later-stage agent on
  earlier problems" requirement, and A cannot meet it with one model.
- Learning the hard distribution directly can be slow or impossible. ADR: "the larger the fixed
  randomization entropy, the longer it takes to train from scratch" (OpenAI et al. 2019, Fig. 16).
  TSCL solved a Minecraft maze "that could not be solved at all when training directly". POET
  reports environments that "cannot be solved by direct optimization alone". Whether stage 2 or 3
  is like that here is an empirical question. M6 shows a random walk finds the 3/10 optimum at
  M = 8 with probability around 1e-6, but PPO with the naive truncation penalty gets a graded
  signal, so the sparse-reward argument applies only partly.
- **Role in the experiment:** a per-stage reference ("how good can a specialist get on stage k?"),
  not a candidate for the shipped agent.

### 3.2 B — sequential warm start
- Forgetting in RL fine-tuning: Wołczyk et al. (2024) show a pretrained agent "deteriorates on the
  state subspace of the downstream task not visited in the initial phase of fine-tuning". On
  RoboticSequence, "vanilla fine-tuning or training from scratch are virtually indistinguishable".
  Retention methods fix it: behavioural-cloning replay was best, then episodic memory, then EWC.
  For this repo: after a warm start on stage 2, M = 2 states are never visited again, so nothing
  holds stage-1 behaviour in place.
- Warm-start pathologies. Ash & Adams (2020, **supervised**): warm starting "seems to yield poorer
  generalization performance than models that have fresh random initializations, even though the
  final training losses are similar". Shrink-and-perturb is their fix. Igl et al. (2021, RL):
  transient non-stationarity "can permanently impact the latent representation and adversely
  affect generalisation"; their fix is distilling into a fresh network (ITER), which improved
  ProcGen. Nikishin et al. (2022): deep RL agents over-rely on early data ("primacy bias"); fixed
  by periodic partial resets. Dohare et al.: plasticity is lost over long task sequences (an
  ImageNet sequence fell from 89% to 77% by the 2000th task). Caveat: these effects were measured
  over thousands of tasks or long replay-based training. With 3 stages the plasticity risk is
  probably small, and I found no study of it at that scale. The forgetting risk is not small.
- **Role:** E already is B inside one run. B as its own arm adds little; drop it unless E and C
  disagree in a way an optimiser reset would explain.

### 3.3 C — staged training with rehearsal
- Rolnick et al. (2019, CLEAR, DMLab): with 100% new data "standard training ... is subject to
  dramatic catastrophic forgetting". At 75–25 new/replay "there is already significant
  resistance". 100% replay prevents forgetting "at the expense of a (slight) decrease". They pick
  50–50 and found performance "not ... very sensitive to this ratio". Also relevant: training all
  three tasks simultaneously was about as good as separate networks, with "slight constructive
  interference". CLEAR replays stored trajectories with off-policy corrections and BC. With
  on-policy PPO the equivalent is **re-sampling earlier-stage instances**, since the generators
  are cheap. That is task-level rehearsal, not trajectory replay, and it needs no off-policy
  machinery.
- Lisicki et al. (2020), the closest match to this repo's question: an attention-model TSP solver
  trained on sizes in increasing order. Models "that do not visit the previous problems ... show
  initial improvement ... but ... the performance decays for the tasks that are further away". A
  classic curriculum "still results in catastrophic forgetting". Uniform sampling over sizes
  produced an optimality gap "larger than most of the other methods". Adaptive staircase (50%
  base trials at the current level, 50% probe trials uniform over [1, t], advance on a 5% gap
  threshold, step back if missed) "results in the best trade-off between forgetting and average
  performance". Caveats: workshop paper (LMCA @ NeurIPS 2020), one architecture, five runs.
- Matiisen et al. (2017, TSCL) list "continuously mix in easier tasks while learning harder ones
  to avoid forgetting" as a requirement of manual curricula. Their automatic teacher samples by
  the **absolute** learning-progress slope, which "guarantees the re-training on tasks which the
  network is starting to forget".
- OpenAI ADR expands the randomisation ranges when performance at a boundary passes a threshold.
  It keeps sampling the whole range, so in effect it is an expanding mixture with built-in
  rehearsal. It beat every fixed-entropy DR run.

### 3.4 D — single run over a stage mixture, possibly adaptive
- Procgen (Cobbe et al. 2020): "agents strongly overfit to small training sets in almost all
  cases. To close the generalization gap, agents need access to as many as 10,000 levels." Also:
  "training performance improves as the training set grows", attributed to "the implicit
  curriculum provided by the distribution of levels". For this repo: mixing stages enlarges the
  training distribution, which matters most for the 16-problem stage 1.
- PLR (Jiang, Grefenstette, Rocktäschel 2021): replay levels scored by L1 value loss (GAE
  magnitude), with rank prioritisation and a staleness term. On Procgen, rank PLR (β = 0.1,
  ρ = 0.1) gives "the largest statistically significant gains in mean normalized test and train
  returns and reduction in generalization gap compared to uniform sampling". UCB-DrAC + PLR
  reaches +76% test return over PPO with uniform sampling. Gains "only occur when Preplay considers
  both level scores and staleness (0 < ρ < 1)". On MiniGrid it induces "a curriculum that gradually
  assigns more weight to harder levels", and with 6,000 training levels beats uniform sampling
  over about 4 billion. Since stages 1–2 are finite pools, PLR can score every training problem
  directly.
- Robust PLR / PLR⊥ (Jiang, Dennis et al. 2021): update the policy only on replayed levels and
  use fresh levels only to score them. This is "counterintuitive" (less data), yet PLR⊥ beats DR
  and PLR on zero-shot OOD mazes (p < 0.05) at equal gradient updates. Both PLR variants beat
  PAIRED, and "both PLR and PLR⊥ significantly outperform PAIRED after 3 billion training steps"
  well before 250M steps. The authors also note the method cannot help "if the set of
  environments defined by the free parameters does not sufficiently align with the transfer
  domain".
- PAIRED (Dennis et al. 2020) and POET (Wang et al. 2019) learn or evolve a level generator. This
  repo's generators are parameterised by a handful of discrete config fields (stage, number of
  targets, demand, accuracy), and robust PLR finds random-search curation enough in comparable
  domains. **Not recommended now:** high cost, and the evidence above says replay-based curation
  is at least as good.
- Learning-progress teachers (TSCL; ALP-GMM, Portelas et al. 2019) target the stages where
  progress is fastest. ALP-GMM is for continuous parameter spaces; with 3 discrete stages, a TSCL
  bandit over stages is the simpler equivalent. The survey by Portelas et al. (2020) covers these
  as a family.
- NCO with mixed sizes. Joshi et al.: "learning from variable TSP sizes helps models retain
  performance across the range of graph sizes seen during training". Compared with training only
  on the largest size it gives "marginal gains on small instances but ... does not enable better
  generalization to larger problems". Zhou et al. (2023, Omni-VRP) add an easy-to-hard
  hierarchical task scheduler over sizes and distributions, plus meta-learning, for cross-size
  and cross-distribution generalisation. In their ablation table the scheduler alone already
  improves POMO in at least one column; I did not check every column. Manchanda et al. (2022):
  "a one-size-fit-all model is out of reach", so they meta-learn for fast adaptation. Fine-tuning
  at test time is out of scope for M8, so I note it without recommending it.
- Return-scale mismatch. Hessel et al. (2019, PopArt): in multi-task RL "many learning algorithms
  can get distracted by certain tasks" whose rewards are larger in "density or magnitude".
  Adaptive per-task normalisation fixed it for 57 Atari games with one set of weights. Stage 3
  returns are several times stages 1–2, so a mixed run's value loss is dominated by stage 3. PPO's
  per-minibatch advantage normalisation does not fix the value head's scale. **Diagnostic to log:**
  value explained variance per stage. If stage 1–2 explained variance collapses in mixed runs,
  per-stage return normalisation is the fix PopArt suggests.

### 3.5 E — automatic promotion on training success (current)
- Promotion on a success window is the "mastery threshold" approach that TSCL criticises: it
  "requires prior knowledge of acceptable performance" or plateau detection, which "can be hard to
  detect given the noise in the learning curve". ADR uses the same idea but keeps the old range in
  the distribution.
- In this repo the window is measured on training draws. On stage 1 these are 16 problems, so
  promotion can be earned by memorisation (Procgen's "illusion of meaningful progress" warning
  about deterministic level sequences is the extreme case).
- After promotion E is sequential fine-tuning, so the forgetting evidence of 3.2 applies in full.
  The literature does not support E as the final regime. Its promotion rule is still the right
  mechanism for deciding **when** to widen an expanding mixture (C).

---

## 4. Does the agent need M as a feature? Transfer across action-space structure

**What the agent can see now.** Every volume is a fraction of M, and the legal candidate list
encodes M's effect (no candidate has k + l > M). M is recoverable from the target row
(`droplet_volume / M` = 1/M, because every stage uses droplet volume 1) and from the smallest
k/M among candidates. So M is observable, but only through an accident of config.

**What the literature says.**
1. Contextual MDPs (Kirk et al. 2023 survey): a varying context must be observed for the problem
   to be an MDP: "The context needs to be observed for the CMDP to be an MDP (and not a POMDP)".
   M is part of the context; the transition and the legal set depend on it.
2. Multi-task routing with varying constraints (Liu et al. 2024): one model across VRP variants.
   Each constraint ("attribute") enters through **both** an explicit attribute vector in the
   decoder context **and** a feasibility mask. The unified model beats single-task POMO models on
   unseen attribute combinations zero-shot. Single-task models fail most "in problems with very
   different attributes". This is the closest analogue to masking plus an M feature. It supports
   using both, rather than relying on the mask alone.
3. Varying action sets (Jain et al. 2020; Jain et al. 2022, ICLR): standard RL "assumes a fixed
   action set". When the available set varies, the best choice depends on relations among the
   available actions, and a policy that attends over the action set beats non-relational
   architectures. Here the candidate scorer scores candidates independently, given the set
   encoder's source and target embeddings. That is fine within one stage, but the candidate set's
   size and shape change sharply between M = 2 and M = 8. *(Inference, not tested in any source
   I read.)*
4. Features whose statistics shift between training and test hurt extrapolation (Joshi et al.).
   SUM aggregation and learned BatchNorm statistics blew up on larger graphs; MEAN/MAX and
   LayerNorm did not. The parallel here is the volume normalisation: dividing by M keeps features
   in [0, 1] (good) but changes what a value means between stages. One unit of sample is 0.5 at
   M = 2 and 0.125 at M = 8.
5. MEDA routing (Liang et al. 2021) trains one router reused "for many fabricated MEDA biochips"
   across chip sizes. That is a different problem (routing on an image of the array), and they do
   not study transfer across mixing limits.

**Conclusion.**
- **Add M explicitly.** Append `M / M_cap` (or `1/M`) to `globals`, where `M_cap` is a config
  value, so the feature stays in [0, 1]. The precedent is SPEC 6.5, which already appends the
  preference weights to `globals`. It is cheap, removes the dependence on `droplet_volume = 1`,
  and matches source 2. It changes the observation, a public interface, so SPEC 9.2/9.3 must be
  updated in the same change (CLAUDE.md rule).
- **Don't expect it to make M transfer work.** With M ∈ {2, 8} only, the network sees two
  contexts. Nothing I read suggests reliable interpolation to M = 4 from two points; ADR and
  domain-randomisation work randomises the context over a range for exactly this reason. If
  cross-M transfer matters, the training distribution must vary M, e.g. run stage-2 dilutions on
  M ∈ {2, 4, 8}. That is a generator/chip-config change and a separate decision.
- **Measure it:** an M = 4 probe set (section 5.2) and a feature-on/off ablation (5.1). A
  candidate-set attention layer (source 3) is a possible later ablation and belongs to M8's
  network design, not to this decision.

---

## 5. Experiment design to settle the regime in this repo

### 5.1 Arms (stages 1–3, PPO set encoder with SPEC 11.4 defaults)

| Arm | Definition | Purpose |
|---|---|---|
| A1, A2, A3 | Three separate runs from scratch on one stage each, budget T per stage | Specialist reference per stage. Gives "how good can stage k get" and the per-stage optimum of training time. |
| E | Current `Curriculum` (stages 1→2→3, promote on success window), budget T total | Baseline (= B inside one run). |
| C-exp | Same promotion rule; on promotion the source becomes a mixture: new stage 50%, earlier stages 50% split evenly. Budget T. | Recommended regime. |
| D-unif | Uniform over stages 1–3 from step 0. Budget T. | Cheap control: does ordering matter at all? |
| D-PLR | Mixture over stages 1–3 with PLR over instances: L1 value-loss score, rank β = 0.1, staleness ρ = 0.1–0.3, replay probability 0.5. Stages 1–2 score their whole training pools; stage 3 keeps a buffer of sampled instances. Budget T. | Main challenger; strongest evidence for held-out levels. |
| M-feature ablation | Best of {C-exp, D-PLR}, with and without `M / M_cap` in `globals` | Answers section 4. |

Optional, only if E vs C-exp is ambiguous: B (separate warm-started runs with a fresh optimiser)
and a 75–25 rehearsal ratio (CLEAR's sensitivity point).

Controls: the same total environment steps T for every single-run arm. The A arms get T each,
which makes them a generous reference. Same seeds and the same `exclude` hashes in every arm
(the hashes of `tiny_exact_v1`, `val_stage1_v2`, `val_stage2_v1`, `val_stage3_v1` and every
probe set). At least 3 training seeds per arm (SPEC 12.3); 5 recommended, because the stage-1
and stage-2 validation sets are small and seed variance in RL is large.

### 5.2 Held-out sets needed (tier 1 unless stated)

| Set | Status | Content and how to build it |
|---|---|---|
| `val_stage1_v2` | exists | 8 stage-1 problems: 3 proved optima, 5 intervals. |
| `val_stage2_v1` | **new** | About 16 of the 63 stage-2 targets n/64, stratified by 2-adic level (n odd, n ≡ 2 mod 4, and so on: how many mixes a naive plan needs), so easy and hard ratios are both held out. Answer key built like `val_stage1_v2`: exact search with node budgets at the instance's accuracy and at the matched level, giving an optimum or [lower bound, best known]. |
| `val_stage3_v1` | **new** | 50–100 draws from `dilution_multi`, seeded and frozen, spread over the number of targets and total demand. Mostly intervals [LB, best known] with best known = min(naive, budgeted exact search). Report `bound_ratio` so loose instances are visible. |
| `unseen_m4_probe` (tier 4) | **new**, proposed | Stage-2-style single dilutions on `meda_m4` (M = 4, accuracy 16 or 64; check factorisability per SPEC 8.3). Measures cross-M transfer. A cheaper first version: stage-1 ratios (n/32) on an M = 8 chip, to check whether a stage-2-and-later agent still solves "the same ratios" on the new chip. |
| `larger_v1` (tier 4, already planned in 8.4) | new | Stage-3 instances with more targets or demand than training (for example 4 experiments, demand up to 4), within `T_max = 12` padding. |

`unseen_ratios_v1` overlaps `val_stage2_v1` for single dilutions. For stage 3 it can be built by
excluding a set of target ratios from all training draws. That needs a ratio-level exclusion,
not just instance hashes; `exclude` works on whole problems today.

### 5.3 Metrics (from the M7 harness, `evaluate` / `summary.md`)
Per set, per arm, over seeds (mean and bootstrap CI over seeds):
- **Success rate.** Greedy decoding, and best-of-k sampled.
- **`gap_to_best_known`**: mean and median over successful plans. It equals `gap_to_optimal` on
  proved instances; report `gap_to_optimal` separately on those.
- **`gap_to_lower_bound`**: the most the true gap can be, on interval instances.
- **`bound_ratio`**: to flag instances that cannot separate arms.
- Dead-end rate, and plan metrics (`mix_ops`, `sharing_count` on stage 3).
- **Forgetting matrix.** Evaluate each arm's checkpoints every N updates on *all* sets.
  Forgetting on set k = best success (or gap) reached during training − final value. Backward
  transfer for the later-stage agent = the final checkpoint's numbers on `val_stage1_v2` and
  `val_stage2_v1`.
- **Training diagnostics per stage**: value explained variance (the PopArt concern) and the
  stage mix actually sampled (C-exp, D-PLR).

### 5.4 Decision rule (fix before running)
1. Reject any arm whose final checkpoint has success < 95% on any of `val_stage1_v2`,
   `val_stage2_v1`, `val_stage3_v1`. This is the M8 bar extended to every stage.
2. Among the rest, pick the lowest mean `gap_to_best_known` averaged with equal weight over the
   three sets. If two arms' CIs overlap, prefer the simpler one: D-unif < C-exp < D-PLR.
3. Tie-breakers: lower forgetting, then better `larger_v1` and cross-M probe numbers.
4. Compare with A: if the chosen single model is more than a few points of gap worse than A_k on
   stage k, record the cost of generality rather than hiding it.

### 5.5 What I expect (to be falsified by the run)
- E forgets stage 1, and probably stage 2, by the end. Strongly supported by Lisicki, CLEAR and
  Wołczyk.
- C-exp and D-PLR both retain earlier stages. D-PLR may beat C-exp on `val_stage3_v1`.
  D-unif may lag on stage 3 if stage 3 is hard to learn from scratch (ADR, TSCL) or be
  competitive (TSCL found uniform "surprisingly efficient").
- No regime extrapolates well to `larger_v1` or the M = 4 probe without varying M or size in
  training (Joshi; Kool).

### 5.6 Prerequisites (no code was written for this note)
- `Curriculum` currently returns a single-stage source. C-exp needs a mixture source; D-PLR needs
  a level buffer and per-instance scores fed from rollouts. SPEC 8.3 describes promotion as
  replacing the stage ("a syllabus, not a controller"), so implementing C-exp needs a SPEC
  change.
- Mixed stages in one vectorised env need one padding config covering stages 1–3. `L_max` is
  2048 and shared already.
- Building `val_stage2_v1` and `val_stage3_v1` needs `make_benchmark.py` builders like
  `build_val_stage1_v2`. Stage-3 exact search will mostly time out, so the stored references will
  be intervals.

---

## 6. Sources

All read in this session. Titles and authors were checked against each paper's header. Venues
that the arXiv header does not print (for example ICML or NeurIPS for Procgen, PLR, CLEAR or
Joshi et al.) are from memory and unverified; the arXiv links are the checked identifiers.
"Full text" means I extracted the paper's text and read the passages
cited above; "abstract" means only the abstract or landing page.

Curriculum, UED and generalisation in procedurally generated environments
- Narvekar, Peng, Leonetti, Sinapov, Taylor, Stone. *Curriculum Learning for Reinforcement
  Learning Domains: A Framework and Survey.* JMLR 21(181), 2020. https://arxiv.org/abs/2003.04960
  (abstract; used only for framing)
- Cobbe, Hesse, Hilton, Schulman. *Leveraging Procedural Generation to Benchmark Reinforcement
  Learning.* ICML 2020. https://arxiv.org/abs/1912.01588 (full text)
- Jiang, Grefenstette, Rocktäschel. *Prioritized Level Replay.* ICML 2021.
  https://arxiv.org/abs/2010.03934 (full text)
- Jiang, Dennis, Parker-Holder, Foerster, Grefenstette, Rocktäschel. *Replay-Guided Adversarial
  Environment Design.* NeurIPS 2021. https://arxiv.org/abs/2110.02439 (full text)
- Dennis, Jaques, Vinitsky, Bayen, Russell, Critch, Levine. *Emergent Complexity and Zero-shot
  Transfer via Unsupervised Environment Design.* NeurIPS 2020. https://arxiv.org/abs/2012.02096
  (abstract, plus robust PLR's comparison with it)
- Wang, Lehman, Clune, Stanley. *Paired Open-Ended Trailblazer (POET).* 2019.
  https://arxiv.org/abs/1901.01753 (abstract)
- Matiisen, Oliver, Cohen, Schulman. *Teacher-Student Curriculum Learning.* 2017 (IEEE TNNLS
  2019). https://arxiv.org/abs/1707.00183 (full text)
- Portelas, Colas, Hofmann, Oudeyer. *Teacher algorithms for curriculum learning of Deep RL in
  continuously parameterized environments* (ALP-GMM). CoRL 2019.
  https://arxiv.org/abs/1910.07224 (abstract)
- Portelas, Colas, Weng, Hofmann, Oudeyer. *Automatic Curriculum Learning For Deep RL: A Short
  Survey.* IJCAI 2020. https://arxiv.org/abs/2003.04664 (skimmed)
- OpenAI (Akkaya, Andrychowicz, et al.). *Solving Rubik's Cube with a Robot Hand* (ADR). 2019.
  https://arxiv.org/abs/1910.07113 (full text, sections 5 and 8.2)
- Kirk, Zhang, Grefenstette, Rocktäschel. *A Survey of Zero-shot Generalisation in Deep
  Reinforcement Learning.* JAIR 76, 2023. https://arxiv.org/abs/2111.09794 (CMDP section)

Continual learning, forgetting, warm start
- Rolnick, Ahuja, Schwarz, Lillicrap, Wayne. *Experience Replay for Continual Learning* (CLEAR).
  NeurIPS 2019. https://arxiv.org/abs/1811.11682 (full text)
- Kirkpatrick et al. *Overcoming catastrophic forgetting in neural networks* (EWC). PNAS 2017.
  https://arxiv.org/abs/1612.00796 (skimmed; cited only as the EWC method Wołczyk compares)
- Wołczyk, Cupiał, Ostaszewski, Bortkiewicz, Zając, Pascanu, Kuciński, Miłoś. *Fine-tuning
  Reinforcement Learning Models is Secretly a Forgetting Mitigation Problem.* ICML 2024.
  https://arxiv.org/abs/2402.02868 (full text). *Venue from the arXiv v3 header; not otherwise
  checked.*
- Ash, Adams. *On Warm-Starting Neural Network Training.* NeurIPS 2020.
  https://arxiv.org/abs/1910.08475 (abstract and method; supervised setting)
- Igl, Farquhar, Luketina, Boehmer, Whiteson. *Transient Non-Stationarity and Generalisation in
  Deep Reinforcement Learning* (ITER). ICLR 2021. https://arxiv.org/abs/2006.05826 (abstract)
- Nikishin, Schwarzer, D'Oro, Bacon, Courville. *The Primacy Bias in Deep Reinforcement
  Learning.* ICML 2022. https://arxiv.org/abs/2205.07802 (abstract)
- Dohare, Hernandez-Garcia, Rahman, Mahmood, Sutton. *Maintaining Plasticity in Deep Continual
  Learning* (arXiv title of the version read). https://arxiv.org/abs/2306.13812 (abstract).
  *Uncertain: I believe a later version appeared in Nature (2024) as "Loss of plasticity in deep
  continual learning"; I did not verify this.*
- Hessel, Soyer, Espeholt, Czarnecki, Schmitt, van Hasselt. *Multi-task Deep Reinforcement
  Learning with PopArt.* AAAI 2019. https://arxiv.org/abs/1809.04474 (abstract)

Neural combinatorial optimisation and size generalisation
- Kool, van Hoof, Welling. *Attention, Learn to Solve Routing Problems!* ICLR 2019.
  https://arxiv.org/abs/1803.08475 (generalisation appendix)
- Joshi, Cappart, Rousseau, Laurent. *Learning the Travelling Salesperson Problem Requires
  Rethinking Generalization.* Constraints 2022 (CP 2021). https://arxiv.org/abs/2006.07054 (full
  text, sections on variable sizes and aggregation/normalisation)
- Lisicki, Afkanpour, Taylor. *Evaluating Curriculum Learning Strategies in Neural Combinatorial
  Optimization.* LMCA workshop @ NeurIPS 2020. https://arxiv.org/abs/2011.06188 (full text)
- Kwon et al. *POMO: Policy Optimization with Multiple Optima for Reinforcement Learning.*
  NeurIPS 2020. https://arxiv.org/abs/2010.16011 (title page only; referenced via Omni-VRP and
  Liu et al.)
- Zhou, Wu, Song, Cao, Zhang. *Towards Omni-generalizable Neural Methods for Vehicle Routing
  Problems.* ICML 2023. https://arxiv.org/abs/2305.19587 (abstract, scheduler section, partial
  ablation)
- Manchanda, Michel, Drakulic, Andreoli. *On the Generalization of Neural Combinatorial
  Optimization Heuristics.* ECML-PKDD 2022. https://arxiv.org/abs/2206.00787 (abstract)
- Liu, Lin, Wang, Zhang, et al. *Multi-Task Learning for Routing Problem with Cross-Problem
  Zero-Shot Generalization.* 2024. https://arxiv.org/abs/2402.16891 (full text, model and
  zero-shot sections). *Only the first four authors were checked against the extracted header;
  the venue (KDD 2024, I believe) was not checked.*

Action-set structure
- Jain, Szot, Lim. *Generalization to New Actions in Reinforcement Learning.* ICML 2020.
  https://arxiv.org/abs/2011.01928 (abstract)
- Jain, Kosaka, Kim, Lim. *Know Your Action Set: Learning Action Relations for Reinforcement
  Learning.* ICLR 2022. https://www.iclr.cc/virtual/2022/poster/6594 (abstract only; PDF not
  retrieved)

Microfluidic biochips
- Liang, Zhou, Chan, Ho, Chakrabarty, Lee. *Parallel Droplet Control in MEDA Biochips Using
  Multi-Agent Reinforcement Learning.* ICML 2021. https://proceedings.mlr.press/v139/liang21c.html
  (full text, observation and conclusion)
- Elfar, Liang, Chakrabarty, Pajic. *Adaptive Droplet Routing for MEDA Biochips via Deep
  Reinforcement Learning.* DATE 2022.
  https://past.date-conference.com/proceedings-archive/2022/pdf/0641.pdf (title and abstract)
- **Gap found:** an extended search turned up no paper applying deep RL to sample preparation
  (mixing-tree or dilution synthesis) on DMFB or MEDA. RL on these chips is routing and bioassay
  synthesis. One unread lead: *Reinforcement Learning-based Reliable Synthesis of Bioassays on
  MEDA Digital Microfluidic Biochips*, ACM, https://dl.acm.org/doi/10.1145/3773910 (title only;
  scope not checked). No regime claim in this note rests on biochip literature.

### Limits of this review
- No source tests regime choice on a problem where the **chip** (action structure) changes
  between stages, as it does between stages 1 and 2. The M recommendations combine CMDP theory,
  the multi-task routing result and the varying-action-set work; they are judgment, not a
  measured result.
- The NCO curriculum evidence (Lisicki) is one workshop paper. The forgetting conclusion is
  corroborated elsewhere; the "adaptive staircase is best" conclusion is not.
- The warm-start and plasticity results come from long task sequences or supervised learning.
  Their size at 3 stages is unknown.
