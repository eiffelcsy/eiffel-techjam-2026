# Results ledger

What has actually been run, what it showed, and what is still open. One row per
experiment. `docs/EXPERIMENTS.md` says how to run each one; this says what came
back.

Last updated 2026-08-29. 319 tests pass (293 `grace_adapter`, 26 `eval_pipeline`).

---

## 0. Round 2 — the restart, and what it is predicted to find

**Everything below section 1 is round 1.** Its checkpoints are gone, its config
set has been consolidated, and the experiment list is being run again from a
clean tree by `grace_adapter/scripts/run_all.sh`. Round 1's numbers are kept
here as **priors**, not as records: they say what to expect, and where a
surprise would be informative.

### What changed between the rounds, and why

| change | reason |
|---|---|
| **One reference arm.** `dinov3_clean.yaml` now carries the tuned loss weights (`w_err` 4.0 / `w_cos` 0.25), and every stage-1 ablation is that file with exactly one key changed. | Round 1 had two near-identical canonical configs (`dinov3_clean` at ratio 4, `dinov3_clean_final` at ratio 16) and the ablations hung off *both*. E2, E3 and E8 were controlled against ratio 4 while E9 and E7 were controlled against ratio 16, so no single number was the reference. |
| **Redundant configs deleted.** `dinov3_clean_final`, `dinov3_sweep_wratio_16`, `dinov3_clean_v1.2`, `dinov3_clean_v1.4`, `dinov3_ladder`. | Each was either key-for-key identical to another config or an off-axis point on no sweep. `wratio_16` and `clean_final` were the same run under two names, which round 1 nearly reported as two arms. |
| **Sweep configs renamed to their axis.** `dinov3_clean_v1.1` → `dinov3_sweep_nblocks_2`, `v1.3` → `dinov3_sweep_bottleneck_256`; added `dinov3_sweep_wratio_4`. | A config's name should say which key it moves. `v1.1` does not. |
| **Broken checkpoint paths repointed.** `dinov3+grace-ladder` and both ladder stage-2 configs pointed at run ids that never existed (`dinov3_ladder`, `dinov3_ladder_e4`). | This is why round 1 produced no harness numbers at all (§5). |
| **E1 moved out of `dinov3_poc_grace.yaml`** into its own run config, scored right after the baseline, with `compare.py --assert-identity` as a hard gate. | E1 gates everything, and in round 1 it could not run until *everything it gates* had already been trained. |
| **S0, the seed floor, is now a numbered step** rather than an aside inside E9. | Every ablation verdict is quoted in seed-sd. It is not an accessory to E9; it is the unit. |
| **One script covers every arm.** `run_all.sh` replaces `poc.sh`, which skipped D1, E3, E4 and E6. | "Extra arms you run by hand" is how arms go unrun. |

### Pre-registered predictions

Written before round 2 is executed, so that a surprise is legible as a surprise.

| # | prediction | confidence | what would be surprising |
|---|---|---|---|
| **P2** | retention collapses on WildFake far more than the 1.5 AUC points seen on `ntire_val_hard` | medium | it does not — in which case round 1's entire ablation table was measured on an axis with almost no dynamic range, and D1's third outcome is the finding |
| **D1** | both arms collapse; preprocessing was not the confound on NTIRE | low | the crop arm collapses and the resize arm does not — the resize head took the semantic shortcut, and the whole PoC has to be refit on the crop detector |
| **E1** | exact, every delta 0.0000 | very high | anything else, and nothing downstream is a measurement |
| **E0** | `significant: true`, `parallel_fraction` ≈ 0.03 | high | a materially larger parallel fraction — the ceiling would rise and every gain below would be less impressive relative to it |
| **S0** | hard-split Δ AUC sd ≈ 0.0007 | high | a much larger spread would put E9's one live axis back inside noise |
| **E2** | arm A gains ≈ 0.0000; arm B gains ≈ +0.013 hard | high | arm A gaining anything at all. This is the load-bearing positive result |
| **E3** | weighted beats plain MSE by ≈ 3–4 seed-sd on the hard split; `cos_decision` is **not** a usable readout | medium | plain MSE matching it — the objective's one non-obvious idea would not be earning its place |
| **E4** | `\|auc_aux − 0.5\|` **rises** monotonically with stage-1 progress | medium | a fall, which is the hypothesis the section is written around and would be the most interesting negative in the project |
| **E5** | `auc_fused − auc_main` ≈ 0; the β sweep bounds retention below 1.0 at every β | high | any β clearing 1.0 — the headline claim, and the one result that would change the paper |
| **E7** | the 12-epoch ladder lands within ~2 seed-sd of the plain arm; `tap_gate` stays near init | medium | a real gap either way. Round 1 could not answer this at all — its ladder runs were a third of the training |
| **E8** | with `decay_gate: false` the gate **falls** | high | it rising, which would mean the objective does pull the correction magnitude up and round 1 mis-attributed it |
| **E6** | cached ≈ live | medium | a gap, which invalidates every cached number and forces a re-render at higher `n_epochs` |

**The one that matters most is P2.** Round 1's stage-1 validation gap was 1.5 AUC
points on `ntire_val_hard` — a scratch, not the collapse the method was designed
against. Every ablation landing inside noise of every other is the expected
consequence of measuring on an axis that narrow. If WildFake's 26-condition sweep
shows a real collapse, the ablations should separate. If it does not, §7 below is
the finding.

---

## 1. Round 1 — the one-line summary

The instrument is finished and the label-free objective works, but only inside
the third decimal place, and **no harness retention number exists yet** —
`eval_pipeline/results/` is empty because three of the four DINOv3 detector
configs point at checkpoints that do not exist or do not load (§5).

Every number below is stage-1 in-loop validation on `ntire_val` /
`ntire_val_hard`. None of it is the WildFake retention curve the project is
about.

---

## 2. Round 1 — status at a glance

| # | Experiment | Status | Verdict |
|---|---|---|---|
| **P0** manifests | done | 277,643 / 10,000 / 2,500 / 13,841 rows |
| **P1** stage 0 probe | done | both arms; resize head epoch 36, mean AUC 0.9032 |
| **P2** baseline eval | **not run** | blocks D1, E1, E5 headline |
| **P3** caches | done | 6 caches, 27 GB including the tap family |
| **D1** preprocessing confound | half | both probes trained; retention comparison blocked on P2 |
| **E0** drift asymmetry | done | significant but tiny; **97% of drift is invisible to the head** |
| **E1** identity adapter | not run | needs the harness |
| **E2** clean teacher | done, **stale objective** | **the teacher is the mechanism** — the control gains exactly 0.0000 |
| **E3** loss ablation | done, **stale objective** | **negative** — plain MSE is not worse, and `cos_decision` is a bad proxy |
| **E4** erasure trade-off | done, **stale objective** | **refuted** — drift gets *more* separable as stage 1 trains, not less |
| **E5** GRACE-D | half | stage 2 trained and beta swept: **fusion gain about 0**; harness eval blocked on P2 |
| **E6** cached vs live | not run | `dinov3_live` has no checkpoint |
| **E7** ladder / taps | half | 4-epoch runs only; `dinov3_ladder_final` (12 epochs) unrun |
| **E8** gate | done | **the gate's climb was weight decay**; the objective pulls it *down* |
| **E9** hyperparameter sweeps | done | nothing outside seed noise except the loss-weight ratio, marginally |

---

## 3. Round 1 — the headline numbers

Stage-1 validation, mean over held-out epochs. `auc_clean` and `auc_degraded`
are the frozen detector with no adapter; `retention` is the harness formula on
that pair.

| set | `auc_clean` | `auc_degraded` | baseline retention |
|---|---|---|---|
| `ntire_val` | 0.9594 | 0.9509 | 0.9814 |
| `ntire_val_hard` | 0.8534 | 0.8378 | 0.9559 |
| held-out degradations | 1.0000 | 0.9991 | 0.9983 |

**Read the gap first.** It is 0.85 AUC points on `ntire_val`, 1.56 on
`ntire_val_hard`, 0.09 on the cache's own held-out degradations. That is not the
collapse the method was designed against — it is a scratch. Everything GRACE
does here happens inside those 1.5 points, which is why every ablation below
lands within noise of every other.

Best arm (`dinov3_clean_final`, 5 seeds):

| set | retention | gap closed | delta AUC vs no adapter |
|---|---|---|---|
| `ntire_val` | 0.9876 +/- 0.0002 | 33% | +0.00287 +/- 0.00008 |
| `ntire_val_hard` | 0.9944 +/- 0.0020 | 87% | +0.01363 +/- 0.00071 |

**Seed noise is the unit of measurement.** Across 5 seeds the hard-split delta
AUC has sd 0.00071. Any two arms closer than about 0.0015 apart are the same arm.

---

## 4. Round 1 — findings, one per experiment

### E0 — drift asymmetry: significant, and it does not matter much

`results/dinov3_poc_drift.json`, all 14 epochs `significant: true`.

- asymmetry **0.0051** on drift magnitudes of about 0.12 — a 4% relative gap. The
  CI `[0.0041, 0.0062]` excludes zero because n = 277k, not because the effect is
  large.
- **`parallel_fraction` = 0.0298.** Only 3% of the drift lies along the direction
  the frozen head is sensitive to. **This is the ceiling on everything else in
  the project** — 97% of what the adapter could correct is invisible to the head
  by construction.
- Carried by `gaussian_noise` (0.030) and `resize` (0.021). `center_crop` is
  **negative** (-0.023): reals drift *further* than fakes there.

### E2 — clean teacher: confirmed, cleanly

`dinov3_clean` (arm B) vs `dinov3_degraded` (arm A).

Arm A gains **-0.0000** AUC on both val sets — exactly nothing, as predicted.
Arm B gains +0.0035 / +0.0126. The clean teacher is the mechanism; this is the
one unambiguous positive result in the project.

**Caveat: measured under the superseded objective.** Both arms predate the
`lam_sw` / `lam_id` removal and still log `sw` and `identity` in `history`. Those
terms were **1.65% of arm B's objective** (2.29% at the final step) and **0.000%
of arm A's** — under `target_view: degraded` the adapter reproduces its input, so
both terms are trivially satisfied (8e-11). The pair was therefore not a one-key
change in effect. The conclusion survives, because arm A gained *exactly zero*
and a 2% term cannot move a floor, but E2 has **not been re-run** on the current
objective. It costs minutes.

### E3 — loss ablation: negative, and it invalidates the documented readout

`dinov3_plain_mse` vs `dinov3_clean`.

| arm | final `cos_decision` | hard delta AUC | hard gap closed |
|---|---|---|---|
| `dinov3_clean` (weighted, `w_err` 2) | 0.0655 | +0.0126 | 81% |
| `dinov3_plain_mse` | **0.1072** | +0.0103 | 66% |
| `dinov3_clean_final` (weighted, `w_err` 4) | **0.0488** | +0.0148 | 95% |

Two things fall out, both contrary to `EXPERIMENTS.md` §8 as written:

1. Plain MSE does **not** sit near 0 on `cos_decision` — it is *twice* the
   weighted arm's. The predicted signature of the weighting term is absent.
2. `cos_decision` is **anti-correlated** with the outcome across these arms: the
   arm with the lowest alignment has the highest AUC gain. It is not a usable
   proxy for whether the weighting earns its place.

The weighting still wins on AUC (+0.0148 vs +0.0103, about 3.5 seed-sd apart), so
the term keeps its place — but not for the documented reason.

**Caveat: also pre-removal.** Both arms still log `sw` and `identity`. Here the
ablation stays clean — the removed terms were 1.65% of `dinov3_clean`'s objective
and 1.49% of `dinov3_plain_mse`'s, near-symmetric across the pair — but the
magnitudes are from a superseded objective and **E3 has not been re-run**. Note
`dinov3_clean_final` in the table is a *post*-removal run at different loss
weights; it is there for the `cos_decision` anti-correlation, not as an arm of
this ablation.

### E4 — erasure trade-off: hypothesis refuted

Six stage-2 runs against `dinov3_clean/step_*.pt`. The claim was that `auc_aux`
would **fall** as stage 1 improves.

| stage-1 step | `auc_aux` (val) | dev from chance | `auc_aux` (hard) | dev |
|---|---|---|---|---|
| 1,084 | 0.4713 | 0.029 | 0.4986 | 0.001 |
| 3,252 | 0.7171 | 0.217 | 0.5755 | 0.076 |
| 5,420 | 0.2293 | 0.271 | 0.3886 | 0.111 |
| 7,588 | 0.2738 | 0.226 | 0.4150 | 0.085 |
| 9,756 | 0.7919 | 0.292 | 0.6333 | 0.133 |
| 11,924 | 0.8097 | 0.310 | 0.6375 | 0.138 |

Separability **rises monotonically** on both sets — 0.03 to 0.31, and 0.001 to
0.138. Restoration is not erasing forensic evidence here; it is concentrating it.
The obvious critique of the whole approach does not land on this data.

> Caveat: the aux head's *polarity* flips (AUC below 0.5 at steps 5,420 and
> 7,588). Each stage-2 run is independent and only 4 epochs, so the sign is not
> stable. The magnitude is what the curve is about, and it is monotone.

**Also pre-removal.** All six x-axis points are `dinov3_clean/step_*.pt`, trained
under the objective that still carried `lam_sw` / `lam_id`. Stage 2 itself is
current. Re-running E4 means re-running stage 1 first, so it rides on the E2/E3
re-run — but note that the erasure hypothesis is about *stage 1's objective*, so
this is the arm where the stale objective matters most. `disc_gate_ctrl`,
`disc_ladder` and `disc_ladder_taps` all use post-removal adapters and are clean.

### E5 — GRACE-D: the branch works, the fusion does not

Stage 2 is trained and the beta sweep is done. `auc_aux` reaches 0.855 standalone
— Δ genuinely carries signal. But `auc_fused - auc_main` is between **-0.0008 and
+0.0006** everywhere, learned beta is +/-0.06 to +/-0.51, and `sweep_beta.py` on
`disc_gate_ctrl` put the ceiling at **0.9917 retention over all beta**, with the
two image sets peaking at **opposite signs**.

So the aux logit is *redundant with the main head*, not uninformative. No
weighting rescues it. `exceeds_clean_ceiling` cannot be claimed, and the harness
run would not change that conclusion — though it is still needed for the record.

### E7 — ladder / taps: inconclusive by construction

`dinov3_ladder_e4` and the `tap_dim` sweep ran **4 epochs (4,336 steps)** against
the plain arm's 12 (13,008). Their 66-71% hard gap-closed is a third of the
training, not a worse architecture — the comparison is not legitimate yet.
`dinov3_ladder_final.yaml` (12 epochs, matched loss weights) exists and is
**unrun**; that is the arm that settles it.

The figure the ladder exists to make is flat regardless: final `tap_gate` is
0.0188 / 0.0221 / 0.0213 / 0.0265 / 0.0199 across blocks 0/2/4/6/9 — all within a
hair of the 0.018 init, and per E8 that drift is weight decay anyway. No
localization signal.

`tap_dim` in {32, 64, 128, 256} spans hard delta AUC 0.0114-0.0118. Nothing.

### E8 — the gate: the climb was the optimizer

`dinov3_gate_nodecay` (`decay_gate: false`), controlled against
`dinov3_sweep_gate_-4`.

| run | gate first -> last |
|---|---|
| `dinov3_sweep_gate_-4` (decay on) | 0.01799 -> 0.02089 (**+0.0029**) |
| `dinov3_clean_final` (decay on) | 0.01799 -> 0.02095 (**+0.0030**) |
| `dinov3_gate_nodecay` (decay off) | 0.01799 -> **0.01645** (**-0.0015**) |

Exempt `gate_logit` from weight decay and the gate **falls**. The objective's net
pull on the correction magnitude is *downward*: the alignment term wants a
smaller correction than the init. Every "the gate is climbing, so the adapter is
learning to apply itself" reading in this project's history was AdamW.

Corroboration: `dinov3_sweep_gate_-3` starts 2.6x higher and moves *less*
(+0.0021), because a larger gate is decayed back harder.

**`gate` is not a health signal.** Do not report it as one.

### E9 — hyperparameter sweeps: everything is saturated

Against the 0.00071 seed sd on hard delta AUC:

| axis | arms | hard delta AUC range | verdict |
|---|---|---|---|
| loss ratio `w_err`/`w_cos` | 0.25, 1, 4, 16 | 0.0122-0.0149 | **the only live axis** — 16 beats 0.25 by about 4 sd |
| bottleneck | 128, 256 | 0.0125-0.0126 | saturated |
| `n_blocks` | 2, 3 | 0.0119-0.0126 | saturated |
| `tap_dim` | 32, 64, 128, 256 | 0.0114-0.0118 | saturated |
| gate init | -4, -3 | 0.0120-0.0132 | immaterial (see E8) |
| removed-path arms | 9 runs | 0.0116-0.0130 | all within noise; the feature was removed for this reason |

The ratio result is split-dependent and should be reported as such: on
`ntire_val` the preference **reverses** (`w_err` 2 / `w_cos` 0.5 gives +0.00324
vs +0.00287). Two five-seed families back this, so it is measured, not guessed.

---

## 5. Round 1's blockers — why it produced no harness numbers

`eval_pipeline/results/` was empty because three of the four DINOv3 detector
configs could not load. **All three are fixed in the round-2 tree** — the
right-hand column records what they point at now.

| config | round-1 problem | round-2 state |
|---|---|---|
| `dinov3+identity.yaml` | none | `checkpoint: null` — the arm, not a broken config |
| `dinov3+grace.yaml` | loaded, but stale `adapter_cfg` key (below) | `dinov3_clean/ema.pt`, written by step 8 |
| `dinov3+grace-d.yaml` | **`dinov3_disc/` did not exist** — stage-2 runs were named `e4_step_*`, `disc_ladder`, `disc_gate_ctrl` | `dinov3_disc/discrepancy.pt`, which is exactly what `dinov3_discrepancy.yaml`'s `run_id` writes (step 9) |
| `dinov3+grace-ladder.yaml` | **`dinov3_ladder/` did not exist** — the ladder run on disk was `dinov3_ladder_e4` | `dinov3_ladder_final/ema.pt`, written by step 19, and scored by the new `runs/dinov3_poc_ladder.yaml` |

The two ladder stage-2 configs had the same fault — both named
`dinov3_ladder_e4/ema.pt`, a run with no config in the tree — and both now name
`dinov3_ladder_final/ema.pt`. **Every checkpoint path in `configs/` now names a
run that some step of `run_all.sh` actually produces**, which is the property
that was missing.

### The stale `noise_dim` key — recoverable, and less bad than it looks

25 of 37 stage-1 checkpoints carry `noise_dim` in `adapter_cfg`, a field
`AdapterConfig` no longer accepts, so `load_adapter` raises. But:

- **24 of the 25 have `noise_dim: 0`** — the removed stochastic path was never
  active. Their `state_dict` is 21 tensors, structurally identical to a
  current-code run. Popping the key loads them with **zero missing or unexpected
  keys** (verified). Their numbers are comparable to current-code runs.
- **Only `dinov3_posterior` is genuinely dead** — `noise_dim: 16`, 24 tensors,
  three `noise.*` layers with no home in the current model. Its result (+0.0130
  hard) was inside noise of everything else, which is why the path went.

A one-line migration over `checkpoints/grace/*/*.pt` unblocks the other 24.

### Other loose ends

- `disc_ladder_taps/` has `discrepancy.pt` but **no `summary.json`** — the run
  (2026-08-29 15:23) did not finish writing. Rerun it.
- `sweep_beta.py`'s result exists **only as prose** in
  `dinov3_discrepancy_ladder_taps.yaml`'s header. It has no artifact on disk.
  Give it an `--out` JSON so the 0.9917 ceiling is a file, not a comment.
- 24 checkpoint directories have **no config in the tree** (`dinov3_final_r4*`,
  `dinov3_clean_final_s*`, the `*_e4` runs, `dinov3_sweep_gate_-4`, and the 9
  deleted removed-path arms). The seed replicates were CLI overrides and are
  fine; the deleted ones are reconstructible only from git history.

---

## 6. Round 1's "what to run next" — superseded

> **Superseded by the restart.** With no checkpoints on disk there is nothing to
> migrate and nothing to repoint; the order below is now steps 1-22 of
> `scripts/run_all.sh`, which additionally covers D1, E3, E4, E6 and S0. Kept for
> the reasoning, which is unchanged: the harness baseline gates everything, and
> the slow control is the cheapest thing to defer.

1. ~~**Migrate the 24 recoverable checkpoints** (strip `noise_dim`).~~ Moot.
2. **Repoint the three broken detector configs** at checkpoints that exist, and
   decide which stage-1 run is canonical — `dinov3_clean_final` is the current
   best and loads today.
3. **P2, the WildFake baseline.** Nothing downstream is a measurement until this
   exists, and it is about 8% of what the cache cost.
4. **E1 + E5 through the harness** — one `dinov3_poc_grace.yaml` run scores all
   three arms.
5. **D1** — the crop baseline, read against P2.
6. **Re-run E2 and E3 on the current objective** — `dinov3_degraded`,
   `dinov3_clean`, `dinov3_plain_mse`. All three predate the `lam_sw` / `lam_id`
   removal, and E2's pair was asymmetric in those terms (1.65% vs 0.000%). Both
   conclusions are expected to hold; this makes them reportable. Minutes each.
7. `dinov3_ladder_final` (12 epochs) to make E7 legitimate; rerun
   `disc_ladder_taps`.
8. **E6** (`dinov3_live`) last — it is the slow control and the cheapest to defer.

---

## 7. The framing risk

The stage-1 validation gap is 1.5 AUC points. Closing 87% of 1.5 points is a real
but small result, and the project's premise — that detectors *collapse* under
degradation — is not visible on `ntire_val_hard` at all. Either the WildFake
26-condition sweep shows the collapse and these numbers were measured on the
wrong axis, or it does not and D1's third outcome ("the dataset separates on
content") is the actual finding. **P2 decides which, and it has not been run.**
