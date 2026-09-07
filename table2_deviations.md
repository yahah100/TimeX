# Table 2: What Went Wrong

Root-cause analysis of the deviations between the reproduced Table 2 numbers
(`timex_table2_21846132/seed_42/results/`, base seed 42) and the published values.
Companion to [`results_reproduction.md`](results_reproduction.md), which reports the
numbers; this document explains the four rows that do not match.

## Summary

| Deviation | Cause | Status |
| :--- | :--- | :--- |
| TimeX, SeqComb-MV, −0.3143 AUPRC | Predictor folds 3 and 4 failed to converge, **plus** a residual coverage gap on the healthy folds | Partly explained |
| CoRTX, LowVar, −0.3859 AUPRC | The CoRTX stand-in does not implement CoRTX; on LowVar it degenerated to a near-constant mask | Not a CoRTX measurement |
| CoRTX, SeqComb-MV, −0.0006 AUPRC | Same stand-in. The apparent exact match is a coincidence, not a validation | Not a CoRTX measurement |
| SGT + Grad, both datasets | The SGT model is trained from scratch for 10 epochs and never converges | Not an SGT measurement |
| IG, Dynamask, WinIT, TimeX on LowVar | — | Replicate within ±0.03 |

Hyperparameters are not the cause of any of these. `experiments/seqcomb_mv/bc_model_ptype.py`
(r = 0.5, λ_LC = 1.0, λ_E = 2.0, λ_con = 2.0, τ = 1.0, n_L = 50, 100 epochs) matches paper
Table 6, and `train_transformer.py` (1000 predictor epochs for SeqComb-MV, 120 for LowVar)
matches paper Table 8. Every large deviation comes from either a reimplemented method or a
predictor fold that did not train.

## 1. TimeX on SeqComb-MV

### 1.1 Predictor folds 3 and 4 did not converge

From `seqcomb_mv_predictor_train.log` and `seqcomb_mv_ours_results.json`:

| Split | Best val AUC | Test F1 | TimeX AUPRC | TimeX AUP | TimeX AUR |
| :---: | :---: | :---: | :---: | :---: | :---: |
| 1 | 1.0000 | 0.9539 | 0.4604 | 0.8370 | 0.2533 |
| 2 | 1.0000 | 0.9841 | 0.3035 | 0.6840 | 0.2847 |
| 3 | 0.7872 | 0.7787 | 0.2812 | 0.6196 | 0.3036 |
| 4 | 0.8687 | 0.7524 | 0.3758 | 0.7027 | 0.3468 |
| 5 | 1.0000 | 0.9639 | 0.4465 | 0.8110 | 0.2807 |

Folds 3 and 4 ran with identical hyperparameters to the folds that worked; they are
seed-dependent optimization failures. Training selects the best-validation-AUC checkpoint,
so a fold that never reaches a good epoch saves a bad classifier. An explainer cannot
recover signal the classifier never learned, so these two folds drag the mean down.

### 1.2 The healthy folds still miss by ~0.23 AUPRC

Splits 1 and 5 average 0.4535 AUPRC against the published 0.6878. On those same folds:

- **AUP matches**: 0.8370 and 0.8110 against a published 0.8326.
- **AUR does not**: 0.2533 and 0.2807 against a published 0.3872.

Precision is right and recall is not, which means the mask is correct but too narrow — it
locks onto a clean core of the salient subsequence without covering the rest of it. This is
not a tie artifact of the straight-through binary mask: evaluation reads `out['mask_in']`
(`txai/models/bc_model.py:243-249`), the continuous pre-STE mask.

Since r, the loss weights, and the epoch budget all match the paper, the remaining
candidates are the model-selection criterion stopping on an over-sparse mask (selected
epochs were 83–100 out of 100, per `seqcomb_mv_timex_train.log`) or seed variance in the
GSAT/connectivity balance. **This part is a hypothesis and has not been tested.** The
fold 3/4 predictor failure above is established from the logs.

### 1.3 The gap is not fold noise

Reported uncertainty is the standard error across the five fold means:

- SeqComb-MV: 0.3735 ± 0.0363 against 0.6878 — roughly 8.7 SE.
- LowVar: 0.8371 ± 0.0231 against 0.8673 — roughly 1.3 SE, a clean replication.

## 2. CoRTX

### 2.1 The implementation is not CoRTX

`experiments/other_baselines/train_synth_baselines.py:39-88` trains an encoder with symmetric
in-batch InfoNCE, then trains the mask decoder with `F.mse_loss(mask, x_tb)`. The resulting
"explanation" is a reconstruction of the input. It never observes the predictor's outputs
and never approximates a Shapley value, which is the whole mechanism of CoRTX (contrastive
pretraining plus an explanation head distilled against approximate Shapley targets).

`experiments/TABLE2.md` records the InfoNCE substitution, made because the legacy snapshot
depends on the absent `PyGCL`. That is the smaller of the two substitutions. The decoder
objective is the one that makes the row measure something other than CoRTX.

### 2.2 LowVar collapsed to a constant mask

The decoder loss flatlines at 0.4847 by epoch ~30 (`lowvar_cortx_train.log`), against
0.0376 on SeqComb-MV (`seqcomb_mv_cortx_train.log`) — it never learned to reconstruct.
The metrics show the resulting degeneracy directly (`lowvar_cortx_results.json`):

- AUR is `0.6667` on all five folds, standard error 4.9e-7.
- AUP is 0.0590, approximately the ground-truth density.

That is chance-level behaviour from a near-constant saliency map, not a CoRTX score.

### 2.3 The SeqComb-MV match is coincidental

0.3623 against a published 0.3629 reads as the strongest replication in the table. It is
not: on SeqComb-MV an input reconstruction happens to correlate with the ground-truth
subsequences. The fold-level standard error of 0.0007 is the tell — a genuinely
model-dependent explainer would vary across folds whose predictors range from F1 0.75 to
0.98 (Section 1.1). This one does not, because it never looks at the model.

## 3. SGT + Grad

`train_sgt` (`experiments/other_baselines/train_synth_baselines.py:88-113`) builds a freshly
initialized `TransformerMVTS` and trains it for **10 epochs** with the saliency-guided
objective; `saliency_exp_synth.py:487-497` then takes `|∂logit/∂x|` of that model. The
paper trains these predictors for 1000 epochs (SeqComb-MV) and 120 epochs (LowVar).

`lowvar_sgt_train.log` shows the loss moving 1.3118 → 1.0911 → 1.2772 across the ten
epochs. With four classes, ln 4 ≈ 1.386, so the model sits near chance. Gradients of a
near-chance model are noise, and the attribution scores land at base rate: 0.1436 AUPRC on
LowVar and 0.1012 on SeqComb-MV.

A second issue compounds it: SGT is the only row in the table that explains a *different*
classifier from every other method, so even a fully trained SGT number would not be
directly comparable to the others.

## 4. Corrections to `results_reproduction.md`

- **§5.2** attributes the SeqComb-MV gap to predictor convergence. That is correct but
  incomplete: the two healthy folds still miss by ~0.23 AUPRC with a low-recall signature
  (Section 1.2).
- **§5.3.2** attributes the CoRTX LowVar deviation to the InfoNCE reimplementation. The
  decoder objective is the larger substitution, and it invalidates the SeqComb-MV row too,
  which the document currently presents as an "almost exact reproduction" (Section 2).
- **§5.3.3** attributes the SGT shortfall to using absolute gradients rather than masked
  training inputs. That is an evaluation-time detail; the dominant effect is that the model
  is trained for 10 epochs and never converges (Section 3).

## 5. Evidence

All paths relative to `timex_table2_21846132/seed_42/results/`:

- `seqcomb_mv_predictor_train.log` — per-fold best val AUC and test F1
- `seqcomb_mv_timex_train.log` — per-fold selected epoch
- `seqcomb_mv_ours_results.json`, `lowvar_ours_results.json` — per-fold TimeX metrics
- `lowvar_cortx_train.log`, `seqcomb_mv_cortx_train.log` — encoder and decoder loss curves
- `lowvar_cortx_results.json` — the constant-AUR degeneracy
- `lowvar_sgt_train.log`, `seqcomb_mv_sgt_train.log` — SGT loss curves

Source: `txai/baselines/synth_baselines.py`, `experiments/other_baselines/train_synth_baselines.py`,
`experiments/evaluation/saliency_exp_synth.py:470-497`.
