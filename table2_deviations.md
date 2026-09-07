# Table 2 deviations and repair evidence

This audit supersedes the earlier attribution of every deviation to predictor
convergence or baseline substitutions. Historical results are from
`timex_table2_21846132/seed_42/`; repaired GPU results are pending. See the
[workflow and commands](experiments/TABLE2.md) and
[hashed CPU evidence](experiments/table2_cpu_evidence.json).

## Confirmed defects

| Finding | Evidence | Repair |
|---|---|---|
| Multivariate TimeX connectivity compares samples | `MaskGenerator` emits `(T,B,d)` for d>1; `ConnectLoss` differences axis 1 | Explicit `(B,T,d)` conversion and `temporal-l1-v1` |
| Connectivity reduction differs from the paper | Global L2 norm divided by the number of differences | Sum absolute temporal differences divided by B×T×d; zero for T=1 |
| Frozen reference dropout remains active | `model.train()` recursively enables the frozen `encoder_main` | Keep the frozen reference in evaluation mode in `repaired-v1` |
| Clipping runs before backward | Released trainer clips gradients immediately after zeroing them | Backward, finite-gradient check/clip, optimizer step |
| Predictor folds 3/4 fail validation | Reloaded validation macro-F1 **0.78716346 / 0.86871410** | Gate at ≥0.95; at most three predefined initializations |
| Logs call macro-F1 “AUC” | Trainer uses `sklearn.metrics.f1_score(average='macro')` | Correct metric labels |
| Loss weights omitted from TimeX config | `set_config()` lacked `loss_weight_dict` | Serialize loss weights, connectivity and training versions, and outer training weights; keep legacy defaults |
| Unsafe checkpoint reuse | Runner skipped training on file existence | Data/config/source/dependency hashes plus output integrity checks |
| SGT objective differs from released code | Current baseline used CE and detached original logits in KL | Restore Poly1 and attached KL target |

The [paper's explanation loss](https://arxiv.org/html/2306.02109v2#S4.S1)
regularizes temporal changes. The new connectivity test deliberately uses
samples with different but temporally constant masks: the repaired loss is zero
and invariant to sample order or batch duplication. The old univariate formula
remains available as `legacy` for old checkpoints and diagnostic controls.

The predictor failure is established; how much of the attribution deficit it
caused is not established. Healthy SeqComb-MV folds also miss the published
recall. Sparsity, checkpoint selection and random-seed variation remain
hypotheses for residual gaps. Table 6 weights and the validation consistency
selection criterion have therefore been retained. No test explanation metric
selects a checkpoint, retry, hyperparameter or preferred seed.

## Protocol changes and baseline provenance

The author snapshot already contains ten-epoch SGT and reconstruction-based
CoRTX scripts. Restoring missing paper settings is not an accurate description
of replacing either of those choices.

### SGT

`experiments/other_baselines/train_SGT.py` specifies Poly1 classification,
`KLDivLoss(log_target=False, reduction='batchmean')`, and gradients through both
original and masked logits. It masks the bottom 90% of target-gradient features
with per-sample uniform random replacements. The generalized implementation
keeps that objective and correct batch/time layout. PyTorch generates replacement
noise in the generalized helper; the released helper uses NumPy, so identical
noise realizations across implementations are not claimed.

The repaired classifier uses 1000/120 epochs and selects by validation macro-F1.
That is an explicit convergence repair, as is reporting classifier performance
beside attribution scores. `sgt-control` retains ten epochs, final checkpoint
selection, and the released masked-input evaluation behavior separately. The
main SGT + Grad row uses absolute target-logit gradients. A fresh random
classifier is part of the SGT method; it is not itself a defect.

### CoRTX

Local history (`df130a3`, then the rename `f5bd9ae`) and these released files
provide the available recipe:

- `experiments/other_baselines/cortx_exp.py`: one FreqShape fold, predictor-copy
  initialization, Adam 0.005, 100 encoder/50 decoder epochs, reconstruction MSE.
- `contrast_generator.py`: independent Bernoulli retention of 20% of input values.
  `pos_num`, `neg_num` and the reference predictor are unused in the executed path.
- `infonce.py`: temperature 0.7; cross-view paired positives. Forward losses **and
  gradients of both views** agree with the local symmetric implementation in
  float64 to 1e-12 on tested batch sizes 1, 4 and 16. No InfoNCE replacement is
  justified by these checks.
- The released decoder is univariate (`d_inp=1`). DatasetwInds supplies batches
  `(B,T,d)`, while some released calls leave `captum_input=False`, which expects
  `(T,B,d)`. The generalized runner explicitly converts layouts at boundaries.
- Encoder dropout remains active during decoder training under `no_grad` in the
  snapshot. The reproduction adaptation preserves this; it does not silently
  change encoder mode. The released `model.mlp.requires_grad_ = False` assignment
  is ineffective; the generalized helper freezes the classifier head correctly,
  which receives no gradient from the embedding-only contrastive objective anyway.

The public [TimeX baseline directory](https://github.com/mims-harvard/TimeX/tree/main/experiments/other_baselines)
was checked as an additional author artifact; it did not provide an accessible,
verified multivariate recipe in this audit. Local history is the concrete source
for the settings above. Reconstruction already exists in that source, so the
old report's assertion that the new runner invented the objective was incorrect.
The [CoRTX method paper](https://arxiv.org/abs/2303.02794) describes contrastive
representations followed by explanation-head training; no supported multivariate
TimeX adaptation resolving the reconstruction issue was recovered here.

## LowVar saturation diagnosis

On **all 5,000 training examples of LowVar fold 1**, **64.0584%** of values lie
outside `[0,1]`. Any sigmoid-bounded reconstruction has MSE at least
**0.48343837**, computed by projecting inputs into that interval. This closely
matches the archived decoder plateau near 0.4847 and establishes an objective
range incompatibility without needing a new GPU run.

On the first 128 training examples, the archived decoder (evaluation mode)
has **63.8105%** of outputs below 0.01 or above 0.99, mean **0.399388**, standard
deviation **0.424000**, and reconstruction MSE **0.517303**. The earlier claim of
a near-constant mask is **not supported** by this audit. Similar AUR values across
folds alone do not establish a constant mask. Saturation is directly observed.

The repair logs these quantities during decoder training. No normalization,
unbounded decoder or alternative SHAP target has been substituted without an
author-supported recipe. Both CoRTX rows remain **unresolved provenance**, even
if all three SeqComb-MV metric differences happen to be within tolerance.

## What remains to be measured

The focused CPU checks pass; the workflow documents Slurm pilots for legacy,
connectivity-only and combined TimeX corrections, predictor recovery, and LowVar
regression/baseline diagnosis. GPU execution is delegated to the user by request.
Then run every seed-42 fold/method, reusing only matching repaired pilot artifacts.
If supported rows still miss ±0.05 in any metric, run the affected chains at both
seeds 43 and 44 and report every seed. No repaired row is yet numerically verified.
