# Reproduction findings and retired variants

The maintained tables in [results_reproduction.md](results_reproduction.md) keep
all six methods, selecting the highest AUPRC among complete five-fold seed-42
configurations. AUP and AUR always come from that same configuration. This is
retrospective selection using observed test attribution scores. A weak method is
still included when it is the best completed version available; inclusion does
not imply agreement with the paper or establish an author-supported recipe.

## What the new cluster runs established

| Dataset | TimeX configuration | Folds | AUPRC | AUP | AUR |
|---|---|---:|---:|---:|---:|
| LowVar | Original, 21846132 | 5 | 0.8371 | 0.5070 | 0.9031 |
| LowVar | Combined repairs, 21899369 | 5 | 0.3134 | 0.2492 | 0.6473 |
| LowVar | Connectivity rollback, 21926721 | 5 | 0.8547 | 0.6076 | 0.9062 |
| SeqComb-MV | Original, 21846132 | 5 | 0.3735 | 0.7308 | 0.2938 |
| SeqComb-MV | Combined repairs, 21899369 | 5 | 0.0859 | 0.0595 | 0.4578 |

The LowVar rollback is the retained TimeX recipe: released connectivity,
frozen reference in evaluation mode, and gradient clipping after backward.
The seed-43/44 follow-ups in job 21927128 also recovered AUPRC to 0.8711/0.8704.
They are reported separately, without replacing the main seed-42 result.

SeqComb-MV controls are **single-fold diagnostics**, not five-fold estimates:

| Fold-1 configuration | Job | AUPRC | AUP | AUR |
|---|---|---:|---:|---:|
| Original | 21846132 | 0.4604 | 0.8370 | 0.2533 |
| Combined repairs | 21899369 | 0.0795 | 0.0583 | 0.5049 |
| Connectivity rollback | 21921843 / 21926653 | 0.0746 | 0.0575 | 0.5221 |
| Full legacy control | 21926723 | 0.4604 | 0.8370 | 0.2533 |
| Temporal L1 with legacy training | 21927129 | 0.0858 | 0.0526 | 0.5072 |

The four new fold-1 configurations share identical dataset and predictor hashes.
Restoring connectivity alone did not recover SeqComb-MV. Restoring all original
training behavior recovered the original fold almost exactly. Temporal L1 also
failed with the original training behavior. These observations implicate both
changes under the tested settings; they do not isolate dropout handling from
clipping, nor establish the behavior of unmeasured folds.

## Retained numerical behavior

- **Connectivity:** the released global-L2 reduction on the generator's native
  mask layout. Multivariate masks use `(T,B,d)`, so differencing axis 1 compares
  examples. This known defect is retained explicitly because changing it damaged
  the measured scores. No claim is made that it implements the paper's temporal
  objective. The unused temporal-L1 implementation and protocol controls were removed.
- **TimeX training:** FreqShapes, SeqComb-UV and SeqComb-MV keep reference dropout
  active and effectively unclipped gradients. The old clip-before-backward call
  was a no-op; it is replaced with explicit absence of clipping. LowVar keeps
  reference evaluation mode and effective clipping. Initialization order,
  optimizer settings, seeds and validation checkpoint criteria remain part of
  each recipe. Finite-value checks and loss-weight recording remain useful.
- **Predictors:** SeqComb-MV TimeX and CoRTX use the original one-attempt predictor
  initialization, including weak folds 3/4. Their reloaded validation macro-F1
  was 0.78716346/0.86871410. IG, Dynamask and WinIT use the qualified predictors:
  at most three predefined seeds, stopping at validation macro-F1 >=0.95. The
  improved run reached 1.0 on these two folds. The two predictor families are
  stored separately. Quality is recorded without rejecting original recipes.
- **SGT:** the completed Table 1 runs use cross-entropy, a detached original-logit
  KL target, 100/200 epochs and the final checkpoint. Table 2 retains Poly1 with
  gradients through both KL views, 1000/120 epochs and best validation macro-F1.
  Both evaluate absolute target-logit input gradients. The ten-epoch Table 2 and
  masked-input attribution controls were removed. The Table 1 recipe is retained
  because no complete measurement establishes the repaired objective there.
- **CoRTX:** retain predictor initialization, 80% dropout augmentation, temperature
  0.7 symmetric InfoNCE, 100 encoder epochs and 50 reconstruction-decoder epochs.
  SeqComb-MV selects the original predictor version by the agreed AUPRC rule:
  0.3623238919 versus 0.3621972008 with the retried predictors. This tiny difference
  is not evidence of a meaningful performance improvement. LowVar ties exactly.

SGT LowVar AUPRC improved from 0.1436 to 0.4149; SeqComb-MV improved from 0.1012
to 0.1265. Both still miss the paper substantially on at least one metric.
TimeX's original SeqComb-MV five-fold result remains below the paper too.

## CoRTX limitations and prior corrections

The multivariate CoRTX recipe remains unresolved. The local/released cross-view
InfoNCE formulas agree in forward values and both input gradients to 1e-12 in
float64 CPU checks. Replacing InfoNCE was not supported by that evidence.

On all 5,000 LowVar fold-1 training examples, 64.0584% of input values lie outside
`[0,1]`. A sigmoid-bounded reconstruction therefore has MSE at least 0.48343837,
close to the archived training plateau near 0.4847. The first 128 training
examples produced 63.8105% saturated decoder outputs, mean 0.399388 and standard
deviation 0.424000. The output was saturated, not nearly constant. The
[recorded CPU evidence](experiments/table2_cpu_evidence.json) remains available.

Reconstruction already appeared in the author snapshot, as did ten-epoch SGT.
The convergence repairs were explicit protocol changes, not recovered missing
paper settings. No replacement normalization, decoder or attribution target is
introduced in this cleanup.

## Cleanup and verification boundary

Unused one-fold CoRTX/SGT research launchers and the archived-checkpoint diagnosis
script were removed. Their evidence is recorded here; the released InfoNCE formula
is retained as a small test fixture. The old files remain available in Git history.

The five experimental protocol switches were replaced with fixed dataset/method
recipes. Both tables now share one runner, with independent seeds, source/data/
checkpoint hashes, partial-result handling and preserved stale outputs. Old
checkpoint-version arguments and compatibility tests were removed. Archived
models/results are left in place and are not silently reused by the new runner.

The retained recipes are checked with reduced CPU training and runner tests.
These checks cannot establish GPU five-fold numerical equivalence. Reproduction
commands must be rerun on the cluster before calling the cleaned implementation
numerically verified; the displayed measurements remain their original archives.

## Integration checks after cleanup

The real command-line training and evaluation paths passed on all four synthetic
datasets and all six methods using generated CPU data, one training epoch per
stage and two evaluated examples. Fresh model files loaded successfully, all
attribution metrics were finite, and every populated summary row remained
labelled diagnostic. Run `PYTHONPATH=. uv run python tests/smoke_synth_cli.py` to
repeat this check; it does not use published datasets or report benchmark scores.

This follow-up also found and fixed cluster staging issues: Table 1 now preserves
its submitted source, both wrappers stage every seed from that saved snapshot,
and Table 2 rejects retired protocol/predictor-override options before starting
work. Local wrapper tests cover a checkout edit between two sequential seeds.
Full five-fold GPU numerical verification is still pending.
