# Reproducing Table 2

The repaired workflow evaluates SeqComb-MV and LowVar, six methods, and the five
published folds. The reference is the [TimeX paper, Tables 2, 6 and 8](https://arxiv.org/html/2306.02109v2).
Numerical reproduction requires five traceable folds and **each** AUPRC, AUP and
AUR difference within ±0.05. GPU runs remain to be performed by the user; CPU
checks below establish correctness only.

## Setup and preview

From the repository root, run `uv sync`. Initialize the WinIT submodule if it is
absent. Set `TIMEX_DATA_ROOT` to a directory containing `SeqCombMV/split=1.pt`
through `split=5.pt` and `LowVarDetect/split=1.pt` through `split=5.pt`.

```bash
./run_table2.sh --help
./run_table2.sh --datasets seqcomb_mv --methods ours --folds 1 --dry-run
```

The preview prints dependency commands and whether current artifacts match;
it does not create directories, train, or evaluate. A dependency marked pending
will receive its actual checkpoint hash after training.

`--models-dir PATH` and `--results-dir PATH` select a run's storage roots. Both
contain `<protocol>/seed_<seed>/<dataset>/fold_<fold>/`. Choose new roots for an
independent run. The runner checks all existing folds when aggregating results,
but launches only the folds and methods requested. It supports `--stage train`
and `--stage evaluate`; evaluation refuses stale training artifacts.

## Protocols

| Protocol | TimeX connectivity | Frozen reference / clipping | SGT |
|---|---|---|---|
| `repaired-v1` (default) | Temporal L1, normalized by B×T×d | Evaluation mode; clip after backward | Poly1 + attached KL; 1000/120 epochs; best validation macro-F1; target-logit gradients |
| `legacy-control` | Released layout and global L2 | Released training behavior | Same convergence repair |
| `connectivity-control` | Temporal L1 | Released training behavior | Same convergence repair |
| `sgt-control` | Temporal L1 | Repaired training | Released 10 epochs, final checkpoint, masked-input attribution control |

Controls are diagnostic and cannot receive a `matched` status. The legacy
control isolates the TimeX implementation; it still uses the predictor quality
gate and is not an exact replay of the old failed run.

TimeX keeps Table 6: 100 epochs, batch 64, AdamW lr 0.001 and weight decay
0.001, r=0.5, GSAT weight 1, connectivity weight 2, explanation weight 2,
consistency/label weights 1, temperature 1, and 50 landmarks. Selection remains
the existing validation consistency criterion. Each checkpoint stores the loss
weights, connectivity and training versions. Epoch JSON includes separate loss
terms, mask moments/tails, temporal variation, validation score and selected epoch.
Old checkpoints remain loadable with legacy defaults.

Predictors retain 1000/120 epochs and existing optimizers. The gate is validation
macro-F1 ≥0.95. Attempt `a` uses `base_seed + fold - 1 + 1000*a`, for `a=0,1,2`.
The best validation checkpoint within each attempt is selected; retrying stops
at the first qualifying attempt. `--predictor-attempts` permits 1–3 attempts;
`--predictor-min-f1` may tighten the gate. Failed attempts and quality records
are retained. Exhausted attempts stop that fold's dependent stages; independent
folds continue, and the job exits nonzero after writing its status and summaries.
Every downstream stage resets its RNG to `base_seed + fold - 1`.

SGT trains its own classifier with 90% least-salient features replaced by uniform
noise within each sample's range. Poly1 classification and KL both propagate
gradients through the original logits; the masked-input tensor is detached.
The longer budget and validation checkpoint selection are explicit convergence
repairs, not settings recovered from the released ten-epoch script. The existing
SGT optimizer settings remain 5e-4/0.01 for SeqComb-MV and 1e-3/0.01 for LowVar
(lr/weight decay). Evaluation records SGT's test macro-F1 on the full test set,
while attribution metrics retain the published nonzero-class population.

CoRTX retains the released reconstruction adaptation: predictor initialization,
80% random dropout augmentation, symmetric cross-view InfoNCE at temperature
0.7, Adam lr 0.005, 100 encoder and 50 decoder epochs. Decoder inputs use
(time, batch, features); the encoder remains in training mode under `no_grad`
as in the author snapshot. The decoder logs saturation, output moments, input
range incompatibility and the bounded-reconstruction MSE lower bound. No new
SHAP or reconstruction objective is introduced. Its multivariate provenance
remains unresolved; see [deviation notes](../table2_deviations.md).

## CPU validation

```bash
PYTHONPATH=. uv run python -m unittest discover -s tests -p 'test_table2_repairs.py' -v
PYTHONPATH=. uv run python experiments/evaluation/diagnose_table2.py \
  --archive timex_table2_21846132/seed_42 \
  --output results/table2_cpu_diagnosis.json
bash -n run_table2.sh sj_timex_table2
```

The diagnostic command requires the archived checkpoints and both local datasets.
It performs inference on CPU without training. The checked-in
[CPU evidence](table2_cpu_evidence.json) records data and checkpoint hashes.
The tests use small tensors and temporary directories; they cover connectivity
invariances, legacy compatibility, clipping order, reference determinism,
checkpoint round trips, InfoNCE forward/backward parity against the released
formula, SGT masking/gradients, retries, reduced baseline training, artifact
integrity, dependency invalidation, seed isolation, partial results and relocated
archive reuse. Slurm itself is verified syntactically, not executed locally.

For attribution plumbing on previously trained artifacts:

```bash
./run_table2.sh --datasets seqcomb_mv --methods ours --folds 1 \
  --stage evaluate --max-samples 2
```

Reduced evaluations are labelled diagnostic. They never count as complete folds.
Changing the sample limit causes evaluation to run again with preserved stale
outputs. Their metrics must not be used as Table 2 estimates.

## GPU pilots and full run

Run these commands on the existing Slurm cluster. The wrapper uses the `gpu`
partition, `etechnik_gpu` account, one GPU, 16 CPUs and a three-day limit. The
submission checkout needs its `.venv` and WinIT source; data defaults to
`/beegfs/hahn/workspace/TimeX/dataset`. Override the persistent project root with
`TIMEX_CLUSTER_PROJECT_DIR` when needed.

```bash
# TimeX controls: identical fold, seed, budget and validation selection.
sbatch sj_timex_table2 --datasets seqcomb_mv --folds 1 --methods ours --protocol legacy-control
sbatch sj_timex_table2 --datasets seqcomb_mv --folds 1 --methods ours --protocol connectivity-control

# Combined repair, including fold 3 predictor recovery.
sbatch sj_timex_table2 --datasets seqcomb_mv --folds 1,3 --methods ours

# LowVar regression and baseline diagnosis.
sbatch sj_timex_table2 --datasets lowvar --folds 1 --methods ours,cortx,sgt+grad

# Small released SGT control, kept separate from repaired estimates.
sbatch sj_timex_table2 --datasets all --folds 1 --methods sgt+grad --protocol sgt-control
```

After checking pilot logs for finite training and the validation gates, launch
seed 42 with all five folds and all six methods. Replace the two archive paths
with those of the **combined repair** and **LowVar** pilot jobs:

```bash
sbatch sj_timex_table2 --seed 42 \
  --resume-from /beegfs/hahn/workspace/TimeX/cluster_runs/timex_table2_SEQ_PILOT_JOB \
  --resume-from /beegfs/hahn/workspace/TimeX/cluster_runs/timex_table2_LOWVAR_PILOT_JOB
```

Without pilot reuse, the complete command is `sbatch sj_timex_table2 --seed 42`.
Do not alter hyperparameters or choose seeds based on test explanation metrics.
If supported rows remain outside tolerance, run the affected dependency chains
with **both** seeds 43 and 44 and report each seed separately, for example:

```bash
sbatch sj_timex_table2 --seeds "43 44" --datasets seqcomb_mv --methods ours
```

## Artifacts, resume and acceptance

Each job archives `source/`, the Git revision/diff, and
`seed_<seed>/{models,results}/<protocol>/seed_<seed>/...` beneath
`<project>/cluster_runs/timex_table2_<job-id>/`. Historical `timex_*` directories,
checkpoints, datasets, result trees and environments are excluded from source
staging. The wrapper synchronizes models, metadata, logs and summaries on normal
exit and handled signals. A hard SIGKILL/node loss cannot invoke the exit trap.

Use repeatable `--resume-from` options to stage prior archive roots. Paths are
excluded from cache identity, so relocated artifacts can be reused. Data hashes,
source-file hashes, stage configuration, seeds and upstream checkpoint hashes
must match, and artifact contents are rehashed. Missing metadata rejects reuse;
the historical archive cannot silently qualify as repaired output. Replaced
predictors invalidate TimeX/CoRTX and every evaluation using that predictor.
WinIT generator identity is independent of the predictor. Stale checkpoints,
failed attempts, diagnostic outputs and old logs are preserved before replacement.
Changing implementations conservatively invalidates affected source identities.

Dataset summary JSON preserves `folds`, `pooled`, and `cross_validation` metric
fields and adds protocol, provenance, completion and predictor-quality fields.
The summary CSV/Markdown includes all twelve published rows and each of the
three metric differences, even when results are missing. Primary uncertainty is
fold SE (`ddof=1`); `historical_pooled_standard_error` is separately labelled and
uses the historical sample-pooled convention (`ddof=0`). A row is `matched` only
with five complete, verified repaired folds, qualifying predictors, supported
provenance and all three differences ≤0.05. CoRTX remains `unresolved provenance`.
