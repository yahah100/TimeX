# Reproducing Table 2

SeqComb-MV and LowVar retain all six methods. The runner chooses the fixed recipe
corresponding to each selected five-fold seed-42 row in
[the report](../results_reproduction.md). There is no `--protocol` option.

```bash
uv sync
git submodule update --init --recursive
./run_table2.sh --help
./run_table2.sh --seed 42 --dry-run
./run_table2.sh --seed 42
# On the configured GPU cluster:
sbatch sj_timex_table2 --seed 42
```

Data defaults to `dataset/`; override with `TIMEX_DATA_ROOT` or `--data-root`.
It must contain `SeqCombMV/split=1.pt` through `split=5.pt` and
`LowVarDetect/split=1.pt` through `split=5.pt`. The cluster data root defaults to
`/beegfs/hahn/workspace/TimeX/dataset`; `TIMEX_CLUSTER_PROJECT_DIR` overrides the
persistent project root. The wrapper uses one GPU, 16 CPUs and a three-day limit.

## Retained recipes

| Method | SeqComb-MV | LowVar |
|---|---|---|
| TimeX | Original single-attempt predictor; original connectivity and reference dropout; effectively unclipped gradients | Qualified predictor; original connectivity; reference in eval mode; clip after backward |
| IG / Dynamask / WinIT | Qualified predictor with retries | Qualified predictor with retries |
| CoRTX | Original single-attempt predictor; reconstruction recipe | Qualified predictor; same reconstruction recipe |
| SGT + Grad | Independent classifier; Poly1 + attached KL; best validation macro-F1 | Same |

“Qualified” means validation macro-F1 >=0.95, stopping at the first successful
attempt among seeds `base_seed+fold-1+1000*a` for a=0,1,2. Original predictors
have one attempt with no quality rejection: retaining the weak SeqComb-MV folds
is necessary to reproduce the original TimeX/CoRTX rows. Each method resets its
RNG to `base_seed+fold-1`; original and qualified predictors have separate paths.

Predictors and SGT use 1000 epochs on SeqComb-MV and 120 on LowVar. TimeX uses
100 epochs, batch 64, GSAT r=0.5, weights GSAT=1, connectivity=2, explanation=2,
consistency=1 and label=1, temperature=1 and 50 prototypes. AdamW is lr=0.001,
weight decay=0.001 for SeqComb-MV TimeX, and lr=0.003, weight decay=0.0001 for
LowVar TimeX. The original validation consistency criterion selects TimeX.
CoRTX uses Adam lr=0.005, 100 encoder/50 decoder epochs and InfoNCE temperature
0.7. WinIT uses 1000 generator epochs. No test attribution metric selects a
checkpoint or predictor retry during training.

The released connectivity formula retains its known multivariate-axis defect.
CoRTX's multivariate reconstruction recipe remains unresolved. The best observed
scores are not uniformly within the paper's tolerance; see
[findings](../reproduction_findings.md) for the evidence and failed variants.

## Outputs and resume

Both tables use `experiments/synth_workflow.py`. Options include `--datasets`,
`--methods`, `--folds`, `--stage all|train|evaluate`, `--seed`, `--models-dir`,
`--results-dir`, `--dry-run` and `--max-samples`. Defaults write results beneath
`results/table2/selected-v1/seed_42/` and models beneath
`results/table2/models/selected-v1/seed_42/`.

Data/source/configuration/dependency hashes and artifact integrity govern reuse.
Stale outputs are preserved. Failed dependencies affect only dependent methods;
independent methods/folds continue and the job returns nonzero. Summaries report
all six methods, including missing or partial rows. A full result requires all
five folds and an unrestricted evaluation. Fold SE uses ddof=1; historical pooled
SE remains separately labelled. Numerical agreement requires all three metric
differences within ±0.05, and unresolved provenance is labelled separately.

The cluster wrapper preserves models, results and submitted source snapshots.
`--resume-from ARCHIVE` can resume artifacts produced by the cleaned workflow.
Pre-cleanup checkpoints are not migrated or silently accepted. Use fresh output
roots for numerical validation after the cleanup.

```bash
# Resume a cleaned-code run:
sbatch sj_timex_table2 --seed 42 --resume-from /path/to/cleaned_run_archive
# Repeat both additional LowVar seeds:
sbatch sj_timex_table2 --seeds "43 44" --datasets lowvar --methods ours
# Validate locally without full GPU training:
PYTHONPATH=. uv run python -m unittest discover -s tests -p 'test_*.py' -v
PYTHONPATH=. uv run python experiments/evaluation/report_synth.py --check
bash -n run_table1.sh run_table2.sh sj_timex sj_timex_table2
```

CPU tests validate retained numerical behavior and workflow integrity, not the
five-fold GPU scores. The displayed results remain archived measurements until
the cleaned implementation is numerically verified on the cluster.

## End-to-end diagnostic check

```bash
PYTHONPATH=. uv run python tests/smoke_synth_cli.py
```

This CPU check creates temporary data and runs the actual training and evaluation
entry points for all four datasets and all six methods. Training uses one epoch;
attribution uses two examples. It verifies fresh-checkpoint loading, finite
metrics, and diagnostic status in the generated summaries. These outputs never
replace the recorded five-fold results. Logs and checkpoints remain under the
printed `/tmp/timex-cli-smoke-*` directory. Use `--datasets lowvar` or
`--methods ours ig` for a smaller check.

Both Slurm wrappers now archive the submitted source once and run every seed
from that saved snapshot. The cluster-wrapper tests simulate edits to the live
checkout between seeds and verify that archived results still use the original
snapshot. These tests run locally without submitting cluster jobs.
