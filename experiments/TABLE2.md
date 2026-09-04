# Reproducing Table 2

Table 2 evaluates SeqComb-MV and LowVar over five folds with TimeX, Integrated
Gradients, Dynamask, WinIT, CoRTX, and SGT + Grad. Run commands from the
repository root. Data defaults to `dataset/`; `TIMEX_DATA_ROOT` can point to a
different root containing `SeqCombMV/` and `LowVarDetect/`.

## Local workflow

```bash
./run_table2.sh
./run_table2.sh --datasets seqcomb_mv --methods ours,ig --stage evaluate
```

Use `./run_table2.sh --help` for selection and output options. Every command has
its own log. Evaluations write JSON records plus `table2_summary.csv` and
`table2_summary.md`. Existing complete checkpoint sets are reused; a failed run
can therefore be restarted with the same command.

For a fast plumbing check, limit evaluation samples without changing the public
runner interface:

```bash
TIMEX_MAX_SAMPLES=2 ./run_table2.sh --datasets seqcomb_mv --methods ours --stage evaluate
```

Each `*_results.json` document has `schema_version`, dataset/method/base-seed
metadata, a `folds` list containing each fold's sample count and metric means,
sample-pooled metric means under `pooled`, and fold mean/standard error pairs
under `cross_validation`. The fold-level standard error is the primary reported
uncertainty; the pooled block is retained for comparison with historical runs.

The paper settings are 1,000/120 predictor epochs for SeqComb-MV/LowVar, 100
TimeX epochs, 100 CoRTX encoder epochs at temperature 0.7, 50 CoRTX decoder
epochs, and 10 SGT epochs. WinIT defaults to 1,000 generator epochs. Fold `i`
uses base seed plus `i - 1`; the default base seed is 42. CoRTX uses a local
symmetric in-batch InfoNCE implementation because the legacy snapshot's PyGCL
dependency is absent. SGT is evaluated with absolute gradients from the trained
model, rather than with the masked training input.

These are expensive research runs. Dynamask optimizes a mask per test sample,
and WinIT repeatedly samples counterfactuals, so a full CPU run is impractical.
Use a GPU and allow up to several days.

## Slurm

The cluster wrapper uses one GPU, 16 CPUs, the `gpu` partition, the
`etechnik_gpu` account, and a three-day limit:

```bash
sbatch sj_timex_table2 --seed 42
sbatch sj_timex_table2 --seeds "42 43"
```

It stages only `SeqCombMV`, `LowVarDetect`, and source into job-local scratch.
The virtual environment remains at the submission location, so initialize it
and the WinIT submodule before submission. A run is archived as:

```text
cluster_runs/timex_table2_<job-id>/
└── seed_<seed>/
    ├── results/   # command logs, JSON, CSV, and Markdown summaries
    └── models/
        ├── seqcomb_mv/
        └── lowvardetect/
```

Exit, failure, timeout, SIGHUP, SIGINT, and SIGTERM all invoke the archive sync,
so partial logs and checkpoints are retained. The Slurm output is written as
`timex-table2-<job-id>.out` in the submission directory and does not require a
pre-existing `logs/` directory.
