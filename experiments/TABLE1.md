# Reproducing Table 1

FreqShapes and SeqComb-UV use all six methods: TimeX, IG, Dynamask, WinIT, CoRTX
and SGT + Grad. The fixed recipes correspond to the completed seed-42 rows in
[the report](../results_reproduction.md).

```bash
uv sync
git submodule update --init --recursive
./run_table1.sh --help
./run_table1.sh --seed 42 --dry-run
./run_table1.sh --seed 42
# On the configured GPU cluster:
sbatch sj_timex --seed 42
```

`TIMEX_DATA_ROOT` defaults to `dataset/`, containing `FreqShape/split=1.pt` through
`split=5.pt` and `SeqCombSingle/split=1.pt` through `split=5.pt`. Set it explicitly
for another location. The cluster wrapper uses its existing persistent data root.

The common runner supports `--datasets freqshape,seqcomb_uv`, `--methods`,
`--folds 1,2,3,4,5`, `--stage all|train|evaluate`, `--seed`, `--data-root`,
`--models-dir` and `--results-dir`. Default base seed is **42**; fold i uses
`seed+i-1`. Results default to `results/table1/selected-v1/seed_42/` and models to
`results/table1/models/selected-v1/seed_42/`. Predictors are shared only between
methods with identical dependency recipes. Reuse requires matching artifact,
source, configuration and data hashes. Different seeds never reuse checkpoints.

| Stage | FreqShapes | SeqComb-UV |
|---|---|---|
| Predictor | Original single initialization; 100 epochs | Original single initialization; 200 epochs |
| TimeX | Original connectivity/dropout, no effective clipping; 50 epochs | Same; 50 epochs |
| SGT | CE + detached KL, final checkpoint; 100 epochs | Same; 200 epochs |
| CoRTX | 100 encoder / 50 decoder epochs | Same |
| WinIT | 1000 generator epochs | Same |

Architectures, optimizers and validation selection for predictors/TimeX remain
those used by the original scripts. CoRTX and SGT use their own checkpoints.
SGT has no reference-predictor dependency. CoRTX preserves the reconstruction
recipe despite its remaining score gap. See [findings](../reproduction_findings.md).

For a reduced attribution check after fresh training:

```bash
./run_table1.sh --datasets freqshape --methods ours --folds 1 --stage evaluate --max-samples 2
```

Reduced evaluations and incomplete folds never count as full results. Old
checkpoint formats are not migrated; use fresh artifacts from the cleaned code.
The report preserves the measured original results until a new GPU run verifies
numerical reproduction.

```bash
PYTHONPATH=. uv run python -m unittest discover -s tests -p 'test_*.py' -v
PYTHONPATH=. uv run python experiments/evaluation/report_synth.py --check
```

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
