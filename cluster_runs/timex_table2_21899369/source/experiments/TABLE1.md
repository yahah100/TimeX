# Reproducing Table 1

Table 1 evaluates FreqShapes and SeqComb-UV over the five published folds with
TimeX, Integrated Gradients, Dynamask, WinIT, CoRTX, and SGT + Grad. The
Dataverse directory `SeqCombSingle/` is the SeqComb-UV data used by the
`scs_better` experiment scripts.

All commands below run from the repository root. Data defaults to `dataset/`;
set `TIMEX_DATA_ROOT=/another/path` to override it.

The complete functional workflow can be launched with:

```bash
./run_table1.sh
```

Runs use base seed `0` by default. Pass `--seed N` to select another seed;
fold `i` uses `N + i - 1`, making each fold reproducible independently.

Use `./run_table1.sh --help` for dataset, method, and stage selection. Every
command has its own log; evaluations write JSON records plus `table1_summary.csv`
and `table1_summary.md` into the results directory. Training stages skip any
method whose five split checkpoints already exist, so adding one method to an
existing run does not retrain the predictors.

## First smoke test

Confirm that both datasets and their ground-truth masks load:

```bash
uv run python -c "from txai.utils.data import process_Synth; from txai.utils.constants import dataset_path; print(process_Synth(1, base_path=dataset_path('FreqShape'))['test'][0].shape); print(process_Synth(1, base_path=dataset_path('SeqCombSingle'))['test'][0].shape)"
```

## Train reference predictors and TimeX

```bash
uv run python experiments/freqshape/train_transformer.py
uv run python experiments/freqshape/bc_model_ptype.py
uv run python experiments/scs_better/train_transformer.py
uv run python experiments/scs_better/bc_model_ptype.py
```

These reproduce the paper settings: five folds; 100/200 predictor epochs for
FreqShapes/SeqComb-UV; and 50 TimeX epochs. Checkpoints are written below each
experiment's `models/` directory.

## Train the CoRTX and SGT baselines

```bash
uv run python experiments/other_baselines/train_synth_baselines.py --dataset freqshape --method cortx --seed 42
uv run python experiments/other_baselines/train_synth_baselines.py --dataset freqshape --method sgt --seed 42
uv run python experiments/other_baselines/train_synth_baselines.py --dataset seqcomb_uv --method cortx --seed 42
uv run python experiments/other_baselines/train_synth_baselines.py --dataset seqcomb_uv --method sgt --seed 42
```

CoRTX warm-starts from the reference predictor and trains 100 encoder epochs at
temperature 0.7 plus 50 decoder epochs; it therefore needs
`Scomb_transformer_split={i}.pt` to exist first. SGT trains its own classifier
from scratch and gets the same optimizer settings and epoch budget as the
predictor it stands in for (100 for FreqShapes, 200 for SeqComb-UV).

## Evaluate explanations

Run all folds by supplying the fold-1 checkpoint; the evaluator substitutes
`split=2` through `split=5` automatically:

```bash
uv run python experiments/evaluation/saliency_exp_synth.py --dataset freqshape --exp_method ours --split_no -1 --model_path experiments/freqshape/models/bc_full_split=1.pt
uv run python experiments/evaluation/saliency_exp_synth.py --dataset freqshape --exp_method ig --split_no -1 --model_path experiments/freqshape/models/Scomb_transformer_split=1.pt
uv run python experiments/evaluation/saliency_exp_synth.py --dataset freqshape --exp_method dyna --split_no -1 --model_path experiments/freqshape/models/Scomb_transformer_split=1.pt

uv run python experiments/evaluation/saliency_exp_synth.py --dataset scs_better --exp_method ours --split_no -1 --model_path experiments/scs_better/models/bc_full_split=1.pt
uv run python experiments/evaluation/saliency_exp_synth.py --dataset scs_better --exp_method ig --split_no -1 --model_path experiments/scs_better/models/Scomb_transformer_split=1.pt
uv run python experiments/evaluation/saliency_exp_synth.py --dataset scs_better --exp_method dyna --split_no -1 --model_path experiments/scs_better/models/Scomb_transformer_split=1.pt

uv run python experiments/evaluation/saliency_exp_synth.py --dataset freqshape --exp_method cortx --split_no -1 --model_path experiments/freqshape/models/cortx_split=1.pt
uv run python experiments/evaluation/saliency_exp_synth.py --dataset freqshape --exp_method sgt+grad --split_no -1 --model_path experiments/freqshape/models/sgt_split=1.pt
```

CoRTX and SGT + Grad take their *own* checkpoints, not the predictor's.

WinIT additionally requires `git submodule update --init --recursive` and a
trained generator per fold via `experiments/evaluation/winit_wrapper.py`.

The committed `cortx_exp.py` and `train_SGT.py` are one-fold research snapshots
with hard-coded absolute paths and are not used. The reimplementations in
`txai/baselines/synth_baselines.py` stand in: CoRTX uses a local symmetric
in-batch InfoNCE because the legacy snapshot's PyGCL dependency is absent, and
SGT is evaluated with absolute gradients from the trained model rather than with
the masked training input. Both caveats also apply to Table 2.
