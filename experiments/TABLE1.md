# Reproducing Table 1

Table 1 evaluates FreqShapes and SeqComb-UV over the five published folds. The
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

Use `./run_table1.sh --help` for dataset, method, and stage selection.

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
```

WinIT additionally requires `git submodule update --init --recursive` and a
trained generator per fold via `experiments/evaluation/winit_wrapper.py`.
The committed CoRTX and SGT scripts are incomplete research snapshots: they
hard-code one fold and do not currently provide a five-fold Table 1 CLI.
