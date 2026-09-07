# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

Repository-wide conventions (structure, style, commit/PR expectations) live in `AGENTS.md`;
this file covers what you need to actually run and modify the experiments.

## Environment

Python is pinned to 3.10 and dependencies are locked in `uv.lock`.

```bash
uv sync                      # env + editable install of txai
uv sync --extra notebooks    # adds Jupyter tooling
uv run python <script>       # run anything inside the env
git submodule update --init --recursive   # required for the WinIT baseline
```

`uvx ty check` is the type checker; `black` (pinned) formats touched files only.
There is no build step and no test suite — validate changes by running the smallest
relevant experiment (see "Smoke tests").

## Datasets

Data is never in the repo (`dataset/` is gitignored). Download from the Harvard Dataverse
(https://doi.org/10.7910/DVN/B0DEQJ) into `dataset/`, or point `TIMEX_DATA_ROOT` elsewhere.
`txai/utils/constants.py` resolves every path through `dataset_path(name)`.

Dataset naming is inconsistent across the three layers, and this is the most common source
of confusion:

| Paper name | Dataverse dir | `--dataset` value | experiment dir |
| --- | --- | --- | --- |
| FreqShapes | `FreqShape` | `freqshape` | `experiments/freqshape` |
| SeqComb-UV | `SeqCombSingle` | `scs_better` | `experiments/scs_better` |
| SeqComb-MV | `SeqCombMV` | `seqcomb_mv` | `experiments/seqcomb_mv` |
| LowVar | `LowVarDetect` | `lowvardetect` | `experiments/lowvardetect` |

Real-world sets (`MITECG`, `Boiler`, `Epilepsy`, `PAM`) follow the same pattern via
`process_MITECG` / `process_Boiler` / `process_Epilepsy` / `process_PAM`.

## Reproduction workflows

The paper tables are reproduced by two runners that wrap the underlying scripts, resolve
data through `TIMEX_DATA_ROOT`, log every command, and skip already-complete checkpoint sets:

```bash
./run_table1.sh                                                  # freqshape + scs_better, base seed 0
./run_table2.sh --datasets seqcomb_mv --methods ours,ig --stage evaluate
sbatch sj_timex --seed 42          # Table 1 on Slurm
sbatch sj_timex_table2 --seeds "42 43"
```

`--stage train|evaluate|all` splits the pipeline. Both tables run the same six methods.
The committed CoRTX and SGT scripts in `experiments/other_baselines/` are one-fold research
snapshots and are not used; `cortx` and `sgt+grad` instead go through
`experiments/other_baselines/train_synth_baselines.py` plus reimplementations in
`txai/baselines/synth_baselines.py` (in-batch InfoNCE for CoRTX, absolute input gradients
for SGT).

`experiments/TABLE1.md` and `experiments/TABLE2.md` document the workflows in detail;
`results_reproduction.md` records the results already obtained and where they diverge from the paper.

### The three-stage pipeline

Everything is five-fold; fold `i` (1..5) runs under seed `base_seed + i - 1` via
`seed_everything` in `txai/utils/reproducibility.py`.

1. `experiments/<dataset>/train_transformer.py` — trains the reference predictor to explain.
   Writes `Scomb_transformer_split={i}.pt` (Table 1 dirs) or `transformer_split={i}.pt`
   (Table 2 dirs) into `experiments/<dataset>/models/`.
2. `experiments/<dataset>/bc_model_ptype.py` — trains TimeX against that predictor.
   Writes `bc_full_split={i}.pt`; ablation flags (`--no_ste`, `--simclr`, `--no_la`,
   `--no_con`, `--eq_ge`, ...) change the filename via that script's `naming_convention`.
3. `experiments/evaluation/saliency_exp_synth.py` — computes AUPRC/AUP/AUR (and IoU where
   ground truth allows) against ground-truth saliency. `--split_no -1` rewrites `split=N`
   in `--model_path` and runs all five folds, reporting fold mean ± standard error.
   `experiments/evaluation/occlusion_exp.py` takes the same arguments for the occlusion study.

Pass the TimeX checkpoint for `--exp_method ours`; pass the *predictor* checkpoint for every
other method. Results JSON (`--results-json`) carries `schema_version`, per-fold records,
`pooled`, and `cross_validation`; `summarize_synth.py --table 1|2` turns a directory of
those into CSV/Markdown next to the published numbers.

### Smoke tests

```bash
TIMEX_MAX_SAMPLES=2 ./run_table2.sh --datasets seqcomb_mv --methods ours --stage evaluate
uv run python experiments/evaluation/saliency_exp_synth.py --dataset freqshape \
    --exp_method ig --split_no 1 --model_path experiments/freqshape/models/Scomb_transformer_split=1.pt \
    --max-samples 8 --no-progress
```

Full runs are expensive: Dynamask optimizes a mask per test sample and WinIT repeatedly
samples counterfactuals, so plan on a GPU and up to several days.

## Architecture

`TimeXModel` (`txai/models/bc_model.py`) is the whole method. It holds a *frozen* copy of
the reference predictor plus a trainable explanation branch: a mask generator produces a
straight-through-estimated binary mask over `(time, feature)`, the masked input is re-encoded,
and the resulting explanation embedding is pushed to mirror the reference model's geometry.
`AblationParameters` toggles every ablation from the paper. Prototypes
(`ptype_assimilation`, `txai/prototypes/`) give the landmark explanations.

- **Loss** — model behavior consistency lives in `txai/utils/predictors/loss_cl.py`:
  `EmbedConsistencyLoss` (pairwise-similarity matching against the reference embeddings) and
  `LabelConsistencyLoss` (prediction agreement). `--simclr`/`--no_la`/`--no_con` swap these.
  Mask regularizers (`GSATLoss_Extended`, `ConnectLoss_Extended`) come from
  `txai/utils/predictors/loss.py` and are weighted by `loss_weight_dict`.
- **Training loop** — `txai/trainers/train_mv6_consistency.py`. Model selection is not by
  validation loss but by a `selection_criterion` from
  `txai/utils/predictors/select_models.py` (`simloss_on_val_wboth` and friends), which must
  match the chosen `sim_criterion` combination.
- **Predictor training** — `txai/trainers/train_transformer.py` with
  `TransformerMVTS` (`txai/models/encoders/transformer_simple.py`).
- **Explainer dispatch** — `get_explainer` in `txai/utils/experimental.py` maps
  `--exp_method` to a callable `(model, x, time, y) -> attribution`. New baselines plug in
  there; TimeX, CoRTX, SGT and WinIT are instead special-cased in `saliency_exp_synth.py`.
- **Metrics** — `ground_truth_xai_eval` and `ground_truth_IoU` in `txai/utils/evaluation.py`.
- **Vendored baselines** — `txai/baselines/`: `WinIT` and `FIT` are git submodules,
  `Dynamask` and `SGT` are vendored copies.

### Conventions that bite

- Tensors are **time-first**, `(T, B, d)`. Captum and Dynamask want batch-first; the
  `captum_input` flag on model forwards and per-explainer `transpose(0, 1)` calls handle the
  conversion. Getting this wrong yields plausible-looking but wrong attributions.
- `saliency_exp_synth.py` has its own `get_model(args, X)` that **re-declares the predictor
  architecture per dataset**. If you change transformer hyperparameters in a
  `train_transformer.py`, mirror them there or `load_state_dict` will fail (or silently
  mismatch shapes). The same duplication exists in
  `txai/baselines/synth_baselines.py::DATASET_CONFIGS`, which the CoRTX and SGT baselines
  build their transformers from.
- `MaskGenerator.forward` returns the mask **batch-first for univariate** inputs and
  time-first for multivariate ones. Route it through
  `txai/baselines/synth_baselines.py::cortx_mask`, which normalizes to time-first.
- `TimeXModel.get_saliency_explanation` returns a dict; the attribution is `out['mask_in']`.
  The full `forward` returns a much larger dict used only during training.
- The per-dataset `bc_model_ptype.py` scripts are near-copies with hard-coded hyperparameters.
  A change to the method usually has to be applied to each of the four (plus `epilepsy`,
  `mitecg_*`, `PAM`, `Boiler` when those matter).
- `results/`, `*.pt`, `*.out` are gitignored; archived cluster runs (`timex_*_<jobid>/`) hold
  the logs and checkpoints from previous jobs.
