# Repository Guidelines

## Project Structure & Module Organization

`txai/` contains the reusable Python package. Core TimeX models live in `txai/models/`, training loops in `txai/trainers/`, data and evaluation helpers in `txai/utils/`, synthetic generators in `txai/synth_data/`, and plotting code in `txai/vis/`. Baseline implementations are vendored under `txai/baselines/`. Dataset-specific entry points and evaluation scripts are organized under `experiments/<dataset>/` and `experiments/evaluation/`. Use `notebooks/` for exploratory analysis only. See `reproducibility.md` for experiment workflows; large datasets and trained checkpoints should remain outside Git.

## Build, Test, and Development Commands

- `uv sync` creates the Python 3.10 environment and installs `txai` in editable mode with its core dependencies.
- `uv sync --extra notebooks` also installs Jupyter tooling for exploratory work.
- `uv run python <script>` runs commands inside the managed environment without manual activation.
- `python experiments/<dataset>/train_transformer.py` trains a reference predictor; review the script's dataset and output paths first.
- `python experiments/<dataset>/bc_model_ptype.py` trains TimeX for a dataset after configuring its predictor checkpoint.
- `python experiments/evaluation/saliency_exp_synth.py --help` inspects evaluation options before running a full experiment.

There is no separate build step or repository-wide automated test command.

## Coding Style & Naming Conventions

Use Python with four-space indentation and PEP 8 conventions. Follow existing names: `snake_case` for functions, variables, and modules; `PascalCase` for classes; and uppercase names for true constants. Keep imports grouped as standard library, third-party, then `txai` modules. Add type hints and short docstrings when introducing public or non-obvious APIs. `black` is pinned in `requirements.txt`; format touched Python files with `black <paths>` and avoid reformatting unrelated research code.

## Testing Guidelines

No dedicated test suite or coverage threshold is currently configured. Validate changes with the smallest relevant experiment or evaluation script, preferably on a reduced sample or epoch count. For numerical code, add focused assertions for tensor shapes, device placement, finite losses, and deterministic outputs where seeds permit. Document required datasets, checkpoints, GPU assumptions, and the exact validation command in the pull request.

## Commit & Pull Request Guidelines

Recent history uses short, imperative summaries such as `Update README.md` and `Calculates IoU`. Keep commits focused and describe the affected behavior more specifically when possible, for example `Fix padding mask in transformer encoder`. Pull requests should explain the motivation, list commands run, link related issues, and call out data or checkpoint prerequisites. Include plots or screenshots when visualization or reported results change, and do not commit generated models, large datasets, or notebook checkpoint directories.
