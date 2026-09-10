"""Exercise real synthetic training/evaluation CLIs with temporary, diagnostic data.

Run from the repository root with:
    PYTHONPATH=. uv run python tests/smoke_synth_cli.py

Each stage trains for one epoch and evaluates two examples on CPU. This checks
integration, not benchmark scores. Outputs stay under a printed temporary path.
"""

import argparse
import csv
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import tempfile

import torch

from experiments.synth_workflow import DATASETS, METHODS
from txai.synth_data.generate_spikes import SpikeTrainDataset

ROOT = Path(__file__).resolve().parents[1]
SHAPES = {
    "freqshape": (50, 1),
    "seqcomb_uv": (200, 1),
    "seqcomb_mv": (200, 4),
    "lowvar": (200, 2),
}


def make_split(path: Path, length: int, features: int) -> None:
    """Create shape-compatible signals and nonempty attribution targets."""
    rng = torch.Generator().manual_seed(42)

    def samples(count):
        x = torch.randn(length, count, features, generator=rng) * 0.2
        times = torch.arange(1, length + 1).float().unsqueeze(1).repeat(1, count)
        y = torch.arange(count) % 4
        mask = torch.zeros_like(x)
        for index, label in enumerate(y.tolist()):
            start = (label + 1) * (length // 6)
            feature = label % features
            x[start : start + 5, index, feature] += label + 1
            if label:
                mask[start : start + 5, index, feature] = 1
        return (x, times, y), mask

    train, _ = samples(64)
    val, _ = samples(16)
    test, gt = samples(8)
    dataset = SpikeTrainDataset.__new__(SpikeTrainDataset)
    dataset.X, dataset.times, dataset.y = train
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"train_loader": dataset, "val": val, "test": test, "gt_exps": gt}, path)


def main(datasets: list[str], methods: list[str]) -> Path:
    """Run actual subprocess entry points, retaining logs for failures."""
    if (
        "winit" in methods
        and not (
            ROOT / "txai/baselines/WinIT/winit/explainer/winitexplainers.py"
        ).is_file()
    ):
        raise RuntimeError(
            "Initialize WinIT first: git submodule update --init --recursive txai/baselines/WinIT"
        )
    output = Path(tempfile.mkdtemp(prefix="timex-cli-smoke-"))
    env = dict(
        os.environ,
        CUDA_VISIBLE_DEVICES="",
        OMP_NUM_THREADS="1",
        MKL_NUM_THREADS="1",
        MPLBACKEND="Agg",
        MPLCONFIGDIR=str(output / "matplotlib"),
        PYTHONPATH=str(ROOT),
        TIMEX_DATA_ROOT=str(output / "data"),
    )
    print(f"Diagnostic artifacts: {output}", flush=True)

    def run(script, options, log):
        command = [sys.executable, str(ROOT / script), *map(str, options)]
        with log.open("w") as handle:
            completed = subprocess.run(
                command,
                cwd=ROOT,
                env=env,
                stdout=handle,
                stderr=subprocess.STDOUT,
                timeout=600,
            )
        if completed.returncode:
            raise RuntimeError(
                f"{script} failed; see {log}\n" + log.read_text()[-5000:]
            )
        print(f"PASS {log.parent.name}/{log.stem}", flush=True)

    for dataset in datasets:
        directory, experiment, _ = DATASETS[dataset]
        make_split(output / "data" / directory / "split=1.pt", *SHAPES[dataset])
        folder = output / dataset
        models = folder / "models"
        models.mkdir(parents=True)
        shared = [
            "--seed",
            42,
            "--split-no",
            1,
            "--models-path",
            models,
            "--data-path",
            output / "data" / directory,
            "--epochs",
            1,
        ]
        # Diagnostic predictors deliberately bypass the publication-quality gate.
        extra = (
            ["--original-predictor"]
            if dataset == "seqcomb_mv"
            else (
                ["--max-attempts", 1, "--min-val-f1", 0] if dataset == "lowvar" else []
            )
        )
        predictor_name = (
            "Scomb_transformer"
            if dataset in {"freqshape", "seqcomb_uv"}
            else "transformer"
        )
        if set(methods) != {"sgt+grad"}:
            run(
                f"experiments/{experiment}/train_transformer.py",
                shared + extra,
                folder / "predictor.log",
            )
        if "ours" in methods:
            run(
                f"experiments/{experiment}/bc_model_ptype.py",
                shared,
                folder / "timex_train.log",
            )
        for method in ("cortx", "sgt+grad"):
            if method not in methods:
                continue
            stage = "sgt" if method == "sgt+grad" else method
            run(
                "experiments/other_baselines/train_synth_baselines.py",
                [
                    "--dataset",
                    dataset,
                    "--method",
                    stage,
                    "--seed",
                    42,
                    "--split-no",
                    1,
                    "--models-path",
                    models,
                    "--data-root",
                    output / "data",
                    "--encoder-epochs",
                    1,
                    "--decoder-epochs",
                    1,
                    "--sgt-epochs",
                    1,
                ],
                folder / f"{stage}_train.log",
            )
        if "winit" in methods:
            run(
                "experiments/evaluation/winit_wrapper.py",
                [
                    "--dataset",
                    experiment,
                    "--models_path",
                    models,
                    "--data_path",
                    output / "data",
                    "--seed",
                    42,
                    "--split-no",
                    1,
                    "--epochs",
                    1,
                ],
                folder / "winit_train.log",
            )
        for method in methods:
            prefix = {"ours": "bc_full", "cortx": "cortx", "sgt+grad": "sgt"}.get(
                method, predictor_name
            )
            result = folder / f"{dataset}_{method.replace('+', '_')}_results.json"
            run(
                "experiments/evaluation/saliency_exp_synth.py",
                [
                    "--dataset",
                    experiment,
                    "--exp_method",
                    method,
                    "--split_no",
                    1,
                    "--model_path",
                    models / f"{prefix}_split=1.pt",
                    "--seed",
                    42,
                    "--data_path",
                    output / "data",
                    "--results-json",
                    result,
                    "--max-samples",
                    2,
                    "--no-progress",
                ],
                folder / f"{method}_evaluate.log",
            )
            record = json.loads(result.read_text())
            assert record["completion_status"] == "diagnostic", result
            assert (
                len(record["folds"]) == 1 and record["folds"][0]["n_samples"] == 2
            ), result
            assert all(
                math.isfinite(v) for v in record["folds"][0]["metrics"].values()
            ), result
        table = 1 if dataset in {"freqshape", "seqcomb_uv"} else 2
        run(
            "experiments/evaluation/summarize_synth.py",
            ["--table", table, folder],
            folder / "summary.log",
        )
        rows = list(csv.DictReader((folder / f"table{table}_summary.csv").open()))
        populated = [r for r in rows if r["reproduced_mean"]]
        assert len(populated) == 3 * len(methods), folder
        assert all(r["row_status"] == "diagnostic" for r in populated), folder
    print(
        f"Completed {len(datasets)} datasets × {len(methods)} methods; diagnostic outputs only.",
        flush=True,
    )
    return output


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--datasets", nargs="+", choices=DATASETS, default=list(DATASETS)
    )
    parser.add_argument("--methods", nargs="+", choices=METHODS, default=list(METHODS))
    args = parser.parse_args()
    main(args.datasets, args.methods)
