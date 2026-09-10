"""Render the retained, source-linked reproduction tables without loading checkpoints."""

import argparse
import hashlib
import json
import math
from pathlib import Path

from experiments.evaluation.summarize_synth import PUBLISHED_BY_TABLE, METRICS
from experiments.synth_workflow import METHODS, TABLE_DATASETS, recipe_name

ROOT = Path(__file__).resolve().parents[2]
NAMES = {
    "freqshape": "FreqShapes",
    "seqcomb_uv": "SeqComb-UV",
    "seqcomb_mv": "SeqComb-MV",
    "lowvar": "LowVar",
}
METHOD_NAMES = {
    "ours": "TimeX",
    "ig": "IG",
    "dyna": "Dynamask",
    "winit": "WinIT",
    "cortx": "CoRTX",
    "sgt+grad": "SGT + Grad",
}


def validate(data: dict, verify_sources: bool = False) -> None:
    """Reject incomplete, mixed-seed or inconsistent selected measurements."""
    expected = {
        (table, dataset, method)
        for table, datasets in TABLE_DATASETS.items()
        for dataset in datasets
        for method in METHODS
    }
    keys = [(r["table"], r["dataset"], r["method"]) for r in data["rows"]]
    if len(keys) != len(set(keys)) or set(keys) != expected:
        raise ValueError("Expected exactly one row per dataset and method")
    for row in data["rows"]:
        if (
            row["seed"] != 42
            or row["n_folds"] != 5
            or len(row["folds"]) != 5
            or {f["split"] for f in row["folds"]} != set(range(1, 6))
        ):
            raise ValueError("Selected rows require five folds at base seed 42")
        tolerance = 0.000101 if row["precision"] == "rounded_log_4dp" else 1e-9
        for metric in METRICS:
            stats = row["metrics"][metric]
            values = [f["metrics"][metric] for f in row["folds"]]
            if not all(
                math.isfinite(v)
                for v in values + [stats["mean"], stats["standard_error"]]
            ):
                raise ValueError("Nonfinite metric")
            if abs(sum(values) / 5 - stats["mean"]) > tolerance:
                raise ValueError("Mean disagrees with recorded folds")
    if verify_sources:
        for row in data["rows"] + data["additional_seeds"]:
            path = ROOT / row["source"]
            if hashlib.sha256(path.read_bytes()).hexdigest() != row["source_sha256"]:
                raise ValueError(f"Changed source: {path}")


def render(data: dict) -> str:
    """Render the canonical report from the source-hashed measurement snapshot."""
    validate(data)
    lines = [
        "# TimeX synthetic reproduction results",
        "",
        "All six methods are retained. Each main row is the highest-AUPRC complete "
        "five-fold configuration observed at base seed **42**; AUP and AUR come from "
        "that same run. Exact ties use the traceable completed run. This is a retrospective "
        "selection of observed configurations, not a new uniformly configured benchmark.",
        "",
        "Cells show **mean ± fold SE / paper (difference)**. Higher is better. "
        "“Within ±0.05” requires all three metric differences to satisfy that bound; "
        "it describes numerical agreement only. CoRTX’s multivariate recipe remains unresolved. "
        "Results are historical measurements; the cleaned implementation has not yet been rerun on the cluster.",
        "",
        "Recipes and source artifacts are linked in each row. The compact "
        "[measurement snapshot](experiments/reproduction_results.json) preserves fold values "
        "and source hashes even when the large local archives are absent. "
        "[Findings and discarded variants](reproduction_findings.md) explain the remaining gaps.",
        "",
    ]
    for table in (1, 2):
        lines += [
            f'## Table {table}: {"Univariate" if table == 1 else "Multivariate"} attribution',
            "",
            "| Dataset | Method | AUPRC | AUP | AUR | Paper metric agreement | Recipe / source |",
            "|---|---|---:|---:|---:|---|---|",
        ]
        for row in (r for r in data["rows"] if r["table"] == table):
            dataset, method = row["dataset"], row["method"]
            paper = PUBLISHED_BY_TABLE[table][dataset][method]
            cells = []
            for metric, reference in zip(METRICS, paper):
                stats = row["metrics"][metric]
                mean, se = stats["mean"], stats["standard_error"]
                cells.append(
                    f"{mean:.4f} ± {se:.4f} / {reference:.4f} ({mean-reference:+.4f})"
                )
            close = all(
                abs(row["metrics"][m]["mean"] - p) <= 0.05
                for m, p in zip(METRICS, paper)
            )
            agreement = "Within ±0.05" if close else "Outside ±0.05"
            if table == 2 and method == "cortx":
                agreement += "; provenance unresolved"
            source = f"[{recipe_name(dataset, method)}]({row['source']})"
            lines.append(
                f"| {NAMES[dataset]} | {METHOD_NAMES[method]} | "
                + " | ".join(cells)
                + f" | {agreement} | {source} |"
            )
        lines.append("")
    lines += [
        "Table 1 TimeX/IG/Dynamask/WinIT fold SE is reconstructed from fold means "
        "rounded to four decimal places in the original logs; other SE values use full-precision JSON. "
        "SE describes variation across folds, not across independent training seeds.",
        "",
        "SeqComb-MV TimeX and CoRTX retain the original single-attempt predictors, including weak "
        "folds 3/4. IG, Dynamask and WinIT use the validation-qualified predictors. "
        "SGT trains its own classifier. The predictor differences are part of the selected recipes.",
        "",
        "LowVar TimeX improves over the original run from 0.8371 to 0.8547 AUPRC and "
        "from 0.5070 to 0.6076 AUP. Its AUP is higher than the paper by 0.0625, "
        "so the row remains outside the numerical-agreement tolerance. "
        "SeqComb-MV TimeX remains below the paper: 0.3735 versus 0.6878 AUPRC.",
        "",
        "## Additional LowVar TimeX seeds",
        "",
        "Both completed follow-up seeds are shown separately; neither replaces seed 42.",
        "",
        "| Seed | Folds | AUPRC | AUP | AUR | Source |",
        "|---|---:|---:|---:|---:|---|",
    ]
    for row in data["additional_seeds"]:
        cells = [
            f"{row['metrics'][m]['mean']:.4f} ± {row['metrics'][m]['standard_error']:.4f}"
            for m in METRICS
        ]
        lines.append(
            f"| {row['seed']} | {row['n_folds']} | "
            + " | ".join(cells)
            + f" | [Cluster result]({row['source']}) |"
        )
    lines += [
        "",
        "## SeqComb-UV IoU (Table 10)",
        "",
        "| Method | Reproduction | Paper | Difference |",
        "|---|---:|---:|---:|",
    ]
    for method, reference in [("ours", 0.5214), ("ig", 0.3750), ("dyna", 0.2958)]:
        row = next(
            r
            for r in data["rows"]
            if r["dataset"] == "seqcomb_uv" and r["method"] == method
        )
        mean = row["metrics"]["iou"]["mean"]
        lines.append(
            f"| {METHOD_NAMES[method]} | {mean:.4f} | {reference:.4f} | {mean-reference:+.4f} |"
        )
    lines += [
        "",
        "## Reproduce",
        "",
        "Use the same datasets, seed 42 and all five folds. The runners select the fixed "
        "recipe for each method automatically; no protocol switches are needed. "
        "Training budgets and dataset paths are documented in "
        "[Table 1](experiments/TABLE1.md) and [Table 2](experiments/TABLE2.md).",
        "",
        "```bash",
        "./run_table1.sh --seed 42",
        "./run_table2.sh --seed 42",
        "# Regenerate this Markdown from the recorded measurements:",
        "PYTHONPATH=. uv run python experiments/evaluation/report_synth.py",
        "```",
        "",
        "Use fresh checkpoints from the cleaned code. Old checkpoint formats are not migrated. "
        "Cluster numerical verification is still required; CPU checks validate the retained "
        "recipes and workflow behavior, not a new five-fold result.",
        "",
    ]
    return "\n".join(lines)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--check",
        action="store_true",
        help="Check the committed report without writing",
    )
    parser.add_argument(
        "--verify-sources",
        action="store_true",
        help="Also require and hash the local archived sources",
    )
    args = parser.parse_args()
    data = json.loads((ROOT / "experiments/reproduction_results.json").read_text())
    validate(data, args.verify_sources)
    report = render(data)
    path = ROOT / "results_reproduction.md"
    if args.check:
        if path.read_text() != report:
            raise SystemExit("Report differs; run report_synth.py to regenerate")
    else:
        path.write_text(report)
