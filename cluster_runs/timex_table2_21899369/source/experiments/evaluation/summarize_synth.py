"""Build compact CSV and Markdown comparisons from synthetic-benchmark JSON results."""

import argparse
import csv
import json
from pathlib import Path


PUBLISHED_BY_TABLE = {
    1: {
        "freqshape": {
            "ig": (0.7516, 0.6912, 0.5975),
            "dyna": (0.2201, 0.2952, 0.5037),
            "winit": (0.5071, 0.5546, 0.4557),
            "cortx": (0.6978, 0.4938, 0.3261),
            "sgt+grad": (0.5312, 0.4138, 0.3931),
            "ours": (0.8324, 0.7219, 0.6381),
        },
        "seqcomb_uv": {
            "ig": (0.5760, 0.8157, 0.2868),
            "dyna": (0.4421, 0.8782, 0.1029),
            "winit": (0.4568, 0.7872, 0.2253),
            "cortx": (0.5643, 0.8241, 0.1749),
            "sgt+grad": (0.5731, 0.7828, 0.2136),
            "ours": (0.7124, 0.9411, 0.3380),
        },
    },
    2: {
        "seqcomb_mv": {
            "ig": (0.3298, 0.7483, 0.2581),
            "dyna": (0.3136, 0.5481, 0.1953),
            "winit": (0.2809, 0.7594, 0.2077),
            "cortx": (0.3629, 0.5625, 0.3457),
            "sgt+grad": (0.4893, 0.4970, 0.4289),
            "ours": (0.6878, 0.8326, 0.3872),
        },
        "lowvar": {
            "ig": (0.8691, 0.4827, 0.8165),
            "dyna": (0.1391, 0.1640, 0.2106),
            "winit": (0.1667, 0.1140, 0.3842),
            "cortx": (0.4983, 0.3281, 0.4711),
            "sgt+grad": (0.3449, 0.2133, 0.3528),
            "ours": (0.8673, 0.5451, 0.9004),
        },
    },
}
METRICS = ("auprc", "aup", "aur")

# The evaluator's --dataset values differ from the paper's dataset names.
DATASET_ALIASES = {"lowvardetect": "lowvar", "scs_better": "seqcomb_uv"}

FIELDNAMES = (
    "dataset",
    "method",
    "metric",
    "reproduced_mean",
    "fold_standard_error",
    "historical_pooled_standard_error",
    "published",
    "difference",
    "within_tolerance",
    "protocol",
    "base_seed",
    "n_folds",
    "completion_status",
    "provenance_status",
    "row_status",
)


def comparison_rows(results_dir, table):
    """Include every published row; do not label partial or untraceable results matched."""
    published = PUBLISHED_BY_TABLE[table]
    records = {}
    for path in sorted(results_dir.glob("*_results.json")):
        record = json.loads(path.read_text())
        dataset = DATASET_ALIASES.get(record["dataset"], record["dataset"])
        key = (dataset, record["method"])
        if key in records:
            raise ValueError(
                f"Duplicate result identity {key}; summarize protocols/seeds separately"
            )
        records[key] = record
    rows = []
    for dataset, methods in published.items():
        for method, reference in methods.items():
            record = records.get((dataset, method), {})
            cv = record.get("cross_validation", {})
            metrics = cv.get("metrics", {})
            completion = record.get(
                "completion_status", "historical_unverified" if record else "missing"
            )
            provenance = record.get("provenance_status", "unknown")
            differences = [
                metrics[k]["mean"] - reference[i]
                for i, k in enumerate(METRICS)
                if k in metrics
            ]
            qualified = all(
                q.get("qualified") and q.get("validation_macro_f1", 0) >= 0.95
                for q in record.get("predictor_quality", [])
            )
            traceable = (
                record.get("protocol") == "repaired-v1"
                and len(record.get("provenance", [])) == 5
                and len(record.get("predictor_quality", [])) == 5
                and qualified
            )
            if provenance.startswith("unresolved"):
                status = "unresolved provenance"
            elif completion != "complete" or cv.get("n_folds") != 5 or not traceable:
                status = (
                    completion
                    if not traceable or completion != "complete"
                    else "partial"
                )
                if status == "complete":
                    status = "unverified"
            elif len(differences) == 3 and all(abs(d) <= 0.05 for d in differences):
                status = "matched"
            else:
                status = "outside tolerance"
            for index, metric in enumerate(METRICS):
                reproduced = metrics.get(metric, {})
                mean = reproduced.get("mean")
                difference = mean - reference[index] if mean is not None else None
                rows.append(
                    dict(
                        dataset=dataset,
                        method=method,
                        metric=metric.upper(),
                        reproduced_mean=mean,
                        fold_standard_error=reproduced.get("standard_error"),
                        historical_pooled_standard_error=record.get("pooled", {})
                        .get("historical_standard_error", {})
                        .get(metric),
                        published=reference[index],
                        difference=difference,
                        within_tolerance=abs(difference) <= 0.05
                        if difference is not None
                        else None,
                        protocol=record.get(
                            "protocol", "historical" if record else "pending"
                        ),
                        base_seed=record.get("base_seed"),
                        n_folds=cv.get("n_folds", 0),
                        completion_status=completion,
                        provenance_status=provenance,
                        row_status=status,
                    )
                )
    return rows


def main(results_dir, table):
    rows = comparison_rows(results_dir, table)
    results_dir.mkdir(parents=True, exist_ok=True)
    with (results_dir / f"table{table}_summary.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDNAMES)
        writer.writeheader()
        writer.writerows(rows)
    md = [
        f"# Table {table} reproduction summary",
        "",
        "Primary uncertainty is SE across fold means. Historical pooled SE is labelled separately in CSV.",
        "",
        "| Dataset | Method | Metric | Mean ± fold SE | Published | Difference | Protocol / seed | Folds | Row status |",
        "|---|---|---|---:|---:|---:|---|---:|---|",
    ]
    for row in rows:
        mean, se, delta = (
            row["reproduced_mean"],
            row["fold_standard_error"],
            row["difference"],
        )
        value = (
            "pending"
            if mean is None
            else f"{mean:.4f} ± " + (f"{se:.4f}" if se is not None else "n/a")
        )
        difference = "—" if delta is None else f"{delta:+.4f}"
        md.append(
            f'| {row["dataset"]} | {row["method"]} | {row["metric"]} | {value} | {row["published"]:.4f} | {difference} | {row["protocol"]} / {row["base_seed"]} | {row["n_folds"]} | {row["row_status"]} |'
        )
    (results_dir / f"table{table}_summary.md").write_text("\n".join(md) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("results_dir", type=Path)
    parser.add_argument("--table", type=int, choices=(1, 2), required=True)
    args = parser.parse_args()
    main(args.results_dir, args.table)
