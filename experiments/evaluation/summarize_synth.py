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
    "published",
    "difference",
)


def main(results_dir, table):
    published = PUBLISHED_BY_TABLE[table]
    rows = []
    for path in sorted(results_dir.glob("*_results.json")):
        record = json.loads(path.read_text())
        dataset = DATASET_ALIASES.get(record["dataset"], record["dataset"])
        method = record["method"]
        if dataset not in published or method not in published[dataset]:
            continue
        for index, metric in enumerate(METRICS):
            reproduced = record["cross_validation"]["metrics"][metric]
            paper = published[dataset][method][index]
            rows.append(
                {
                    "dataset": dataset,
                    "method": method,
                    "metric": metric.upper(),
                    "reproduced_mean": reproduced["mean"],
                    "fold_standard_error": reproduced["standard_error"],
                    "published": paper,
                    "difference": reproduced["mean"] - paper,
                }
            )
    csv_path = results_dir / f"table{table}_summary.csv"
    with csv_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDNAMES)
        writer.writeheader()
        writer.writerows(rows)
    md = [
        f"# Table {table} reproduction summary",
        "",
        "| Dataset | Method | Metric | Reproduced mean ± fold SE | Published | Difference |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for row in rows:
        se = row["fold_standard_error"]
        reproduced = (
            f'{row["reproduced_mean"]:.4f} ± {se:.4f}'
            if se is not None
            else f'{row["reproduced_mean"]:.4f} ± n/a'
        )
        md.append(
            f'| {row["dataset"]} | {row["method"]} | {row["metric"]} | {reproduced} | {row["published"]:.4f} | {row["difference"]:+.4f} |'
        )
    (results_dir / f"table{table}_summary.md").write_text("\n".join(md) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("results_dir", type=Path)
    parser.add_argument("--table", type=int, choices=(1, 2), required=True)
    args = parser.parse_args()
    main(args.results_dir, args.table)
