"""Build compact CSV and Markdown comparisons from Table 2 JSON results."""

import argparse
import csv
import json
from pathlib import Path


PUBLISHED = {
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
}
METRICS = ("auprc", "aup", "aur")


def main(results_dir):
    rows = []
    for path in sorted(results_dir.glob("*_results.json")):
        record = json.loads(path.read_text())
        dataset = "lowvar" if record["dataset"] == "lowvardetect" else record["dataset"]
        method = record["method"]
        if dataset not in PUBLISHED or method not in PUBLISHED[dataset]:
            continue
        for index, metric in enumerate(METRICS):
            reproduced = record["cross_validation"]["metrics"][metric]
            paper = PUBLISHED[dataset][method][index]
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
    csv_path = results_dir / "table2_summary.csv"
    with csv_path.open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=rows[0].keys()
            if rows
            else (
                "dataset",
                "method",
                "metric",
                "reproduced_mean",
                "fold_standard_error",
                "published",
                "difference",
            ),
        )
        writer.writeheader()
        writer.writerows(rows)
    md = [
        "# Table 2 reproduction summary",
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
    (results_dir / "table2_summary.md").write_text("\n".join(md) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("results_dir", type=Path)
    main(parser.parse_args().results_dir)
