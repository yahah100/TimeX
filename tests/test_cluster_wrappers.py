"""Exercise cluster source staging locally without submitting Slurm jobs."""

import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest
import uuid

ROOT = Path(__file__).resolve().parents[1]


@unittest.skipUnless(shutil.which("rsync"), "Cluster staging requires rsync")
class ClusterWrapperChecks(unittest.TestCase):
    def test_all_seeds_use_archived_source_despite_checkout_edits(self):
        for table, wrapper, prefix, datasets in [
            (1, "sj_timex", "timex", ("FreqShape", "SeqCombSingle")),
            (2, "sj_timex_table2", "timex_table2", ("SeqCombMV", "LowVarDetect")),
        ]:
            with self.subTest(table=table), tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                submit, project = root / "submit", root / "project"
                for relative in [
                    ".venv/bin/activate",
                    "pyproject.toml",
                    "txai/baselines/WinIT/winit/explainer/winitexplainers.py",
                ]:
                    path = submit / relative
                    path.parent.mkdir(parents=True, exist_ok=True)
                    path.write_text(":\n" if relative.endswith("activate") else "")
                for dataset in datasets:
                    (project / "dataset" / dataset).mkdir(parents=True)
                (submit / "recipe.txt").write_text("submitted recipe\n")
                (submit / "old_checkpoint.pt").write_text(
                    "must not enter source snapshot"
                )
                runner = submit / f"run_table{table}.sh"
                runner.write_text(
                    """#!/usr/bin/env bash
set -euo pipefail
task_results=''
task_models=''
while [[ $# -gt 0 ]]; do
    case "$1" in
        --results-dir) task_results="$2"; shift 2 ;;
        --models-dir) task_models="$2"; shift 2 ;;
        *) shift ;;
    esac
done
mkdir -p "$task_results" "$task_models"
cp recipe.txt "$task_results/recipe.txt"
cp recipe.txt "$task_models/checkpoint.pt"
# Simulate edits to the live checkout between sequential seeds.
printf 'edited after first seed\n' > "$SLURM_SUBMIT_DIR/recipe.txt"
"""
                )
                runner.chmod(0o755)
                job_id = str(uuid.uuid4().int)
                env = dict(
                    os.environ,
                    SLURM_JOB_ID=job_id,
                    SLURM_SUBMIT_DIR=str(submit),
                    TIMEX_CLUSTER_PROJECT_DIR=str(project),
                )
                completed = subprocess.run(
                    ["bash", str(ROOT / wrapper), "--seeds", "42 43"],
                    env=env,
                    capture_output=True,
                    text=True,
                    timeout=30,
                )
                self.assertEqual(
                    completed.returncode, 0, completed.stdout + completed.stderr
                )
                archive = project / "cluster_runs" / f"{prefix}_{job_id}"
                self.assertEqual(
                    (archive / "source/recipe.txt").read_text(), "submitted recipe\n"
                )
                self.assertFalse((archive / "source/old_checkpoint.pt").exists())
                self.assertEqual(
                    (submit / "recipe.txt").read_text(), "edited after first seed\n"
                )
                for seed in (42, 43):
                    self.assertEqual(
                        (archive / f"seed_{seed}/results/recipe.txt").read_text(),
                        "submitted recipe\n",
                    )
                    self.assertEqual(
                        (archive / f"seed_{seed}/models/checkpoint.pt").read_text(),
                        "submitted recipe\n",
                    )

    def test_retired_options_are_rejected_before_staging(self):
        for option in ("--protocol", "--predictor-attempts", "--predictor-min-f1"):
            completed = subprocess.run(
                ["bash", str(ROOT / "sj_timex_table2"), option, "unused"],
                capture_output=True,
                text=True,
            )
            self.assertEqual(completed.returncode, 2)
            self.assertIn(f"unknown option: {option}", completed.stderr)


if __name__ == "__main__":
    unittest.main()
