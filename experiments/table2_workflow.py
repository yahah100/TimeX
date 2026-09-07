"""Traceable Table 2 dependency runner. Dry runs need only the Python standard library."""

import argparse
import hashlib
import json
import os
from pathlib import Path
import shlex
import shutil
import subprocess
import sys
from datetime import datetime, timezone

ROOT = Path(__file__).resolve().parents[1]
DATASETS = {
    "seqcomb_mv": ("SeqCombMV", "seqcomb_mv", 1000),
    "lowvar": ("LowVarDetect", "lowvardetect", 120),
}
METHODS = ("ours", "ig", "dyna", "winit", "cortx", "sgt+grad")
PROTOCOLS = ("repaired-v1", "legacy-control", "connectivity-control", "sgt-control")


def digest(path):
    """Hash a file, or a directory tree including relative names."""
    path = Path(path)
    h = hashlib.sha256()
    if path.is_dir():
        for child in sorted(p for p in path.rglob("*") if p.is_file()):
            h.update(str(child.relative_to(path)).encode())
            h.update(digest(child).encode())
    else:
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                h.update(chunk)
    return h.hexdigest()


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def sidecar(path):
    return Path(str(path) + ".metadata.json")


def reusable(path, identity):
    """File existence never authorizes reuse; also verify output integrity."""
    try:
        record = json.loads(sidecar(path).read_text())
        return record["identity"] == identity and record["sha256"] == digest(path)
    except (OSError, ValueError, KeyError):
        return False


def quarantine(path):
    """Preserve stale artifacts, attempts and diagnostics before replacing a stage."""
    related = list(path.parent.glob(path.name + "*"))
    if path.name.startswith("transformer_split="):
        related += list(path.parent.glob(path.stem + "*"))
    related = sorted(set(related))
    if related:
        archive = (
            path.parent
            / "stale"
            / datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%f")
        )
        archive.mkdir(parents=True)
        for item in related:
            shutil.move(str(item), archive / item.name)


def source_identity(stage, experiment):
    # Generator identity deliberately excludes predictor/TimeX implementation.
    common = [
        "experiments/table2_workflow.py",
        "txai/utils/data",
        "txai/utils/reproducibility.py",
        "pyproject.toml",
        "uv.lock",
    ]
    if stage == "winit":
        paths = common + [
            "experiments/evaluation/winit_wrapper.py",
            "txai/baselines/WinIT",
        ]
    else:
        paths = common + [
            "txai/models",
            "txai/utils/predictors",
            "txai/utils/functional.py",
        ]
        if stage == "predictor":
            paths += [
                f"experiments/{experiment}/train_transformer.py",
                "txai/trainers/train_transformer.py",
            ]
        elif stage == "ours":
            paths += [
                f"experiments/{experiment}/bc_model_ptype.py",
                "txai/trainers/train_mv6_consistency.py",
            ]
        elif stage in {"cortx", "sgt"}:
            paths += [
                "experiments/other_baselines/train_synth_baselines.py",
                "txai/baselines/synth_baselines.py",
            ]
        else:
            paths += [
                "experiments/evaluation/saliency_exp_synth.py",
                "txai/utils/evaluation.py",
                "txai/utils/experimental.py",
                "txai/baselines",
            ]
    files = set()
    for name in paths:
        path = ROOT / name
        files.update(path.rglob("*.py") if path.is_dir() else [path])
    return {str(p.relative_to(ROOT)): digest(p) for p in sorted(files) if p.is_file()}


def run_stage(path, identity, command, log, args, enrich=None):
    match = reusable(path, identity)
    action = "reuse" if match else "run"
    print(f"{action}: {path}\n  {shlex.join(map(str, command))}", flush=True)
    if args.dry_run:
        return
    if match:
        return
    if args.stage == "evaluate" and identity["stage"] != "evaluation":
        raise RuntimeError(f"Missing or stale training artifact: {path}")
    quarantine(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    log.parent.mkdir(parents=True, exist_ok=True)
    if log.exists():
        old_logs = log.parent / "stale_logs"
        old_logs.mkdir(exist_ok=True)
        log.rename(
            old_logs
            / (datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%f_") + log.name)
        )
    write_json(Path(str(path) + ".run.json"), dict(identity=identity,
               command=list(map(str, command)), git_revision=args.git_revision,
               started_utc=datetime.now(timezone.utc).isoformat()))
    with log.open("w") as handle:
        subprocess.run(
            list(map(str, command)),
            cwd=ROOT,
            stdout=handle,
            stderr=subprocess.STDOUT,
            check=True,
        )
    if not path.exists():
        raise RuntimeError(f"Stage did not produce {path}; see {log}")
    if enrich:
        record = json.loads(path.read_text())
        record.update(enrich)
        write_json(path, record)
    metadata = dict(
        identity=identity,
        sha256=digest(path),
        command=list(map(str, command)),
        git_revision=args.git_revision,
        created_utc=datetime.now(timezone.utc).isoformat(),
    )
    # The quality record is part of the artifact's integrity guarantee.
    quality = (
        Path(str(path).replace(".pt", ".quality.json"))
        if identity["stage"] == "predictor"
        else Path(str(path) + ".quality.json")
    )
    if quality.is_file():
        metadata["quality"] = json.loads(quality.read_text())
    write_json(sidecar(path), metadata)


def aggregate(records, dataset, method, args):
    """Combine only verified current fold results; partial runs remain partial."""
    folds = [r["folds"][0] for r in records]
    metrics = {}
    pooled = {}
    pooled_se = {}
    total = sum(f["n_samples"] for f in folds)
    for key in folds[0]["metrics"] if folds else []:
        values = [f["metrics"][key] for f in folds]
        mean = sum(values) / len(values)
        se = (
            (sum((v - mean) ** 2 for v in values) / (len(values) - 1) / len(values))
            ** 0.5
            if len(values) > 1
            else None
        )
        metrics[key] = dict(mean=mean, standard_error=se)
        pm = sum(f["metrics"][key] * f["n_samples"] for f in folds) / total
        pooled[key] = pm
        # Reconstruct the historical ddof=0 pooled variance from sufficient statistics.
        variance = (
            sum(
                f["n_samples"]
                * (
                    r["pooled"]["historical_standard_error"][key] ** 2 * f["n_samples"]
                    + (f["metrics"][key] - pm) ** 2
                )
                for f, r in zip(folds, records)
            )
            / total
        )
        pooled_se[key] = (variance / total) ** 0.5
    complete = {f["split"] for f in folds} == set(range(1, 6)) and all(
        r["completion_status"] == "complete_fold" for r in records
    )
    return dict(
        schema_version=2,
        dataset=dataset,
        method=method,
        base_seed=args.seed,
        protocol=args.protocol,
        completion_status="complete" if complete else "partial",
        provenance_status="unresolved_multivariate_recipe"
        if method == "cortx"
        else ("repaired" if args.protocol == "repaired-v1" else "diagnostic_control"),
        folds=folds,
        cross_validation=dict(n_folds=len(folds), metrics=metrics),
        pooled=dict(
            n_samples=total, metrics=pooled, historical_standard_error=pooled_se
        ),
        predictor_quality=[r["predictor_quality"] for r in records],
        provenance=[r["provenance"] for r in records],
        training_provenance=[r["training_provenance"] for r in records],
    )


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--datasets", default="all")
    p.add_argument("--methods", default=",".join(METHODS))
    p.add_argument("--folds", default="1,2,3,4,5")
    p.add_argument("--stage", choices=("all", "train", "evaluate"), default="all")
    p.add_argument("--protocol", choices=PROTOCOLS, default="repaired-v1")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--predictor-attempts", type=int, choices=range(1, 4), default=3)
    p.add_argument("--predictor-min-f1", type=float, default=0.95)
    p.add_argument("--winit-epochs", type=int, default=1000)
    p.add_argument("--results-dir", type=Path, default=ROOT / "results/table2_repair")
    p.add_argument(
        "--models-dir", type=Path, default=ROOT / "results/table2_repair/models"
    )
    p.add_argument(
        "--data-root",
        type=Path,
        default=Path(os.environ.get("TIMEX_DATA_ROOT", ROOT / "dataset")),
    )
    p.add_argument("--dry-run", action="store_true")
    p.add_argument(
        "--max-samples",
        type=int,
        default=int(os.environ["TIMEX_MAX_SAMPLES"])
        if "TIMEX_MAX_SAMPLES" in os.environ
        else None,
    )
    args = p.parse_args()
    args.datasets = (
        list(DATASETS) if args.datasets == "all" else args.datasets.split(",")
    )
    args.methods = args.methods.split(",")
    try:
        args.folds = sorted(set(map(int, args.folds.split(","))))
    except ValueError:
        p.error("folds must be comma-separated integers")
    if not args.folds or not set(args.folds) <= set(range(1, 6)):
        p.error("folds must be in 1..5")
    if not set(args.datasets) <= DATASETS.keys() or not set(args.methods) <= set(
        METHODS
    ):
        p.error("unknown dataset or method")
    if (
        args.seed < 0
        or not 0.95 <= args.predictor_min_f1 <= 1
        or args.winit_epochs < 1
        or (args.max_samples is not None and args.max_samples < 1)
    ):
        p.error("invalid seed, quality threshold, epoch budget or sample limit")
    for name in ("models_dir", "results_dir", "data_root"):
        setattr(args, name, getattr(args, name).resolve())
    args.git_revision = (
        subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True
        ).stdout.strip()
        or "source-snapshot"
    )
    return args


def main(args):
    results_root = args.results_dir / args.protocol / f"seed_{args.seed}"
    models_root = args.models_dir / args.protocol / f"seed_{args.seed}"
    failures = []
    if not args.dry_run:
        write_json(
            results_root / "run_configuration.json",
            {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
        )
    for dataset in args.datasets:
        directory, experiment, epochs = DATASETS[dataset]
        by_method = {method: [] for method in METHODS}
        sources = {
            stage: source_identity(stage, experiment)
            for stage in ("predictor", "ours", "sgt", "cortx", "winit", "evaluation")
        }
        # Inspect all folds on resume, but only launch explicitly selected folds.
        for fold in range(1, 6):
            selected = fold in args.folds
            models = models_root / dataset / f"fold_{fold}"
            logs = results_root / dataset / f"fold_{fold}"
            data = args.data_root / directory / f"split={fold}.pt"
            if not data.is_file() and not args.dry_run:
                if selected:
                    failures.append(f"{dataset} fold {fold}: missing {data}")
                continue
            base = dict(
                schema_version=1,
                dataset=dataset,
                fold=fold,
                seed=args.seed + fold - 1,
                data_sha256=digest(data) if data.is_file() else "missing-data",
            )

            def identity(stage, config, dependencies=None):
                return dict(
                    base,
                    stage=stage,
                    implementation=sources[stage],
                    config=config,
                    upstream=dependencies or {},
                )

            def upstream(path):
                return digest(path) if path.exists() else "pending"

            def command(script, *options):
                return [sys.executable, ROOT / script, *options]

            predictor = models / f"transformer_split={fold}.pt"
            predictor_config = dict(
                epochs=epochs,
                max_attempts=args.predictor_attempts,
                minimum_macro_f1=args.predictor_min_f1,
            )
            pi = identity("predictor", predictor_config)
            try:
                if selected:
                    run_stage(
                        predictor,
                        pi,
                        command(
                            f"experiments/{experiment}/train_transformer.py",
                            "--seed",
                            args.seed,
                            "--split-no",
                            fold,
                            "--models-path",
                            models,
                            "--data-path",
                            data.parent,
                            "--max-attempts",
                            args.predictor_attempts,
                            "--min-val-f1",
                            args.predictor_min_f1,
                        ),
                        logs / "predictor.log",
                        args,
                    )
                if args.dry_run:
                    quality = dict(qualified=True, validation_macro_f1="pending")
                else:
                    if not reusable(predictor, pi):
                        if selected:
                            raise RuntimeError("Predictor metadata mismatch")
                        continue
                    quality = json.loads(sidecar(predictor).read_text()).get(
                        "quality", {}
                    )
                    if (
                        not quality.get("qualified")
                        or quality.get("validation_macro_f1", 0) < args.predictor_min_f1
                    ):
                        raise RuntimeError("Predictor did not pass validation gate")
            except (
                OSError,
                ValueError,
                RuntimeError,
                subprocess.CalledProcessError,
            ) as error:
                if selected:
                    failures.append(f"{dataset} fold {fold}: {error}")
                if selected and not args.dry_run:
                    write_json(
                        logs / "status.json",
                        dict(status="predictor_failed", error=str(error)),
                    )
                continue
            for method in METHODS:
                selected_method = selected and method in args.methods
                try:
                    checkpoint = predictor
                    dep = {"predictor_sha256": upstream(predictor)}
                    extra = []
                    if method == "ours":
                        checkpoint = models / f"bc_full_split={fold}.pt"
                        cv = (
                            "legacy"
                            if args.protocol == "legacy-control"
                            else "temporal-l1-v1"
                        )
                        tv = (
                            "legacy"
                            if args.protocol
                            in {"legacy-control", "connectivity-control"}
                            else "repaired-v1"
                        )
                        ci = identity(
                            "ours",
                            dict(
                                connectivity_version=cv,
                                training_version=tv,
                                epochs=100,
                                gsat_r=0.5,
                                gsat=1,
                                connect=2,
                                beta_exp=2,
                                beta_sim=1,
                                lam_label=1,
                            ),
                            dep,
                        )
                        train_cmd = command(
                            f"experiments/{experiment}/bc_model_ptype.py",
                            "--seed",
                            args.seed,
                            "--split-no",
                            fold,
                            "--models-path",
                            models,
                            "--data-path",
                            data.parent,
                            "--connectivity-version",
                            cv,
                            "--training-version",
                            tv,
                        )
                    elif method in {"sgt+grad", "cortx"}:
                        stage = "sgt" if method == "sgt+grad" else "cortx"
                        checkpoint = models / f"{stage}_split={fold}.pt"
                        ci = identity(
                            stage,
                            dict(
                                protocol=args.protocol
                                if stage == "sgt"
                                else "released-reconstruction-v1",
                                epochs=(
                                    10 if args.protocol == "sgt-control" else epochs
                                )
                                if stage == "sgt"
                                else [100, 50],
                            ),
                            {} if stage == "sgt" else dep,
                        )
                        train_cmd = command(
                            "experiments/other_baselines/train_synth_baselines.py",
                            "--dataset",
                            dataset,
                            "--method",
                            stage,
                            "--seed",
                            args.seed,
                            "--split-no",
                            fold,
                            "--models-path",
                            models,
                            "--data-root",
                            args.data_root,
                            "--force",
                        )
                        if stage == "sgt" and args.protocol == "sgt-control":
                            train_cmd += ["--sgt-control"]
                            extra = ["--sgt-masked-input-control"]
                    elif method == "winit":
                        generator = models / f"winit_split={fold}"
                        gi = identity(
                            "winit",
                            dict(
                                epochs=args.winit_epochs,
                                window_size=10,
                                conditional=False,
                                batch_size=256,
                            ),
                        )
                        if selected_method:
                            run_stage(
                                generator,
                                gi,
                                command(
                                    "experiments/evaluation/winit_wrapper.py",
                                    "--dataset",
                                    experiment,
                                    "--models_path",
                                    models,
                                    "--data_path",
                                    args.data_root,
                                    "--split-no",
                                    fold,
                                    "--epochs",
                                    args.winit_epochs,
                                    "--seed",
                                    args.seed,
                                ),
                                logs / "winit_train.log",
                                args,
                            )
                        if not args.dry_run and not reusable(generator, gi):
                            raise RuntimeError("Missing or stale WinIT generator")
                        dep["generator_sha256"] = upstream(generator)
                    if method in {"ours", "sgt+grad", "cortx"}:
                        if selected_method:
                            run_stage(
                                checkpoint,
                                ci,
                                train_cmd,
                                logs / f"{method}_train.log",
                                args,
                            )
                        if not args.dry_run and not reusable(checkpoint, ci):
                            raise RuntimeError(f"Missing or stale {method} checkpoint")
                    dep["checkpoint_sha256"] = upstream(checkpoint)
                    ei = identity(
                        "evaluation",
                        dict(
                            method=method,
                            protocol=args.protocol,
                            max_samples=args.max_samples,
                        ),
                        dep,
                    )
                    output = logs / f"{method}_results.json"
                    complete = (
                        args.max_samples is None
                        and args.winit_epochs == 1000
                        and args.protocol == "repaired-v1"
                    )
                    training_provenance = (
                        {}
                        if args.dry_run
                        else json.loads(sidecar(checkpoint).read_text())
                    )
                    enrich = dict(
                        training_provenance=training_provenance,
                        protocol=args.protocol,
                        completion_status="complete_fold" if complete else "diagnostic",
                        predictor_quality=quality,
                        provenance=ei,
                    )
                    if selected_method and args.stage != "train":
                        eval_cmd = command(
                            "experiments/evaluation/saliency_exp_synth.py",
                            "--dataset",
                            experiment,
                            "--exp_method",
                            method,
                            "--split_no",
                            fold,
                            "--model_path",
                            checkpoint,
                            "--seed",
                            args.seed,
                            "--data_path",
                            args.data_root,
                            "--results-json",
                            output,
                            "--no-progress",
                            *extra,
                        )
                        if args.max_samples is not None:
                            eval_cmd += ["--max-samples", args.max_samples]
                        run_stage(
                            output,
                            ei,
                            eval_cmd,
                            logs / f"{method}_evaluation.log",
                            args,
                            enrich=enrich,
                        )
                    if not args.dry_run and reusable(output, ei):
                        by_method[method].append(json.loads(output.read_text()))
                except (
                    OSError,
                    ValueError,
                    RuntimeError,
                    subprocess.CalledProcessError,
                ) as error:
                    if selected_method:
                        failures.append(f"{dataset} fold {fold} {method}: {error}")
                        write_json(
                            logs / f"{method}_status.json",
                            dict(status="failed", error=str(error)),
                        )
        if not args.dry_run:
            for method, records in by_method.items():
                write_json(
                    results_root / f"{dataset}_{method.replace('+', '_')}_results.json",
                    aggregate(records, dataset, method, args),
                )
    if not args.dry_run:
        write_json(
            results_root / "run_status.json",
            dict(
                status="failed" if failures else "finished_requested_stages",
                failures=failures,
                folds=args.folds,
                datasets=args.datasets,
                methods=args.methods,
                stage=args.stage,
            ),
        )
        subprocess.run(
            [
                sys.executable,
                ROOT / "experiments/evaluation/summarize_synth.py",
                "--table",
                "2",
                results_root,
            ],
            check=True,
        )
    if failures:
        print("\n".join(failures), file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main(parse_args()))
