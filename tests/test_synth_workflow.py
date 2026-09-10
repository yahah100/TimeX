"""Focused CPU correctness checks; these do not measure numerical reproduction."""

import ast
from contextlib import redirect_stdout
from copy import deepcopy
import importlib
import io
import json
import os
from pathlib import Path
import shutil
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

os.environ["CUDA_VISIBLE_DEVICES"] = ""

import torch
import torch.nn.functional as F

from experiments import synth_workflow as workflow
from experiments.evaluation.summarize_synth import comparison_rows, PUBLISHED_BY_TABLE
from txai.baselines.synth_baselines import (
    absolute_input_gradients,
    mask_bottom_features,
    sgt_objective,
    symmetric_infonce,
    cortx_mask,
    make_cortx_decoder,
)
from txai.models.bc_model import TimeXModel, transformer_default_args
from txai.utils.predictors.loss import ConnectLoss, Poly1CrossEntropyLoss
from txai.utils.reproducibility import seed_everything

ROOT = Path(__file__).resolve().parents[1]
torch.set_num_threads(1)


def small_model(d=4, reference_eval=True):
    args = deepcopy(transformer_default_args)
    args.update(nlayers=1, trans_dim_feedforward=16, trans_dropout=0.5)
    model = TimeXModel(
        d_inp=d,
        max_len=6,
        n_classes=4,
        n_prototypes=3,
        gsat_r=0.5,
        transformer_args=args,
        masktoken_stats=(torch.zeros(6, d), torch.ones(6, d)),
        reference_eval=reference_eval,
        loss_weight_dict={"gsat": 1.0, "connect": 2.0},
    )
    model.encoder_main.requires_grad_(False)
    return model


class NumericalChecks(unittest.TestCase):
    def setUp(self):
        seed_everything(42)

    def test_native_connectivity_formula_and_layout(self):
        for d in (1, 4):
            model = small_model(d)
            native = torch.rand(3, 6, d, requires_grad=True)
            differences = native[:, 1:] - native[:, :-1]
            expected = differences.norm(p=2) / differences.numel()
            actual = model.loss_components({"mask_logits": native})["connect"]
            torch.testing.assert_close(actual, expected)
            torch.testing.assert_close(
                torch.autograd.grad(actual, native, retain_graph=True)[0],
                torch.autograd.grad(expected, native)[0],
            )
        x = torch.ones(3, 1, 4, requires_grad=True)
        self.assertEqual(ConnectLoss()(x).item(), 0)

    def test_frozen_reference_and_checkpoint_roundtrip(self):
        model = small_model()
        model.train()
        self.assertFalse(model.encoder_main.training)
        self.assertTrue(model.encoder_t.training)
        x, times = torch.randn(6, 3, 4), torch.arange(1, 7.0).view(6, 1).repeat(1, 3)
        with torch.no_grad():
            a = model.encoder_main(x, times)
            b = model.encoder_main(x, times)
        torch.testing.assert_close(a, b, rtol=0, atol=0)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "model.pt"
            model.save_state(path)
            state, config = torch.load(path, map_location="cpu")
            restored = TimeXModel(**config)
            restored.load_state_dict(state)
            self.assertTrue(restored.reference_eval)
            self.assertEqual(restored.loss_weight_dict["connect"], 2)
        original = small_model(reference_eval=False)
        original.train()
        self.assertTrue(original.encoder_main.training)

    def test_actual_trainer_clips_after_backward(self):
        trainer = importlib.import_module("txai.trainers.train_mv6_consistency")
        model = small_model()
        x = torch.randn(3, 6, 4)
        times = torch.arange(1, 7.0).repeat(3, 1)
        y = torch.tensor([0, 1, 2])
        data = [(x, times, y, torch.arange(3))]
        val = (x.transpose(0, 1), times.T, y)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
        events = []
        original_clip = torch.nn.utils.clip_grad_norm_
        original_step = optimizer.step

        def clip(parameters, *args, **kwargs):
            params = list(parameters)
            self.assertTrue(
                any(p.grad is not None and p.grad.abs().sum() > 0 for p in params)
            )
            events.append("clip")
            return original_clip(params, *args, **kwargs)

        def step(*args, **kwargs):
            self.assertEqual(events[-1], "clip")
            norm = torch.stack(
                [p.grad.norm() for p in model.parameters() if p.grad is not None]
            ).norm()
            self.assertLessEqual(float(norm), 1.00001)
            events.append("step")
            return original_step(*args, **kwargs)

        with tempfile.TemporaryDirectory() as tmp, patch(
            "torch.nn.utils.clip_grad_norm_", side_effect=clip
        ), patch.object(optimizer, "step", side_effect=step):
            save = Path(tmp) / "timex.pt"
            trainer.train_mv6_consistency(
                model,
                optimizer,
                data,
                val,
                1,
                clf_criterion=Poly1CrossEntropyLoss(4, reduction="mean"),
                sim_criterion=None,
                beta_exp=2,
                beta_sim=1,
                train_tuple=val,
                save_path=save,
                selection_criterion=lambda out, val: 1.0,
            )
            self.assertEqual(events, ["clip", "step"])
            self.assertTrue(
                all(p.grad is None for p in model.encoder_main.parameters())
            )
            history = json.loads(Path(str(save) + ".epochs.json").read_text())
            self.assertTrue(np.isfinite(history[0]["temporal_variation"]))
            self.assertEqual(
                torch.load(save)[1]["training_loss_weights"]["beta_exp"], 2
            )

    def test_infonce_loss_and_both_view_gradients_match_author_formula(self):
        # Execute the actual released class and helper AST without its missing
        # packaging-only `losses.Loss` dependency; no formula is reimplemented.
        tree = ast.parse((ROOT / "tests/fixtures/infonce_reference.py").read_text())
        nodes = [
            n
            for n in tree.body
            if isinstance(n, (ast.FunctionDef, ast.ClassDef))
            and n.name in {"_similarity", "InfoNCE"}
        ]
        namespace = {"torch": torch, "F": F, "Loss": object}
        exec(
            compile(
                ast.Module(body=nodes, type_ignores=[]), "released_infonce", "exec"
            ),
            namespace,
        )
        released = namespace["InfoNCE"](tau=0.7)
        for n in (1, 4, 16):
            z1 = torch.randn(n, 7, dtype=torch.double, requires_grad=True)
            z2 = torch.randn(n, 7, dtype=torch.double, requires_grad=True)
            pos = torch.eye(n, dtype=torch.double)
            neg = 1 - pos
            expected = (
                released.compute(z1, z2, pos, neg) + released.compute(z2, z1, pos, neg)
            ) / 2
            actual = symmetric_infonce(z1, z2)
            torch.testing.assert_close(actual, expected, atol=1e-12, rtol=1e-12)
            for a, b in zip(
                torch.autograd.grad(actual, (z1, z2), retain_graph=True),
                torch.autograd.grad(expected, (z1, z2)),
            ):
                torch.testing.assert_close(a, b, atol=1e-12, rtol=1e-12)

    def test_sgt_masking_gradients_and_released_loss(self):
        x = torch.arange(80.0).reshape(10, 2, 4).transpose(0, 1)
        scores = x.clone()
        masked = mask_bottom_features(x, scores)
        self.assertEqual((masked != x).flatten(1).sum(1).tolist(), [36, 36])
        self.assertTrue(torch.all(masked.flatten(1)[:, -4:] == x.flatten(1)[:, -4:]))
        self.assertTrue(
            torch.all(masked.flatten(1).min(1).values >= x.flatten(1).min(1).values)
        )
        self.assertTrue(
            torch.all(masked.flatten(1).max(1).values <= x.flatten(1).max(1).values)
        )

        class LinearClassifier(torch.nn.Module):
            def forward(self, x, times, captum_input=True):
                v = x.flatten(1).sum(1)
                return torch.stack([v, 2 * v], dim=1)

        grads = absolute_input_gradients(
            LinearClassifier(), x, None, torch.tensor([0, 1])
        )
        torch.testing.assert_close(grads[0], torch.ones_like(x[0]))
        torch.testing.assert_close(grads[1], 2 * torch.ones_like(x[1]))
        a = torch.randn(3, 4, requires_grad=True)
        b = torch.randn(3, 4, requires_grad=True)
        y = torch.tensor([0, 1, 2])
        clf, kl = sgt_objective(a, b, y)
        ce = F.cross_entropy(a, y)
        expected_poly1 = ce + (1 - a.softmax(1)[torch.arange(3), y]).mean()
        torch.testing.assert_close(clf, expected_poly1)
        torch.testing.assert_close(
            kl,
            torch.nn.KLDivLoss(reduction="batchmean")(b.log_softmax(1), a.softmax(1)),
        )
        gradients = torch.autograd.grad(kl, (a, b))
        self.assertTrue(
            all(g.abs().sum() > 0 and torch.isfinite(g).all() for g in gradients)
        )

    def test_predictor_retries_stop_at_first_qualifying_attempt(self):
        for dataset in ("seqcomb_mv", "lowvardetect"):
            module = importlib.import_module(f"experiments.{dataset}.train_transformer")
            data = dict(
                train_loader=torch.utils.data.TensorDataset(
                    torch.zeros(4, 6, 4), torch.ones(4, 6), torch.arange(4)
                ),
                val=(torch.zeros(6, 4, 4), torch.ones(6, 4), torch.arange(4)),
            )
            data["test"] = data["val"]
            scores = iter(([0.4, 0.8], [0.96, 0.9]))
            seeds = []

            def train(model, *args, **kwargs):
                return model, [], next(scores)

            with tempfile.TemporaryDirectory() as tmp, patch.object(
                module, "process_Synth", return_value=data
            ), patch.object(module, "train", side_effect=train), patch.object(
                module, "seed_everything", side_effect=seeds.append
            ), patch(
                "torch.cuda.is_available", return_value=False
            ):
                args = SimpleNamespace(
                    models_path=Path(tmp),
                    split_no=3,
                    seed=42,
                    data_path=Path(tmp),
                    epochs=2,
                    max_attempts=3,
                    min_val_f1=0.95,
                    original_predictor=False,
                )
                module.main(args)
                self.assertEqual(seeds, [44, 1044])
                quality = json.loads(
                    (Path(tmp) / "transformer_split=3.quality.json").read_text()
                )
                self.assertTrue(quality["qualified"])
                self.assertEqual(quality["attempts"][1]["selected_epoch"], 1)
                self.assertEqual(len(list(Path(tmp).glob("*_attempt=*.pt"))), 2)
                self.assertTrue((Path(tmp) / "transformer_split=3.pt").exists())

    def test_exhausted_predictor_retries_fail_explicitly(self):
        module = importlib.import_module("experiments.seqcomb_mv.train_transformer")
        data = dict(
            train_loader=torch.utils.data.TensorDataset(
                torch.zeros(4, 6, 4), torch.ones(4, 6), torch.arange(4)
            ),
            val=(torch.zeros(6, 4, 4), torch.ones(6, 4), torch.arange(4)),
        )
        data["test"] = data["val"]

        def train(model, *args, **kwargs):
            return model, [], [0.5]

        with tempfile.TemporaryDirectory() as tmp, patch.object(
            module, "process_Synth", return_value=data
        ), patch.object(module, "train", side_effect=train), patch(
            "torch.cuda.is_available", return_value=False
        ):
            args = SimpleNamespace(
                models_path=Path(tmp),
                split_no=1,
                seed=42,
                data_path=Path(tmp),
                epochs=1,
                max_attempts=3,
                min_val_f1=0.95,
                original_predictor=False,
            )
            with self.assertRaisesRegex(RuntimeError, "after 3 attempts"):
                module.main(args)
            self.assertFalse((Path(tmp) / "transformer_split=1.pt").exists())
            self.assertEqual(len(list(Path(tmp).glob("*_attempt=*.pt"))), 3)

    def test_original_predictor_records_low_quality_without_retry(self):
        module = importlib.import_module("experiments.seqcomb_mv.train_transformer")
        data = dict(
            train_loader=torch.utils.data.TensorDataset(
                torch.zeros(4, 6, 4), torch.ones(4, 6), torch.arange(4)
            ),
            val=(torch.zeros(6, 4, 4), torch.ones(6, 4), torch.arange(4)),
        )
        data["test"] = data["val"]
        with tempfile.TemporaryDirectory() as tmp, patch.object(
            module, "process_Synth", return_value=data
        ), patch.object(
            module, "train", side_effect=lambda model, *a, **kw: (model, [], [0.7])
        ) as train, patch(
            "torch.cuda.is_available", return_value=False
        ):
            module.main(
                SimpleNamespace(
                    models_path=Path(tmp),
                    split_no=3,
                    seed=42,
                    data_path=Path(tmp),
                    epochs=1,
                    max_attempts=3,
                    min_val_f1=0.95,
                    original_predictor=True,
                )
            )
            quality = json.loads(
                (Path(tmp) / "transformer_split=3.quality.json").read_text()
            )
            self.assertFalse(quality["qualified"])
            self.assertEqual(train.call_count, 1)
            self.assertTrue((Path(tmp) / "transformer_split=3.pt").exists())

    def test_original_sgt_matches_ce_detached_kl_update(self):
        from experiments.other_baselines.train_synth_baselines import train_sgt
        from txai.baselines.synth_baselines import make_transformer

        x = torch.randn(4, 6, 1)
        times = torch.arange(1, 7.0).repeat(4, 1)
        y = torch.arange(4)
        seed_everything(42)
        reference = make_transformer("freqshape", 1, 6)
        optimizer = torch.optim.AdamW(reference.parameters(), lr=1e-3, weight_decay=0.1)
        reference.eval()
        scores = absolute_input_gradients(reference, x, times, y)
        masked = mask_bottom_features(x, scores)
        reference.train()
        logits = reference(x, times, captum_input=True)
        masked_logits = reference(masked, times, captum_input=True)
        loss = F.cross_entropy(logits, y) + F.kl_div(
            masked_logits.log_softmax(1),
            logits.detach().softmax(1),
            reduction="batchmean",
        )
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        with tempfile.TemporaryDirectory() as tmp:
            seed_everything(42)
            output = Path(tmp) / "sgt.pt"
            train_sgt("freqshape", [(x, times, y)], (1, 6), output, 1, "cpu", None)
            actual = torch.load(output)
            for name, value in reference.state_dict().items():
                torch.testing.assert_close(actual[name], value, rtol=0, atol=0)

    def test_sgt_and_cortx_reduced_cpu_training(self):
        from experiments.other_baselines.train_synth_baselines import (
            train_sgt,
            train_cortx,
        )
        from txai.baselines.synth_baselines import make_transformer

        x, times, y = (
            torch.randn(4, 6, 4),
            torch.arange(1, 7.0).repeat(4, 1),
            torch.arange(4),
        )
        val = (x.transpose(0, 1), times.T, y)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)
            train_sgt(
                "lowvardetect", [(x, times, y)], (4, 6), path / "sgt.pt", 1, "cpu", val
            )
            self.assertTrue((path / "sgt.pt.quality.json").exists())
            torch.save(
                make_transformer("lowvardetect", 4, 6).state_dict(),
                path / "predictor.pt",
            )
            train_cortx(
                "lowvardetect",
                [(x, times, y)],
                (4, 6),
                path / "predictor.pt",
                path / "cortx.pt",
                1,
                1,
                "cpu",
            )
            evidence = json.loads((path / "cortx.pt.epochs.json").read_text())[0]
            self.assertTrue(evidence["encoder_training"])
            self.assertGreater(evidence["outside_unit_interval"], 0)

    def test_cortx_layout_and_bounded_decoder(self):
        for d in (1, 4):
            decoder = make_cortx_decoder("seqcomb_mv", d, 6).eval()
            x, times = torch.randn(6, 3, d), torch.arange(1, 7.0).view(6, 1).repeat(
                1, 3
            )
            mask = cortx_mask(decoder, torch.randn(6, 3, d + 16), x, times)
            self.assertEqual(mask.shape, x.shape)
            self.assertTrue(torch.all((mask >= 0) & (mask <= 1)))


class RunnerChecks(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.calls = []
        for dataset in workflow.DATASETS.values():
            for fold in range(1, 6):
                path = self.root / "data" / dataset[0] / f"split={fold}.pt"
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(f"data {fold}")
        self.args = self.arguments()

    def tearDown(self):
        self.tmp.cleanup()

    def arguments(self, *extra):
        argv = [
            "runner",
            "--data-root",
            str(self.root / "data"),
            "--results-dir",
            str(self.root / "results"),
            "--models-dir",
            str(self.root / "models"),
            "--table",
            "2",
            "--datasets",
            "seqcomb_mv",
            "--methods",
            "ours,winit",
            "--folds",
            "1",
            *extra,
        ]
        with patch.object(sys, "argv", argv):
            return workflow.parse_args()

    def fake_run(self, command, **kwargs):
        command = list(map(str, command))
        name = Path(command[1]).name
        if name == "summarize_synth.py":
            return SimpleNamespace(returncode=0)
        self.calls.append(command)

        def option(key):
            return command[command.index(key) + 1]

        if name == "saliency_exp_synth.py":
            path = Path(option("--results-json"))
            fold = int(option("--split_no"))
            workflow.write_json(
                path,
                dict(
                    dataset="seqcomb_mv",
                    method=option("--exp_method"),
                    folds=[
                        dict(
                            split=fold,
                            n_samples=2,
                            metrics=dict(auprc=0.5, aup=0.5, aur=0.5),
                        )
                    ],
                    pooled=dict(
                        historical_standard_error=dict(auprc=0.01, aup=0.01, aur=0.01)
                    ),
                ),
            )
        elif name == "winit_wrapper.py":
            path = Path(option("--models_path")) / f'winit_split={option("--split-no")}'
            path.mkdir(parents=True)
            (path / "generator.pt").write_text("generator")
        else:
            fold = option("--split-no")
            if name == "train_transformer.py":
                prefix = (
                    "Scomb_transformer"
                    if Path(command[1]).parent.name in {"freqshape", "scs_better"}
                    else "transformer"
                )
            elif name == "train_synth_baselines.py":
                prefix = option("--method")
            else:
                prefix = "bc_full"
            path = Path(option("--models-path")) / f"{prefix}_split={fold}.pt"
            path.write_text(f"{prefix} {len(self.calls)}")
            if prefix.endswith("transformer"):
                workflow.write_json(
                    path.with_suffix(".quality.json"),
                    dict(qualified=True, validation_macro_f1=0.99),
                )
        return SimpleNamespace(returncode=0)

    def run_workflow(self, args=None):
        with patch.object(
            workflow.subprocess, "run", side_effect=self.fake_run
        ), redirect_stdout(io.StringIO()):
            return workflow.main(args or self.args)

    def test_resume_integrity_dependencies_and_seed_isolation(self):
        self.assertEqual(self.run_workflow(), 0)
        self.assertEqual(len(self.calls), 6)
        self.assertEqual(self.run_workflow(), 0)
        self.assertEqual(len(self.calls), 6)
        predictor = next((self.root / "models").rglob("original/transformer*.pt"))
        predictor.write_text("externally replaced")
        self.assertEqual(self.run_workflow(), 0)
        # Only the original predictor, TimeX and its evaluation change.
        self.assertEqual(len(self.calls), 9)
        self.assertEqual(
            sum(Path(c[1]).name == "winit_wrapper.py" for c in self.calls), 1
        )
        self.assertTrue(list((predictor.parent / "stale").rglob("*.pt")))
        # Relocated archived metadata must remain reusable (paths not in identity).
        archived = self.root / "archive"
        shutil.copytree(self.root / "models", archived / "models")
        shutil.copytree(self.root / "results", archived / "results")
        self.args.models_dir, self.args.results_dir = (
            archived / "models",
            archived / "results",
        )
        self.assertEqual(self.run_workflow(), 0)
        self.assertEqual(len(self.calls), 9)
        self.assertEqual(self.run_workflow(self.arguments("--seed", "43")), 0)
        self.assertEqual(len(self.calls), 15)

    def test_partial_folds_and_diagnostics_never_complete(self):
        self.assertEqual(self.run_workflow(), 0)
        path = self.root / "results/selected-v1/seed_42/seqcomb_mv_ours_results.json"
        self.assertEqual(json.loads(path.read_text())["completion_status"], "partial")
        self.assertEqual(self.run_workflow(self.arguments("--folds", "2,3,4,5")), 0)
        self.assertEqual(json.loads(path.read_text())["completion_status"], "complete")
        self.assertEqual(
            self.run_workflow(
                self.arguments("--folds", "1,2,3,4,5", "--max-samples", "2")
            ),
            0,
        )
        self.assertEqual(json.loads(path.read_text())["completion_status"], "partial")

    def test_evaluate_rejects_stale_predictor(self):
        self.assertEqual(self.run_workflow(), 0)
        predictor = next((self.root / "models").rglob("original/transformer*.pt"))
        workflow.sidecar(predictor).unlink()
        self.assertEqual(self.run_workflow(self.arguments("--stage", "evaluate")), 1)
        self.assertEqual(len(self.calls), 6)

    def test_training_invalidates_unselected_method_summary(self):
        self.assertEqual(self.run_workflow(self.arguments("--folds", "1,2,3,4,5")), 0)
        summary = (
            self.root / "results/selected-v1/seed_42/seqcomb_mv_winit_results.json"
        )
        self.assertEqual(
            json.loads(summary.read_text())["completion_status"], "complete"
        )
        predictor = next(
            (self.root / "models").rglob("qualified/transformer_split=1.pt")
        )
        predictor.write_text("changed predictor")
        self.assertEqual(
            self.run_workflow(self.arguments("--stage", "train", "--methods", "ig")),
            0,
        )
        self.assertEqual(
            json.loads(summary.read_text())["cross_validation"]["n_folds"], 4
        )
        self.assertEqual(
            json.loads(summary.read_text())["completion_status"], "partial"
        )

    def test_failed_fold_does_not_block_independent_folds(self):
        args = self.arguments("--folds", "1,2")
        original = self.fake_run

        def fail_first(command, **kwargs):
            command = list(map(str, command))
            if (
                Path(command[1]).name == "train_transformer.py"
                and command[command.index("--split-no") + 1] == "1"
            ):
                raise workflow.subprocess.CalledProcessError(1, command)
            return original(command, **kwargs)

        with patch.object(
            workflow.subprocess, "run", side_effect=fail_first
        ), redirect_stdout(io.StringIO()):
            result = workflow.main(args)
        self.assertEqual(result, 1)
        records = json.loads(
            (
                self.root / "results/selected-v1/seed_42/seqcomb_mv_ours_results.json"
            ).read_text()
        )
        self.assertEqual([f["split"] for f in records["folds"]], [2])
        self.assertTrue(
            list((self.root / "models").rglob("transformer_split=1.pt.run.json"))
        )

    def test_selected_recipes_and_predictor_isolation(self):
        self.assertEqual(workflow.predictor_policy("seqcomb_mv", "ours"), "original")
        self.assertEqual(workflow.predictor_policy("seqcomb_mv", "cortx"), "original")
        self.assertEqual(workflow.predictor_policy("seqcomb_mv", "winit"), "qualified")
        self.assertEqual(workflow.predictor_policy("lowvar", "sgt+grad"), "independent")
        self.assertEqual(
            workflow.recipe_name("freqshape", "sgt+grad"), "ce-detached-kl-final"
        )
        self.assertEqual(workflow.recipe_name("lowvar", "ours"), "lowvar-timex")
        self.assertEqual(self.run_workflow(), 0)
        commands = [c for c in self.calls if Path(c[1]).name == "train_transformer.py"]
        self.assertEqual(sum("--original-predictor" in c for c in commands), 1)
        self.assertEqual(len(commands), 2)

    def test_table1_all_methods_resume(self):
        args = self.arguments(
            "--table",
            "1",
            "--datasets",
            "freqshape",
            "--methods",
            ",".join(workflow.METHODS),
        )
        self.assertEqual(self.run_workflow(args), 0)
        count = len(self.calls)
        self.assertEqual(self.run_workflow(args), 0)
        self.assertEqual(len(self.calls), count)
        summaries = list(
            (self.root / "results/selected-v1/seed_42").glob("freqshape*_results.json")
        )
        self.assertEqual(len(summaries), 6)
        self.assertTrue(
            all(
                json.loads(p.read_text())["cross_validation"]["n_folds"] == 1
                for p in summaries
            )
        )

    def test_sgt_runs_without_reference_predictor(self):
        self.assertEqual(self.run_workflow(self.arguments("--methods", "sgt+grad")), 0)
        self.assertFalse(
            any(Path(c[1]).name == "train_transformer.py" for c in self.calls)
        )
        self.assertEqual(len(self.calls), 2)

    def test_dry_run_has_no_mutations(self):
        self.assertEqual(self.run_workflow(self.arguments("--dry-run")), 0)
        self.assertFalse((self.root / "models").exists())
        self.assertFalse((self.root / "results").exists())
        self.assertEqual(self.calls, [])

    def test_summary_requires_all_three_metrics_and_provenance(self):
        folder = self.root / "summary"
        record = dict(
            dataset="seqcomb_mv",
            method="ours",
            protocol="selected-v1",
            completion_status="complete",
            provenance_status="repaired",
            provenance=[{}] * 5,
            predictor_quality=[dict(qualified=True, validation_macro_f1=0.99)] * 5,
            cross_validation=dict(
                n_folds=5,
                metrics={
                    k: dict(mean=v, standard_error=0.01)
                    for k, v in zip(
                        ("auprc", "aup", "aur"),
                        PUBLISHED_BY_TABLE[2]["seqcomb_mv"]["ours"],
                    )
                },
            ),
        )
        path = folder / "seqcomb_mv_ours_results.json"
        workflow.write_json(path, record)

        def status():
            return [
                r
                for r in comparison_rows(folder, 2)
                if r["dataset"] == "seqcomb_mv" and r["method"] == "ours"
            ][0]["row_status"]

        self.assertEqual(status(), "matched")
        record["cross_validation"]["metrics"]["aur"]["mean"] += 0.06
        workflow.write_json(path, record)
        self.assertEqual(status(), "outside tolerance")
        record["provenance_status"] = "unresolved_multivariate_recipe"
        workflow.write_json(path, record)
        self.assertEqual(status(), "unresolved provenance")


class ReportChecks(unittest.TestCase):
    def test_recorded_folds_and_report_regeneration(self):
        from experiments.evaluation.report_synth import render, validate

        data = json.loads((ROOT / "experiments/reproduction_results.json").read_text())
        validate(data)
        self.assertEqual(render(data), (ROOT / "results_reproduction.md").read_text())
        broken = deepcopy(data)
        broken["rows"][0]["folds"].pop()
        with self.assertRaises(ValueError):
            validate(broken)
        broken = deepcopy(data)
        broken["rows"][0]["seed"] = 43
        with self.assertRaises(ValueError):
            validate(broken)


if __name__ == "__main__":
    unittest.main(verbosity=2)
