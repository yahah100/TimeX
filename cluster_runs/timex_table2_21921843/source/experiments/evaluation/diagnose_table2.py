"""Read-only CPU audit of archived Table 2 predictor and CoRTX failures."""

import argparse
import json
import os
from pathlib import Path

os.environ["CUDA_VISIBLE_DEVICES"] = ""

import torch

from experiments.table2_workflow import digest
from txai.baselines.synth_baselines import (
    cortx_mask,
    make_cortx_decoder,
    make_transformer,
)
from txai.utils.data import process_Synth
from txai.utils.predictors import eval_mvts_transformer
from txai.utils.reproducibility import seed_everything


def main(args):
    torch.set_num_threads(1)
    seed_everything(42)
    report = {
        "scope": "historical CPU diagnosis; not a repaired reproduction",
        "predictors": [],
    }
    for fold in (3, 4):
        data_path = args.data_root / "SeqCombMV" / f"split={fold}.pt"
        data = process_Synth(fold, device="cpu", base_path=data_path.parent)
        checkpoint = args.archive / "models/seqcomb_mv" / f"transformer_split={fold}.pt"
        shape = data["val"][0].shape
        model = make_transformer("seqcomb_mv", shape[-1], shape[0])
        model.load_state_dict(torch.load(checkpoint, map_location="cpu"))
        report["predictors"].append(
            dict(
                fold=fold,
                data_sha256=digest(data_path),
                checkpoint_sha256=digest(checkpoint),
                validation_macro_f1=float(eval_mvts_transformer(data["val"], model)),
            )
        )
    data_path = args.data_root / "LowVarDetect/split=1.pt"
    data = process_Synth(1, device="cpu", base_path=data_path.parent)
    x = data["train_loader"].X
    times = data["train_loader"].times
    checkpoint = args.archive / "models/lowvardetect/cortx_split=1.pt"
    state = torch.load(checkpoint, map_location="cpu")
    model = make_transformer("lowvardetect", x.shape[-1], x.shape[0]).eval()
    decoder = make_cortx_decoder("lowvardetect", x.shape[-1], x.shape[0]).eval()
    model.load_state_dict(state["encoder"])
    decoder.load_state_dict(state["decoder"])
    with torch.no_grad():
        sample = x[:, :128]
        sample_times = times[:, :128]
        z = model.embed(sample, sample_times, aggregate=False)
        mask = cortx_mask(decoder, z, sample, sample_times)
    report["cortx_lowvar_fold1"] = dict(
        data_sha256=digest(data_path),
        checkpoint_sha256=digest(checkpoint),
        train_shape=list(x.shape),
        outside_unit_interval=float(((x < 0) | (x > 1)).float().mean()),
        bounded_reconstruction_mse_lower_bound=float(
            (x - x.clamp(0, 1)).square().mean()
        ),
        diagnostic_samples=sample.shape[1],
        mask_mean=float(mask.mean()),
        mask_std=float(mask.std(unbiased=False)),
        saturated_fraction=float(((mask < 0.01) | (mask > 0.99)).float().mean()),
        reconstruction_mse=float((mask - sample).square().mean()),
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=Path("dataset"))
    parser.add_argument(
        "--archive", type=Path, default=Path("timex_table2_21846132/seed_42")
    )
    parser.add_argument(
        "--output", type=Path, default=Path("results/table2_cpu_diagnosis.json")
    )
    main(parser.parse_args())
