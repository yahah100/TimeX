"""Train CoRTX or SGT + Grad on the synthetic Table 1 and Table 2 datasets."""

import argparse
import json
from pathlib import Path

import torch
import torch.nn.functional as F

from txai.baselines.synth_baselines import (
    absolute_input_gradients,
    cortx_mask,
    make_cortx_decoder,
    make_transformer,
    mask_bottom_features,
    symmetric_infonce,
    sgt_objective,
    checked_step,
)
from txai.utils.constants import DATA_ROOT, EXPERIMENTS_ROOT
from txai.utils.data import process_Synth
from txai.utils.reproducibility import seed_everything
from txai.utils.predictors import eval_mvts_transformer


# CLI name -> (Dataverse directory, experiment directory and config key, predictor file).
# The Table 1 experiments name their predictor checkpoints differently from Table 2's.
DATASETS = {
    "freqshape": ("FreqShape", "freqshape", "Scomb_transformer_split={}.pt"),
    "seqcomb_uv": ("SeqCombSingle", "scs_better", "Scomb_transformer_split={}.pt"),
    "seqcomb_mv": ("SeqCombMV", "seqcomb_mv", "transformer_split={}.pt"),
    "lowvar": ("LowVarDetect", "lowvardetect", "transformer_split={}.pt"),
}

# SGT trains its own classifier from scratch, so it gets the same optimizer settings and
# epoch budget as the reference predictor it stands in for.
SGT_TRAINING: dict[str, tuple[float, float, int]] = {
    # learning rate, weight decay, epochs
    "freqshape": (1e-3, 0.1, 100),
    "scs_better": (1e-3, 0.01, 200),
    "seqcomb_mv": (5e-4, 0.01, 1000),
    "lowvardetect": (1e-3, 0.01, 120),
}


def cpu_state_dict(module):
    return {key: value.detach().cpu() for key, value in module.state_dict().items()}


def batches(data):
    dataset = torch.utils.data.TensorDataset(
        data["train_loader"].X.transpose(0, 1),
        data["train_loader"].times.transpose(0, 1),
        data["train_loader"].y,
    )
    return torch.utils.data.DataLoader(dataset, batch_size=64, shuffle=True)


def train_cortx(
    dataset,
    loader,
    shape,
    predictor_path,
    output_path,
    epochs,
    decoder_epochs,
    device,
):
    reference = make_transformer(dataset, *shape).to(device)
    reference.load_state_dict(torch.load(predictor_path, map_location=device))
    encoder = make_transformer(dataset, *shape).to(device)
    encoder.load_state_dict(reference.state_dict())
    encoder.mlp.requires_grad_(False)
    reference.eval()
    optimizer = torch.optim.Adam(encoder.parameters(), lr=5e-3)
    for epoch in range(epochs):
        encoder.train()
        total = 0.0
        for x, times, _ in loader:
            augmented = x * (torch.rand_like(x) > 0.8)
            _, z1, _ = encoder(x, times, captum_input=True, get_agg_embed=True)
            _, z2, _ = encoder(augmented, times, captum_input=True, get_agg_embed=True)
            loss = symmetric_infonce(z1, z2, temperature=0.7)
            checked_step(loss, optimizer, encoder.parameters())
            total += loss.item()
        print(f"CoRTX encoder epoch {epoch + 1}/{epochs}: {total / len(loader):.6f}")

    # The author snapshot leaves the encoder in training mode under no_grad.
    # Preserve that behavior; changing dropout here would be a protocol change.
    encoder.train()
    history = []
    decoder = make_cortx_decoder(dataset, *shape).to(device)
    optimizer = torch.optim.Adam(decoder.parameters(), lr=5e-3)
    for epoch in range(decoder_epochs):
        decoder.train()
        total = 0.0
        counts = dict(values=0, outside_unit_interval=0, saturated=0)
        lower_bound_sum = 0.0
        mask_sum = mask_square_sum = 0.0
        for x, times, _ in loader:
            x_tb, times_tb = x.transpose(0, 1), times.transpose(0, 1)
            with torch.no_grad():
                z_seq = encoder.embed(x_tb, times_tb, aggregate=False)
            mask = cortx_mask(decoder, z_seq, x_tb, times_tb)
            loss = F.mse_loss(mask, x_tb)
            if mask.shape != x_tb.shape:
                raise ValueError(
                    f"CoRTX reconstruction layout mismatch: {mask.shape} vs {x_tb.shape}"
                )
            checked_step(loss, optimizer, decoder.parameters())
            counts["values"] += x_tb.numel()
            counts["outside_unit_interval"] += int(((x_tb < 0) | (x_tb > 1)).sum())
            counts["saturated"] += int(((mask < 0.01) | (mask > 0.99)).sum())
            lower_bound_sum += float((x_tb - x_tb.clamp(0, 1)).square().sum())
            mask_sum += float(mask.detach().sum())
            mask_square_sum += float(mask.detach().square().sum())
            total += loss.item()
        n = counts["values"]
        history.append(
            dict(
                epoch=epoch + 1,
                reconstruction_mse=total / len(loader),
                outside_unit_interval=counts["outside_unit_interval"] / n,
                saturated_fraction=counts["saturated"] / n,
                bounded_reconstruction_mse_lower_bound=lower_bound_sum / n,
                mask_mean=mask_sum / n,
                mask_std=max(0, mask_square_sum / n - (mask_sum / n) ** 2) ** 0.5,
                encoder_training=encoder.training,
            )
        )
        Path(str(output_path) + ".epochs.json").write_text(
            json.dumps(history, indent=2, allow_nan=False) + "\n"
        )
        print(
            f"CoRTX decoder epoch {epoch + 1}/{decoder_epochs}: {total / len(loader):.6f}"
        )
    torch.save(
        {
            "dataset": dataset,
            "encoder": cpu_state_dict(encoder),
            "decoder": cpu_state_dict(decoder),
        },
        output_path,
    )


def train_sgt(dataset, loader, shape, output_path, epochs, device, val):
    model = make_transformer(dataset, *shape).to(device)
    lr, weight_decay, default_epochs = SGT_TRAINING[dataset]
    epochs = epochs or default_epochs
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    original = dataset in {"freqshape", "scs_better"}
    best_f1 = -float("inf")
    history = []
    for epoch in range(epochs):
        model.train()
        total = 0.0
        classification_total = consistency_total = 0.0
        for x, times, target in loader:
            model.eval()
            scores = absolute_input_gradients(model, x, times, target)
            masked = mask_bottom_features(x, scores, fraction=0.9)
            model.train()
            logits = model(x, times, captum_input=True)
            masked_logits = model(masked, times, captum_input=True)
            if original:
                classification = F.cross_entropy(logits, target)
                consistency = F.kl_div(
                    masked_logits.log_softmax(dim=1),
                    logits.detach().softmax(dim=1),
                    reduction="batchmean",
                )
            else:
                classification, consistency = sgt_objective(
                    logits, masked_logits, target
                )
            loss = classification + consistency
            checked_step(loss, optimizer, model.parameters())
            total += loss.item()
            classification_total += classification.item()
            consistency_total += consistency.item()
        if original:
            # The completed Table 1 runs used CE, detached KL and the final epoch.
            print(f"SGT epoch {epoch + 1}/{epochs}: loss={total / len(loader):.6f}")
            continue
        model.eval()
        score = float(eval_mvts_transformer(val, model))
        if not torch.isfinite(torch.tensor(score)):
            raise FloatingPointError("Nonfinite SGT validation macro-F1")
        selected = score > best_f1
        if selected:
            best_f1 = score
            best_epoch = epoch + 1
            torch.save(cpu_state_dict(model), output_path)
        history.append(
            dict(
                epoch=epoch + 1,
                loss=total / len(loader),
                classification=classification_total / len(loader),
                kl=consistency_total / len(loader),
                validation_macro_f1=score,
                selected_epoch=best_epoch,
            )
        )
        Path(str(output_path) + ".epochs.json").write_text(
            json.dumps(history, indent=2, allow_nan=False) + "\n"
        )
        quality = dict(
            validation_macro_f1=best_f1,
            selected_epoch=best_epoch,
            recipe="poly1-kl-validation",
        )
        Path(str(output_path) + ".quality.json").write_text(
            json.dumps(quality, indent=2) + "\n"
        )
        print(
            f"SGT epoch {epoch + 1}/{epochs}: loss={total / len(loader):.6f}, validation macro-F1={score:.4f}"
        )

    if original:
        torch.save(cpu_state_dict(model), output_path)


def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    directory, model_dataset, predictor_name = DATASETS[args.dataset]
    data_path = Path(args.data_root) / directory
    models_path = args.models_path or EXPERIMENTS_ROOT / model_dataset / "models"
    models_path.mkdir(parents=True, exist_ok=True)
    splits = [args.split_no] if args.split_no else range(1, 6)
    for split in splits:
        seed_everything(args.seed + split - 1)
        data = process_Synth(split_no=split, device=device, base_path=data_path)
        loader = batches(data)
        max_len, _, d_inp = data["val"][0].shape
        shape = (d_inp, max_len)
        output = models_path / f"{args.method}_split={split}.pt"
        if output.exists() and not args.force:
            print(f"Skipping existing checkpoint: {output}")
            continue
        if args.method == "cortx":
            predictor = models_path / predictor_name.format(split)
            if not predictor.is_file():
                raise FileNotFoundError(f"missing predictor checkpoint: {predictor}")
            train_cortx(
                model_dataset,
                loader,
                shape,
                predictor,
                output,
                args.encoder_epochs,
                args.decoder_epochs,
                device,
            )
        else:
            train_sgt(
                model_dataset,
                loader,
                shape,
                output,
                args.sgt_epochs,
                device,
                data["val"],
            )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=DATASETS, required=True)
    parser.add_argument("--method", choices=("cortx", "sgt"), required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--split-no", type=int, choices=range(1, 6))
    parser.add_argument("--encoder-epochs", type=int, default=100)
    parser.add_argument("--decoder-epochs", type=int, default=50)
    parser.add_argument(
        "--sgt-epochs", type=int, help="override the per-dataset SGT epoch budget"
    )
    parser.add_argument("--data-root", type=Path, default=DATA_ROOT)
    parser.add_argument("--models-path", type=Path)
    parser.add_argument("--force", action="store_true")
    main(parser.parse_args())
