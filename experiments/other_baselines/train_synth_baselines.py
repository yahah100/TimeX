"""Train CoRTX or SGT + Grad on the multivariate Table 2 datasets."""

import argparse
from pathlib import Path

import torch
import torch.nn.functional as F

from txai.baselines.table2 import (
    absolute_input_gradients,
    make_cortx_decoder,
    make_transformer,
    mask_bottom_features,
    symmetric_infonce,
)
from txai.utils.constants import DATA_ROOT, EXPERIMENTS_ROOT
from txai.utils.data import process_Synth
from txai.utils.reproducibility import seed_everything


DATASETS = {
    "seqcomb_mv": ("SeqCombMV", "seqcomb_mv"),
    "lowvar": ("LowVarDetect", "lowvardetect"),
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
    dataset, loader, predictor_path, output_path, epochs, decoder_epochs, device
):
    reference = make_transformer(dataset).to(device)
    reference.load_state_dict(torch.load(predictor_path, map_location=device))
    encoder = make_transformer(dataset).to(device)
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
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total += loss.item()
        print(f"CoRTX encoder epoch {epoch + 1}/{epochs}: {total / len(loader):.6f}")

    decoder = make_cortx_decoder(dataset).to(device)
    optimizer = torch.optim.Adam(decoder.parameters(), lr=5e-3)
    for epoch in range(decoder_epochs):
        decoder.train()
        total = 0.0
        for x, times, _ in loader:
            x_tb, times_tb = x.transpose(0, 1), times.transpose(0, 1)
            with torch.no_grad():
                z_seq = encoder.embed(x_tb, times_tb, aggregate=False)
            mask, _ = decoder(z_seq, x_tb, times_tb)
            loss = F.mse_loss(mask, x_tb)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total += loss.item()
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


def train_sgt(dataset, loader, output_path, epochs, device):
    model = make_transformer(dataset).to(device)
    config = {"seqcomb_mv": (5e-4, 0.01), "lowvardetect": (1e-3, 0.01)}[dataset]
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config[0], weight_decay=config[1]
    )
    for epoch in range(epochs):
        model.train()
        total = 0.0
        for x, times, target in loader:
            model.eval()
            scores = absolute_input_gradients(model, x, times, target)
            masked = mask_bottom_features(x, scores, fraction=0.9)
            model.train()
            logits = model(x, times, captum_input=True)
            masked_logits = model(masked, times, captum_input=True)
            loss = F.cross_entropy(logits, target) + F.kl_div(
                F.log_softmax(masked_logits, dim=1),
                F.softmax(logits.detach(), dim=1),
                reduction="batchmean",
            )
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total += loss.item()
        print(f"SGT epoch {epoch + 1}/{epochs}: {total / len(loader):.6f}")
    torch.save(cpu_state_dict(model), output_path)


def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    directory, model_dataset = DATASETS[args.dataset]
    data_path = Path(args.data_root) / directory
    models_path = args.models_path or EXPERIMENTS_ROOT / model_dataset / "models"
    models_path.mkdir(parents=True, exist_ok=True)
    splits = [args.split_no] if args.split_no else range(1, 6)
    for split in splits:
        seed_everything(args.seed + split - 1)
        data = process_Synth(split_no=split, device=device, base_path=data_path)
        loader = batches(data)
        output = models_path / f"{args.method}_split={split}.pt"
        if output.exists() and not args.force:
            print(f"Skipping existing checkpoint: {output}")
            continue
        if args.method == "cortx":
            predictor = models_path / f"transformer_split={split}.pt"
            if not predictor.is_file():
                raise FileNotFoundError(f"missing predictor checkpoint: {predictor}")
            train_cortx(
                model_dataset,
                loader,
                predictor,
                output,
                args.encoder_epochs,
                args.decoder_epochs,
                device,
            )
        else:
            train_sgt(model_dataset, loader, output, args.sgt_epochs, device)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=DATASETS, required=True)
    parser.add_argument("--method", choices=("cortx", "sgt"), required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--split-no", type=int, choices=range(1, 6))
    parser.add_argument("--encoder-epochs", type=int, default=100)
    parser.add_argument("--decoder-epochs", type=int, default=50)
    parser.add_argument("--sgt-epochs", type=int, default=10)
    parser.add_argument("--data-root", type=Path, default=DATA_ROOT)
    parser.add_argument("--models-path", type=Path)
    parser.add_argument("--force", action="store_true")
    main(parser.parse_args())
