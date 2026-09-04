"""Train the SeqComb-MV reference transformer for Table 2."""

import argparse
from pathlib import Path

import torch

from txai.models.encoders.transformer_simple import TransformerMVTS
from txai.trainers.train_transformer import train
from txai.utils.constants import DATA_ROOT
from txai.utils.data import process_Synth
from txai.utils.predictors import eval_mvts_transformer
from txai.utils.reproducibility import seed_everything


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--split-no", type=int, choices=range(1, 6))
    parser.add_argument("--epochs", type=int, default=1000)
    parser.add_argument("--data-path", type=Path, default=DATA_ROOT / "SeqCombMV")
    parser.add_argument(
        "--models-path", type=Path, default=Path(__file__).parent / "models"
    )
    return parser.parse_args()


def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args.models_path.mkdir(parents=True, exist_ok=True)
    splits = [args.split_no] if args.split_no else range(1, 6)
    for split in splits:
        seed_everything(args.seed + split - 1)
        data = process_Synth(split_no=split, device=device, base_path=args.data_path)
        train_loader = torch.utils.data.DataLoader(
            data["train_loader"], batch_size=64, shuffle=True
        )
        val, test = data["val"], data["test"]
        model = TransformerMVTS(
            d_inp=val[0].shape[-1],
            max_len=val[0].shape[0],
            n_classes=4,
            trans_dim_feedforward=128,
            nlayers=2,
            trans_dropout=0.25,
            d_pe=16,
        ).to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=5e-4, weight_decay=0.001)
        save_path = args.models_path / f"transformer_split={split}.pt"
        model, _, _ = train(
            model,
            train_loader,
            val_tuple=val,
            n_classes=4,
            num_epochs=args.epochs,
            save_path=save_path,
            optimizer=optimizer,
            show_sizes=False,
            use_scheduler=False,
        )
        torch.save(
            {key: value.cpu() for key, value in model.state_dict().items()}, save_path
        )
        print(f"Split {split} test F1: {eval_mvts_transformer(test, model):.4f}")


if __name__ == "__main__":
    main(parse_args())
