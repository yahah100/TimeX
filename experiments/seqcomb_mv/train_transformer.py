"""Train the SeqComb-MV reference transformer for Table 2."""

import argparse
import hashlib
import json
import shutil
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
    parser.add_argument("--max-attempts", type=int, choices=range(1, 4), default=3)
    parser.add_argument("--min-val-f1", type=float, default=0.95)
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
        data = process_Synth(split_no=split, device=device, base_path=args.data_path)
        train_loader = torch.utils.data.DataLoader(
            data["train_loader"], batch_size=64, shuffle=True
        )
        val, test = data["val"], data["test"]
        attempts = []
        quality_path = args.models_path / f"transformer_split={split}.quality.json"
        for attempt in range(args.max_attempts):
            attempt_seed = args.seed + split - 1 + 1000 * attempt
            seed_everything(attempt_seed)
            model = TransformerMVTS(
                d_inp=val[0].shape[-1],
                max_len=val[0].shape[0],
                n_classes=4,
                trans_dim_feedforward=128,
                nlayers=2,
                trans_dropout=0.25,
                d_pe=16,
            ).to(device)
            optimizer = torch.optim.AdamW(
                model.parameters(), lr=5e-4, weight_decay=0.001
            )
            save_path = (
                args.models_path / f"transformer_split={split}_attempt={attempt}.pt"
            )
            model, _, val_scores = train(
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
                {key: value.cpu() for key, value in model.state_dict().items()},
                save_path,
            )
            score = float(max(val_scores))
            attempts.append(
                dict(
                    attempt=attempt,
                    seed=attempt_seed,
                    validation_macro_f1=score,
                    selected_epoch=val_scores.index(score) + 1,
                    checkpoint=save_path.name,
                    checkpoint_sha256=hashlib.sha256(
                        save_path.read_bytes()
                    ).hexdigest(),
                )
            )
            quality = dict(
                attempts=attempts,
                validation_macro_f1=score,
                minimum_macro_f1=args.min_val_f1,
                qualified=score >= args.min_val_f1,
            )
            quality_path.write_text(
                json.dumps(quality, indent=2, allow_nan=False) + "\n"
            )
            if quality["qualified"]:
                shutil.copy2(
                    save_path, args.models_path / f"transformer_split={split}.pt"
                )
                print(
                    f"Split {split} accepted attempt {attempt}: validation macro-F1 {score:.4f}"
                )
                break
        else:
            raise RuntimeError(
                f"Split {split}: predictor failed validation macro-F1 >= {args.min_val_f1} after {args.max_attempts} attempts"
            )


if __name__ == "__main__":
    main(parse_args())
