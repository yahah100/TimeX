import argparse
import json
from pathlib import Path

import torch

from txai.utils.predictors.loss import Poly1CrossEntropyLoss
from txai.trainers.train_transformer import train
from txai.models.encoders.transformer_simple import TransformerMVTS
from txai.utils.data import process_Synth
from txai.utils.predictors import eval_mvts_transformer
from txai.synth_data.simple_spike import SpikeTrainDataset
from txai.utils.constants import dataset_path
from txai.utils.reproducibility import seed_everything

parser = argparse.ArgumentParser()
parser.add_argument(
    "--seed", type=int, default=42, help="base random seed (default: 42)"
)
parser.add_argument("--split-no", type=int, choices=range(1, 6))
parser.add_argument("--epochs", type=int, default=200)
parser.add_argument(
    "--models-path", type=Path, default=Path(__file__).resolve().parent / "models"
)
parser.add_argument("--data-path", type=Path, default=dataset_path("SeqCombSingle"))
args = parser.parse_args()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

clf_criterion = Poly1CrossEntropyLoss(
    num_classes=4, epsilon=1.0, weight=None, reduction="mean"
)

for i in [args.split_no] if args.split_no else range(1, 6):
    seed_everything(args.seed + i - 1)
    D = process_Synth(split_no=i, device=device, base_path=args.data_path)
    train_loader = torch.utils.data.DataLoader(
        D["train_loader"], batch_size=64, shuffle=True
    )

    val, test = D["val"], D["test"]

    model = TransformerMVTS(
        d_inp=val[0].shape[-1],
        max_len=val[0].shape[0],
        n_classes=4,
        nlayers=2,
        nhead=1,
        trans_dim_feedforward=64,
        trans_dropout=0.25,
        d_pe=16,
        # aggreg = 'mean',
        # norm_embedding = True
    )

    model.to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.01)

    model_dir = args.models_path
    model_dir.mkdir(parents=True, exist_ok=True)
    spath = model_dir / "Scomb_transformer_split={}.pt".format(i)

    model, loss, auc = train(
        model,
        train_loader,
        val_tuple=val,
        n_classes=4,
        num_epochs=args.epochs,
        save_path=spath,
        optimizer=optimizer,
        show_sizes=False,
        use_scheduler=False,
    )

    model_sdict_cpu = {k: v.cpu() for k, v in model.state_dict().items()}
    torch.save(
        model_sdict_cpu, model_dir / "Scomb_transformer_split={}_cpu.pt".format(i)
    )

    score = float(max(auc))
    spath.with_suffix(".quality.json").write_text(
        json.dumps(
            {
                "validation_macro_f1": score,
                "qualified": score >= 0.95,
                "selected_epoch": auc.index(score) + 1,
            },
            indent=2,
        )
        + "\n"
    )
    f1 = eval_mvts_transformer(test, model)
    print("Test F1: {:.4f}".format(f1))
