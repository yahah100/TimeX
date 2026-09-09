import argparse
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
parser.add_argument('--seed', type=int, default=0, help='base random seed (default: 0)')
args = parser.parse_args()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

clf_criterion = Poly1CrossEntropyLoss(
    num_classes = 4,
    epsilon = 1.0,
    weight = None,
    reduction = 'mean'
)

for i in range(1, 6):
    seed_everything(args.seed + i - 1)
    D = process_Synth(split_no = i, device = device, base_path = dataset_path('FreqShape'))
    train_loader = torch.utils.data.DataLoader(D['train_loader'], batch_size = 64, shuffle = True)

    val, test = D['val'], D['test']

    model = TransformerMVTS(
        d_inp = val[0].shape[-1],
        max_len = val[0].shape[0],
        n_classes = 4,
        trans_dim_feedforward = 16,
        trans_dropout = 0.1,
        d_pe = 16,
        # aggreg = 'mean',
        # norm_embedding = True
    )

    model.to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr = 1e-3, weight_decay = 0.1)
    
    model_dir = Path(__file__).resolve().parent / 'models'
    model_dir.mkdir(exist_ok=True)
    spath = model_dir / 'Scomb_transformer_split={}.pt'.format(i)

    model, loss, auc = train(
        model,
        train_loader,
        val_tuple = val, 
        n_classes = 4,
        num_epochs = 100,
        save_path = spath,
        optimizer = optimizer,
        show_sizes = False,
        use_scheduler = False,
    )

    f1 = eval_mvts_transformer(test, model)
    print('Test F1: {:.4f}'.format(f1))
