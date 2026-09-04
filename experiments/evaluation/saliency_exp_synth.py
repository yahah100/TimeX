import json
import re, time
import argparse
from pathlib import Path
import sys
import torch
import numpy as np
import matplotlib.pyplot as plt
from tqdm import trange

from txai.models.encoders.transformer_simple import TransformerMVTS
from txai.models.encoders.simple import CNN, LSTM
from txai.utils.experimental import get_explainer
from txai.vis.vis_saliency import vis_one_saliency
from txai.utils.data import process_Synth
from txai.utils.data.preprocess import process_MITECG, process_Boiler
from txai.synth_data.simple_spike import SpikeTrainDataset
from txai.utils.data.preprocess import process_Epilepsy, process_PAM
from txai.utils.constants import DATA_ROOT
from txai.utils.reproducibility import seed_everything
from txai.baselines.table2 import (
    absolute_input_gradients,
    make_cortx_decoder,
    make_transformer,
)

try:
    from txai.utils.data.anomaly import process_Yahoo
except ImportError:
    process_Yahoo = None

from txai.models.modelv6_v2 import Modelv6_v2
from txai.models.bc_model import TimeXModel
from txai.models.bc_model_irreg import TimeXModel_Irregular

from txai.utils.evaluation import ground_truth_xai_eval, ground_truth_IoU

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def get_model(args, X):

    if args.model_type == "cnn":
        # TODO set n_classes dependant on dataset
        model = CNN(
            d_inp=X.shape[-1],
            n_classes=2
            if (args.dataset == "mitecg_hard" or args.dataset == "anomaly")
            else 4,
            dim=64 if args.dataset == "anomaly" else 32,
        )
    elif args.model_type == "lstm":
        # TODO set n_classes dependent on dataset
        model = LSTM(
            d_inp=X.shape[-1], n_classes=2 if args.dataset == "mitecg_hard" else 4
        )

    else:  # transformer
        if args.dataset == "scs_better":
            model = TransformerMVTS(
                d_inp=X.shape[-1],
                max_len=X.shape[0],
                n_classes=4,
                nlayers=2,
                nhead=1,
                trans_dim_feedforward=64,
                trans_dropout=0.25,
                d_pe=16,
            )

        elif args.dataset == "freqshape":
            model = TransformerMVTS(
                d_inp=X.shape[-1],
                max_len=X.shape[0],
                n_classes=4,
                trans_dim_feedforward=16,
                trans_dropout=0.1,
                d_pe=16,
            )

        elif args.dataset == "scs_inline":
            model = TransformerMVTS(
                d_inp=1,
                max_len=200,
                n_classes=4,
                nlayers=2,
                nhead=1,
                trans_dim_feedforward=128,
                trans_dropout=0.2,
                d_pe=16,
                # aggreg = 'mean',
                # norm_embedding = True
            )

        elif args.dataset == "scs_fixone":
            model = TransformerMVTS(
                d_inp=X.shape[-1],
                max_len=X.shape[0],
                nlayers=2,
                n_classes=4,
                trans_dim_feedforward=32,
                trans_dropout=0.1,
                d_pe=16,
            )

        elif args.dataset == "seqcomb_mv":
            model = TransformerMVTS(
                d_inp=X.shape[-1],
                max_len=X.shape[0],
                nlayers=2,
                n_classes=4,
                trans_dim_feedforward=128,
                trans_dropout=0.25,
                d_pe=16,
            )

        elif args.dataset == "mitecg_hard":
            model = TransformerMVTS(
                d_inp=X.shape[-1],
                max_len=X.shape[0],
                nlayers=1,
                n_classes=2,
                trans_dim_feedforward=64,
                trans_dropout=0.1,
                d_pe=16,
                stronger_clf_head=False,
                norm_embedding=True,
            )

        elif args.dataset == "lowvardetect":
            model = TransformerMVTS(
                d_inp=X.shape[-1],
                max_len=X.shape[0],
                nlayers=1,
                n_classes=4,
                trans_dim_feedforward=32,
                trans_dropout=0.25,
                d_pe=16,
                stronger_clf_head=False,
            )

        elif args.dataset == "boiler":
            model = TransformerMVTS(
                d_inp=X.shape[-1],
                max_len=X.shape[0],
                nlayers=1,
                n_classes=2,
                trans_dim_feedforward=64,
                trans_dropout=0.25,
                d_pe=16,
                stronger_clf_head=False,
            )

        elif args.dataset == "epilepsy":
            model = TransformerMVTS(
                d_inp=X.shape[-1],
                max_len=X.shape[0],
                n_classes=2,
                nlayers=1,
                trans_dim_feedforward=16,
                trans_dropout=0.1,
                d_pe=16,
                norm_embedding=False,
            )

        elif args.dataset == "pam":
            model = TransformerMVTS(
                d_inp=X.shape[2],
                max_len=X.shape[0],
                n_classes=8,
            )

        elif args.dataset == "irreg":
            model = TransformerMVTS(
                d_inp=X.shape[2],
                max_len=X.shape[0],
                n_classes=4,
                trans_dim_feedforward=128,
                nlayers=2,
                trans_dropout=0.25,
                d_pe=16,
                # aggreg = 'mean',
                # norm_embedding = True
            )

        # elif args.dataset == 'anomaly':
        #     model = CNN(
        #         d_inp = val[0].shape[-1],
        #         n_classes = 2,
        #         dim = 32,
        #     )

    # model = torch.compile(model)

    return model


def main(args):

    Dname = args.dataset.lower()

    # Switch on loading test data:
    if Dname == "freqshape":
        D = process_Synth(
            split_no=args.split_no,
            device=device,
            base_path=Path(args.data_path) / "FreqShape",
        )
    elif Dname == "seqcombsingle":
        D = process_Synth(
            split_no=args.split_no,
            device=device,
            base_path=Path(args.data_path) / "SeqCombSingle",
        )
    elif Dname == "scs_better":
        D = process_Synth(
            split_no=args.split_no,
            device=device,
            base_path=Path(args.data_path) / "SeqCombSingle",
        )
    elif Dname == "seqcomb_mv":
        D = process_Synth(
            split_no=args.split_no,
            device=device,
            base_path=Path(args.data_path) / "SeqCombMV",
        )
    elif Dname == "freqshapeud":
        D = process_Synth(
            split_no=args.split_no,
            device=device,
            base_path=Path(args.data_path) / "FreqShapeUD",
        )
    elif Dname == "scs_inline":
        D = process_Synth(
            split_no=args.split_no,
            device=device,
            base_path=Path(args.data_path) / "SeqCombSingleInline",
        )
    elif Dname == "scs_fixone":
        D = process_Synth(
            split_no=args.split_no,
            device=device,
            base_path=Path(args.data_path) / "SeqCombSingleFixOne",
        )
    elif Dname == "mitecg_hard":
        D = process_MITECG(
            split_no=args.split_no,
            device=device,
            hard_split=True,
            need_binarize=True,
            exclude_pac_pvc=True,
            base_path=Path(args.data_path) / "MITECG",
        )
    elif Dname == "lowvardetect":
        D = process_Synth(
            split_no=args.split_no,
            device=device,
            base_path=Path(args.data_path) / "LowVarDetect",
        )
    elif Dname == "boiler":
        D = process_Boiler(
            split_no=args.split_no,
            device=device,
            base_path=Path(args.data_path) / "Boiler",
            normalize=True,
        )
    elif Dname == "anomaly":
        if process_Yahoo is None:
            raise RuntimeError(
                "The optional anomaly dataset loader is not included in this repository."
            )
        D = process_Yahoo(split_no=args.split_no, device=device, balance=False)
    elif Dname == "epilepsy":
        trainEpi, val, test = process_Epilepsy(
            split_no=args.split_no,
            device=device,
            base_path=Path(args.data_path) / "Epilepsy",
        )
    elif Dname == "pam":
        trainEpi, val, test = process_PAM(
            split_no=args.split_no,
            device=device,
            base_path=Path(args.data_path) / "PAM",
            gethalf=True,
        )
    elif Dname == "irreg":
        D = process_Synth(
            split_no=args.split_no,
            device=device,
            base_path=Path(args.data_path) / "SeqCombMVIrreg",
        )

    if (Dname == "mitecg_hard") or (Dname == "boiler"):
        _, _, test, gt_exps = D
    # elif Dname == 'anomaly':
    #     test = D['test']
    #     gt_exps = D['gt_exps']
    elif Dname in {"epilepsy", "pam"}:
        val = (val.X, val.time, val.y)
        test = (test.X, test.time, test.y)
    else:
        test = D["test"]

    if (
        Dname == "scs_better"
        or Dname == "seqcombsingle"
        or Dname == "scs_inline"
        or Dname == "seqcomb_mv"
        or Dname == "lowvardetect"
        or Dname == "anomaly"
        or Dname == "irreg"
    ):
        y = test[2]
        X = test[0][:, (y != 0), :]
        times = test[1][:, y != 0]
        gt_exps = D["gt_exps"][:, (y != 0).detach().cpu(), :]
        y = y[y != 0]
    # elif Dname == 'lowvardetect':
    # y = test[2]
    # X = test[0][:,(y == 2),:]
    # times = test[1][:,y == 2]
    # gt_exps = D['gt_exps'][:,(y == 2).detach().cpu(),:]
    # y = y[y == 2]
    elif Dname == "mitecg_hard":
        X, times, y = test.X, test.time, test.y

        # Filter based on 0 samples:
        mask = (y == 1).clone().cpu()
        # Detect when we fail to observe a wave:
        detection_failure = gt_exps.squeeze().sum(0) > 0
        mask = mask & detection_failure
        X = X[:, mask, :]
        times = times[:, mask]
        gt_exps = gt_exps[:, mask, :]
        y = y[mask]
        # exit()
        print(gt_exps.shape)
        print((gt_exps.squeeze().sum(0) == 0).sum())
        # exit()
    elif Dname == "boiler":
        X, times, y = test
        mask = (y == 1).clone().cpu()
        X = X[:, mask, :]
        times = times[:, mask]
        y = y[mask]
        gt_exps = gt_exps[:, mask, :]
        print("gt exps", gt_exps.shape)
        # exit()
    else:
        X, times, y = test
        if Dname not in {"epilepsy", "pam"}:
            gt_exps = D["gt_exps"]
    max_samples = getattr(args, "max_samples", None)
    if max_samples is not None:
        X = X[:, :max_samples]
        times = times[:, :max_samples]
        y = y[:max_samples]
        gt_exps = gt_exps[:, :max_samples]
    T, B, d = X.shape

    if args.exp_method == "ours":
        sdict, config = torch.load(args.model_path, map_location=device)
        # print('Config', config)
        if args.org_v:
            model = Modelv6_v2(**config)
        elif Dname == "irreg":
            model = TimeXModel_Irregular(**config)
        else:
            model = TimeXModel(**config)
        model.load_state_dict(sdict)
        model.eval()
        model.to(device)

        # Keep batch size at 64:
        iters = torch.arange(0, B, step=64)
        generated_exps = torch.zeros_like(X)
        zx_generated = torch.zeros(X.shape[1], model.d_z)

        start_time = time.time()

        for i in range(len(iters)):
            if i == (len(iters) - 1):
                batch_X = X[:, iters[i] :, :]
                batch_times = times[:, iters[i] :]
            else:
                batch_X = X[:, iters[i] : iters[i + 1], :]
                batch_times = times[:, iters[i] : iters[i + 1]]

            with torch.no_grad():
                if args.savepath is None:
                    out = model.get_saliency_explanation(
                        batch_X, batch_times, captum_input=False
                    )
                else:
                    out = model(batch_X, batch_times, captum_input=False)

            # NOTE: below capability only works with univariate for now - will need to edit after adding MV to model
            if args.org_v:
                if i == (len(iters) - 1):
                    generated_exps[:, iters[i] :, :] = (
                        torch.stack(out["mask_in"], dim=0)
                        .sum(dim=0)
                        .unsqueeze(-1)
                        .transpose(0, 1)
                    )
                else:
                    generated_exps[:, iters[i] : iters[i + 1], :] = (
                        torch.stack(out["mask_in"], dim=0)
                        .sum(dim=0)
                        .unsqueeze(-1)
                        .transpose(0, 1)
                    )
            else:
                if args.savepath is None:
                    if i == (len(iters) - 1):
                        if batch_X.shape[-1] == 1:
                            generated_exps[:, iters[i] :, :] = out["mask_in"]
                        else:
                            generated_exps[:, iters[i] :, :] = out["mask_in"].transpose(
                                0, 1
                            )
                    else:
                        if batch_X.shape[-1] == 1:
                            generated_exps[:, iters[i] : iters[i + 1], :] = out[
                                "mask_in"
                            ]
                        else:
                            generated_exps[:, iters[i] : iters[i + 1], :] = out[
                                "mask_in"
                            ].transpose(0, 1)
                else:
                    if i == (len(iters) - 1):
                        # if batch_X.shape[-1] == 1:
                        #     generated_exps[:,iters[i]:,:] = out['mask_logits']
                        #     zx_generated[:,iters[i]:] = out['z_mask_list']
                        # else:
                        generated_exps[:, iters[i] :, :] = out["mask_logits"].transpose(
                            0, 1
                        )
                        zx_generated[iters[i] :, :] = out["z_mask_list"]
                    else:
                        # if batch_X.shape[-1] == 1:
                        #     generated_exps[:,iters[i]:iters[i+1],:] = out['mask_logits']
                        #     zx_generated[:,iters[i]:iters[i+1]] = out['z_mask_list']
                        # else:
                        generated_exps[:, iters[i] : iters[i + 1], :] = out[
                            "mask_logits"
                        ].transpose(0, 1)
                        zx_generated[iters[i] : iters[i + 1], :] = out["z_mask_list"]

        end_time = time.time()

        print(
            "Time elapsed inference TimeX, split {}: {:.6f}".format(
                args.split_no, end_time - start_time
            )
        )

        if args.savepath is not None:
            # Get zp:
            zp = model.prototypes.detach().clone().cpu()
            zx = zx_generated.detach().clone().cpu()
            Xsave = X.detach().clone().cpu()
            ysave = y.detach().clone().cpu()
            gexp = generated_exps.detach().clone().cpu()
            torch.save(
                (Xsave, gexp, ysave, zx, zp), args.savepath
            )  # Saves all needed info

    elif args.exp_method == "cortx":
        checkpoint = torch.load(args.model_path, map_location=device)
        model = make_transformer(Dname).to(device)
        decoder = make_cortx_decoder(Dname).to(device)
        model.load_state_dict(checkpoint["encoder"])
        decoder.load_state_dict(checkpoint["decoder"])
        model.eval()
        decoder.eval()
        generated_exps = torch.zeros_like(X)
        for start in range(0, B, 64):
            batch_x = X[:, start : start + 64]
            batch_times = times[:, start : start + 64]
            with torch.no_grad():
                z_seq = model.embed(batch_x, batch_times, aggregate=False)
                mask, _ = decoder(z_seq, batch_x, batch_times)
            generated_exps[:, start : start + 64] = mask

    elif args.exp_method == "sgt+grad":
        model = make_transformer(Dname).to(device)
        model.load_state_dict(torch.load(args.model_path, map_location=device))
        model.eval()
        generated_exps = torch.zeros_like(X)
        for start in range(0, B, 64):
            batch_x = X[:, start : start + 64].transpose(0, 1)
            batch_times = times[:, start : start + 64].transpose(0, 1)
            batch_y = y[start : start + 64]
            grads = absolute_input_gradients(model, batch_x, batch_times, batch_y)
            generated_exps[:, start : start + 64] = grads.transpose(0, 1)

    elif args.exp_method == "winit":
        from winit_wrapper import WinITWrapper, aggregate_scores

        model = get_model(args, X)
        model.load_state_dict(torch.load(args.model_path, map_location=device))
        model.to(device)
        model.eval()
        winit_path = Path(args.model_path).parent / f"winit_split={args.split_no}/"
        winit = WinITWrapper(
            device, num_features=X.shape[-1], data_name=Dname, path=winit_path
        )
        winit.set_model(model)
        winit.load_generators()
        # winit wrapper expects shape of (batch, num_features, num_times) for X
        # and (batch, num_times) for times
        X_perm = X.permute(1, 2, 0)
        times_perm = times.permute(1, 0)
        start_time = time.time()
        attribution = winit.attribute(X_perm, times_perm)
        end_time = time.time()
        print("Time", end_time - start_time)
        if args.dataset == "epilepsy" or args.dataset == "pam":
            exit()
        # paper notes best performance with mean aggregation
        generated_exps = torch.from_numpy(aggregate_scores(attribution, "mean"))
        # permute (batch, features, times) back to (times, batch, features)
        generated_exps = generated_exps.permute(2, 0, 1)
    else:  # Use other explainer APIs:
        model = get_model(args, X)
        model.load_state_dict(torch.load(args.model_path, map_location=device))
        model.to(device)
        model.eval()
        # model.train()
        if args.model_type == "lstm" and args.exp_method in ["ig", "dyna"]:
            # training mode necessary for cudnn RNN backward
            model.train()

        explainer, needs_training = get_explainer(
            key=args.exp_method, args=args, device=device
        )

        generated_exps = torch.zeros_like(X)

        start_time = time.time()

        for i in trange(B, disable=not args.progress):
            # Eval all explainers:
            if args.exp_method == "dyna":  # This is a lazy solution, fix later
                exp = explainer(
                    model,
                    X[:, i, :].clone(),
                    times[:, i].clone().unsqueeze(1),
                    y=y[i].unsqueeze(0).clone(),
                )
            else:
                exp = explainer(
                    model,
                    X[:, i, :].unsqueeze(1).clone(),
                    times[:, i].unsqueeze(-1).clone(),
                    y[i].unsqueeze(0).clone(),
                )
            # print(exp.shape)
            generated_exps[:, i, :] = exp

        end_time = time.time()

        print(
            "Time elapsed inference {}, split {}: {:.6f}".format(
                args.exp_method, args.split_no, end_time - start_time
            )
        )

        if args.savepath is not None:  # Save based on provided location
            torch.save(generated_exps, args.savepath)

    if generated_exps.shape != gt_exps.shape:
        raise ValueError(
            f"explanation shape {tuple(generated_exps.shape)} does not match "
            f"ground truth {tuple(gt_exps.shape)}"
        )
    if not torch.isfinite(generated_exps).all():
        raise ValueError("explanations contain non-finite values")

    if Dname == "irreg":
        results_dict = ground_truth_xai_eval(generated_exps, gt_exps, times=times.cpu())
    else:
        results_dict = ground_truth_xai_eval(generated_exps, gt_exps)
    iou_dict = ground_truth_IoU(generated_exps, gt_exps)
    results_dict.update(iou_dict)

    # Show all results:
    print(
        "Results for {} explainer on {} with split={}".format(
            args.exp_method, args.dataset, args.split_no
        )
    )
    for k, v in results_dict.items():
        print(
            "\t{} \t = {:.4f} +- {:.4f}".format(
                k, np.mean(v), np.std(v) / np.sqrt(len(v))
            )
        )

    return results_dict


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--exp_method",
        required=True,
        choices=["ig", "dyna", "winit", "ours", "cortx", "sgt+grad"],
    )
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--split_no", default=1, type=int)
    parser.add_argument("--model_path", type=str, required=True, help="path to model")
    parser.add_argument(
        "--model_type",
        type=str,
        default="transformer",
        choices=["transformer", "cnn", "lstm"],
    )
    parser.add_argument("--org_v", action="store_true")
    parser.add_argument(
        "--data_path", default=DATA_ROOT, type=Path, help="path to datasets root"
    )
    parser.add_argument("--savepath", default=None, type=str)
    parser.add_argument(
        "--seed", type=int, default=0, help="base random seed (default: 0)"
    )
    parser.add_argument(
        "--results-json", type=Path, help="write structured evaluation results"
    )
    parser.add_argument(
        "--progress",
        action=argparse.BooleanOptionalAction,
        default=sys.stdout.isatty() and sys.stderr.isatty(),
    )
    parser.add_argument(
        "--max-samples", type=int, help="limit test samples for smoke testing"
    )

    args = parser.parse_args()
    if args.split_no == -1:
        # eval results on all splits
        results = {}
        fold_records = []
        model_path_template = args.model_path
        for split in range(1, 6):
            seed_everything(args.seed + split - 1)
            # replace model path with correct split
            args.model_path = re.sub(
                r"split=\d+", f"split={split}", model_path_template
            )
            print("model path:", args.model_path)
            args.split_no = split
            split_results = main(args)
            fold_records.append(
                {
                    "split": split,
                    "n_samples": len(next(iter(split_results.values()))),
                    "metrics": {k: float(np.mean(v)) for k, v in split_results.items()},
                }
            )
            for k, v in split_results.items():
                if k not in results:
                    results[k] = []
                results[k].extend(v)
        print(
            "Results for {} explainer on all splits".format(
                args.exp_method, args.dataset
            )
        )
        for k, v in results.items():
            fold_means = np.asarray([fold["metrics"][k] for fold in fold_records])
            print(
                "\t{} \t = {:.4f} +- {:.4f} (fold SE)".format(
                    k,
                    fold_means.mean(),
                    fold_means.std(ddof=1) / np.sqrt(len(fold_means)),
                )
            )
        output = {
            "schema_version": 1,
            "dataset": args.dataset,
            "method": args.exp_method,
            "base_seed": args.seed,
            "folds": fold_records,
            "pooled": {
                "metrics": {k: float(np.mean(v)) for k, v in results.items()},
                "n_samples": len(next(iter(results.values()))),
            },
            "cross_validation": {
                "n_folds": len(fold_records),
                "metrics": {
                    k: {
                        "mean": float(np.mean([f["metrics"][k] for f in fold_records])),
                        "standard_error": float(
                            np.std([f["metrics"][k] for f in fold_records], ddof=1)
                            / np.sqrt(len(fold_records))
                        ),
                    }
                    for k in results
                },
            },
        }
    else:
        seed_everything(args.seed + args.split_no - 1)
        split_results = main(args)
        metrics = {k: float(np.mean(v)) for k, v in split_results.items()}
        output = {
            "schema_version": 1,
            "dataset": args.dataset,
            "method": args.exp_method,
            "base_seed": args.seed,
            "folds": [
                {
                    "split": args.split_no,
                    "n_samples": len(next(iter(split_results.values()))),
                    "metrics": metrics,
                }
            ],
            "pooled": {
                "metrics": metrics,
                "n_samples": len(next(iter(split_results.values()))),
            },
            "cross_validation": {
                "n_folds": 1,
                "metrics": {
                    k: {"mean": v, "standard_error": None} for k, v in metrics.items()
                },
            },
        }
    if args.results_json:
        args.results_json.parent.mkdir(parents=True, exist_ok=True)
        args.results_json.write_text(json.dumps(output, indent=2) + "\n")
