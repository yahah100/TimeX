"""Shared implementations for the multivariate Table 2 baselines."""

import torch
import torch.nn.functional as F

from txai.models.encoders.transformer_simple import TransformerMVTS
from txai.models.mask_generators.maskgen import MaskGenerator


DATASET_CONFIGS = {
    "seqcomb_mv": dict(
        d_inp=4,
        max_len=200,
        n_classes=4,
        trans_dim_feedforward=128,
        nlayers=2,
        trans_dropout=0.25,
        d_pe=16,
    ),
    "lowvardetect": dict(
        d_inp=2,
        max_len=200,
        n_classes=4,
        trans_dim_feedforward=32,
        nlayers=1,
        nhead=1,
        trans_dropout=0.25,
        d_pe=16,
        norm_embedding=True,
    ),
}


def make_transformer(dataset):
    """Construct the paper's transformer architecture for a Table 2 dataset."""
    return TransformerMVTS(**DATASET_CONFIGS[dataset])


def make_cortx_decoder(dataset):
    """Construct a continuous multivariate CoRTX mask decoder."""
    config = DATASET_CONFIGS[dataset]
    return MaskGenerator(
        d_z=config["d_inp"] + config["d_pe"],
        max_len=config["max_len"],
        tau=1.0,
        use_ste=False,
    )


def symmetric_infonce(z1, z2, temperature=0.7):
    """Symmetric in-batch InfoNCE with paired rows as positives."""
    if z1.shape != z2.shape or z1.ndim != 2:
        raise ValueError("InfoNCE inputs must be equally shaped 2-D tensors")
    logits = F.normalize(z1, dim=-1) @ F.normalize(z2, dim=-1).T
    logits = logits / temperature
    labels = torch.arange(logits.shape[0], device=logits.device)
    return 0.5 * (F.cross_entropy(logits, labels) + F.cross_entropy(logits.T, labels))


def absolute_input_gradients(model, x, times, targets):
    """Return absolute target gradients for batch-first ``(B,T,d)`` input."""
    x_grad = x.detach().clone().requires_grad_(True)
    logits = model(x_grad, times, captum_input=True)
    selected = logits.gather(1, targets.view(-1, 1)).sum()
    return torch.autograd.grad(selected, x_grad)[0].abs()


def mask_bottom_features(x, scores, fraction=0.9):
    """Replace each sample's lowest-scoring fraction with uniform noise."""
    if x.shape != scores.shape:
        raise ValueError("input and score shapes must match")
    masked = x.detach().clone()
    flat = masked.flatten(1)
    flat_scores = scores.flatten(1)
    count = max(1, int(flat.shape[1] * fraction))
    indices = flat_scores.topk(count, dim=1, largest=False).indices
    mins = flat.min(dim=1, keepdim=True).values
    spans = flat.max(dim=1, keepdim=True).values - mins
    replacements = mins + torch.rand_like(indices, dtype=flat.dtype) * spans
    flat.scatter_(1, indices, replacements)
    return masked
