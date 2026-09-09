"""Shared implementations for the CoRTX and SGT + Grad synthetic baselines."""

import torch
import torch.nn.functional as F

from txai.models.encoders.transformer_simple import TransformerMVTS
from txai.models.mask_generators.maskgen import MaskGenerator


# Architectures of the reference predictors these baselines stand in for; `d_inp` and
# `max_len` are read off the data instead, since FreqShape is T=50 and the rest T=200.
DATASET_CONFIGS = {
    "freqshape": dict(
        n_classes=4,
        trans_dim_feedforward=16,
        trans_dropout=0.1,
        d_pe=16,
    ),
    "scs_better": dict(
        n_classes=4,
        nlayers=2,
        nhead=1,
        trans_dim_feedforward=64,
        trans_dropout=0.25,
        d_pe=16,
    ),
    "seqcomb_mv": dict(
        n_classes=4,
        trans_dim_feedforward=128,
        nlayers=2,
        trans_dropout=0.25,
        d_pe=16,
    ),
    "lowvardetect": dict(
        n_classes=4,
        trans_dim_feedforward=32,
        nlayers=1,
        nhead=1,
        trans_dropout=0.25,
        d_pe=16,
        norm_embedding=True,
    ),
}


def make_transformer(dataset, d_inp, max_len):
    """Construct the paper's transformer architecture for a synthetic dataset."""
    return TransformerMVTS(d_inp=d_inp, max_len=max_len, **DATASET_CONFIGS[dataset])


def make_cortx_decoder(dataset, d_inp, max_len):
    """Construct a continuous CoRTX mask decoder over the encoder's sequence embeddings."""
    return MaskGenerator(
        d_z=d_inp + DATASET_CONFIGS[dataset]["d_pe"],
        max_len=max_len,
        tau=1.0,
        use_ste=False,
    )


def cortx_mask(decoder, z_seq, src, times):
    """Return the CoRTX decoder's continuous mask time-first, ``(T, B, d)``.

    ``MaskGenerator`` emits batch-first masks for univariate inputs and time-first masks
    for multivariate ones; every caller here wants time-first.

    Args:
        decoder: The ``MaskGenerator`` built by :func:`make_cortx_decoder`.
        z_seq: Encoder sequence embeddings, (T, B, d_z).
        src: Input series, (T, B, d).
        times: Timestamps, (T, B).

    Returns:
        The continuous mask, (T, B, d).
    """
    mask, _ = decoder(z_seq, src, times)
    return mask.transpose(0, 1) if decoder.d_inp == 1 else mask


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
    masked = x.detach().clone().contiguous()
    flat = masked.flatten(1)
    flat_scores = scores.flatten(1)
    if not 0 <= fraction <= 1:
        raise ValueError("mask fraction must be between zero and one")
    count = int(flat.shape[1] * fraction)
    indices = flat_scores.topk(count, dim=1, largest=False).indices
    mins = flat.min(dim=1, keepdim=True).values
    spans = flat.max(dim=1, keepdim=True).values - mins
    replacements = mins + torch.rand_like(indices, dtype=flat.dtype) * spans
    flat.scatter_(1, indices, replacements)
    return masked


def sgt_objective(logits, masked_logits, targets):
    """Released Poly1 + KL(P(original) || P(masked)), gradients through both views."""
    from txai.utils.predictors.loss import Poly1CrossEntropyLoss

    classification = Poly1CrossEntropyLoss(
        num_classes=logits.shape[-1], epsilon=1.0, reduction="mean"
    )(logits, targets)
    consistency = F.kl_div(
        F.log_softmax(masked_logits, dim=1),
        F.softmax(logits, dim=1),
        reduction="batchmean",
    )
    return classification, consistency


def checked_step(loss, optimizer, parameters):
    """Fail explicitly before applying a nonfinite update."""
    if not torch.isfinite(loss):
        raise FloatingPointError("Nonfinite baseline loss")
    optimizer.zero_grad()
    loss.backward()
    if any(p.grad is not None and not torch.isfinite(p.grad).all() for p in parameters):
        raise FloatingPointError("Nonfinite baseline gradient")
    optimizer.step()
