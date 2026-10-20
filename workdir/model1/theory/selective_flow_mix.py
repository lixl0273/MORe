from __future__ import annotations

from typing import Tuple, Dict

import torch
import torch.nn.functional as F


def minmax_norm(x: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    xmin = x.amin(dim=(2, 3), keepdim=True)
    xmax = x.amax(dim=(2, 3), keepdim=True)
    return (x - xmin) / (xmax - xmin + eps)


def motion_saliency_mask(
    flow: torch.Tensor,
    q: float = 0.6,
    kernel_size: int = 5,
    temperature: float | None = None,
    eps: float = 1e-6,
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    """
    Build selective flow-mix mask from motion magnitude and directional consistency.

    Args:
        flow: [B,3,H,W], channels [u,v,strain]
        q: quantile threshold
        temperature: if set, use differentiable sigmoid mask

    Returns:
        mask: [B,1,H,W]
        aux: magnitude, consistency, saliency, threshold
    """
    if flow.dim() != 4:
        raise ValueError(f"flow must be [B,C,H,W], got {tuple(flow.shape)}")
    if flow.size(1) < 2:
        raise ValueError("flow must contain u/v channels")

    u = flow[:, 0:1]
    v = flow[:, 1:2]

    mag = torch.sqrt(u ** 2 + v ** 2 + eps)
    mag_norm = minmax_norm(mag, eps=eps)

    theta = torch.atan2(v, u + eps)
    cos_t = torch.cos(theta)
    sin_t = torch.sin(theta)

    pad = kernel_size // 2
    avg_cos = F.avg_pool2d(cos_t, kernel_size=kernel_size, stride=1, padding=pad)
    avg_sin = F.avg_pool2d(sin_t, kernel_size=kernel_size, stride=1, padding=pad)
    consistency = torch.sqrt(avg_cos ** 2 + avg_sin ** 2 + eps).clamp(0.0, 1.0)

    saliency = mag_norm * consistency

    flat = saliency.flatten(1)
    threshold = torch.quantile(flat, q=float(q), dim=1, keepdim=True).view(-1, 1, 1, 1)

    if temperature is None or temperature <= 0:
        mask = (saliency >= threshold).float()
    else:
        mask = torch.sigmoid((saliency - threshold) / temperature)

    aux = {
        "magnitude": mag,
        "directional_consistency": consistency,
        "saliency": saliency,
        "threshold": threshold,
    }
    return mask, aux


def build_same_class_index(labels: torch.Tensor) -> torch.Tensor:
    """
    For each sample, choose another sample from the same class.

    Critical behavior:
    - If a class appears at least twice in the mini-batch, samples are paired within that class.
    - If a class appears only once, keep identity instead of falling back to random cross-class mixing.
      This avoids hidden label noise in same-class selective flow-mix.
    """
    device = labels.device
    B = labels.numel()
    index = torch.arange(B, device=device)

    for c in labels.unique():
        pos = torch.nonzero(labels == c, as_tuple=False).flatten()

        if pos.numel() >= 2:
            # Same-class deterministic cyclic pairing.
            # This guarantees index[pos] != pos for every sample in this class.
            index[pos] = torch.roll(pos, shifts=1, dims=0)

        # If pos.numel() == 1, keep identity.
        # Do not replace with a random cross-class index.

    return index


def selective_flow_mix(
    flow: torch.Tensor,
    labels: torch.Tensor,
    alpha: float = 0.4,
    q: float = 0.6,
    same_class: bool = True,
    temperature: float | None = None,
) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, torch.Tensor]]:
    """
    Selective flow-mix in the optical-flow domain.

    Args:
        flow: [B,3,H,W]
        labels: [B]
        same_class: if True, mix with same-class samples when possible

    Returns:
        mixed_flow: [B,3,H,W]
        labels_mix: unchanged hard labels for same-class mixing
        aux: index, lambda, mask, saliency maps
    """
    B = flow.size(0)
    device = flow.device

    if same_class:
        index = build_same_class_index(labels)
    else:
        index = torch.randperm(B, device=device)

    beta = torch.distributions.Beta(alpha, alpha)
    lam = beta.sample((B,)).to(device=device, dtype=flow.dtype).view(B, 1, 1, 1)

    mask, mask_aux = motion_saliency_mask(flow, q=q, temperature=temperature)

    flow_j = flow[index]
    mixed_core = lam * flow + (1.0 - lam) * flow_j
    mixed_flow = mask * mixed_core + (1.0 - mask) * flow

    aux = {
        "mix_index": index,
        "lambda": lam,
        "mask": mask,
        **mask_aux,
    }
    return mixed_flow, labels, aux
