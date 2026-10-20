from __future__ import annotations

import math
from typing import Tuple, Dict

import torch
import torch.nn.functional as F


def row_normalize(A: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """
    Row-normalize structural prior matrix.

    Args:
        A: [B, N, N] or [N, N]
    """
    if A.dim() == 2:
        A = A.unsqueeze(0)
    denom = A.sum(dim=-1, keepdim=True).clamp_min(eps)
    return A / denom


def ensure_batched_graph(A: torch.Tensor, batch_size: int, device=None, dtype=None) -> torch.Tensor:
    """
    Convert [N,N] or [1,N,N] graph to [B,N,N].
    """
    if device is None:
        device = A.device
    if dtype is None:
        dtype = A.dtype

    A = A.to(device=device, dtype=dtype)

    if A.dim() == 2:
        A = A.unsqueeze(0).expand(batch_size, -1, -1)
    elif A.dim() == 3 and A.size(0) == 1:
        A = A.expand(batch_size, -1, -1)
    elif A.dim() == 3 and A.size(0) == batch_size:
        pass
    else:
        raise ValueError(f"Invalid graph shape: {tuple(A.shape)}, batch={batch_size}")

    return A


def infer_token_grid(num_tokens: int) -> Tuple[int, int]:
    """
    Infer token grid from N.
    Assumes square patch grid when possible.
    """
    root = int(math.sqrt(num_tokens))
    if root * root == num_tokens:
        return root, root

    # fallback: approximate rectangular grid
    for h in range(root, 0, -1):
        if num_tokens % h == 0:
            return h, num_tokens // h

    raise ValueError(f"Cannot infer token grid from num_tokens={num_tokens}")


def patch_motion_descriptor(
    flow: torch.Tensor,
    num_tokens: int,
    eps: float = 1e-6,
) -> torch.Tensor:
    """
    Extract sample-specific optical-flow descriptors for patch tokens.

    Args:
        flow: [B, 3, H, W], channels usually [u, v, strain]
        num_tokens: N, must match structural matrix size

    Returns:
        desc: [B, N, 4], including mean u, mean v, mean strain, mean magnitude.
    """
    if flow.dim() != 4:
        raise ValueError(f"flow must be [B,C,H,W], got {tuple(flow.shape)}")
    if flow.size(1) < 2:
        raise ValueError(f"flow must have at least u/v channels, got C={flow.size(1)}")

    B, C, H, W = flow.shape
    gh, gw = infer_token_grid(num_tokens)

    u = flow[:, 0:1]
    v = flow[:, 1:2]
    if C >= 3:
        strain = flow[:, 2:3]
    else:
        strain = torch.sqrt(u ** 2 + v ** 2 + eps)

    mag = torch.sqrt(u ** 2 + v ** 2 + eps)

    feats = torch.cat([u, v, strain, mag], dim=1)
    pooled = F.adaptive_avg_pool2d(feats, output_size=(gh, gw))  # [B,4,gh,gw]
    desc = pooled.flatten(2).transpose(1, 2).contiguous()        # [B,N,4]

    # Per-sample normalization stabilizes distance scale.
    mean = desc.mean(dim=1, keepdim=True)
    std = desc.std(dim=1, keepdim=True).clamp_min(eps)
    desc = (desc - mean) / std
    return desc


def motion_affinity(
    desc: torch.Tensor,
    base_graph: torch.Tensor,
    tau: float = 0.7,
    eps: float = 1e-6,
) -> torch.Tensor:
    """
    Compute Dg(n,m) = Bg(n,m) * exp(-||dn-dm||^2 / tau).

    Args:
        desc: [B,N,D]
        base_graph: [B,N,N] or [N,N]
    """
    if desc.dim() != 3:
        raise ValueError(f"desc must be [B,N,D], got {tuple(desc.shape)}")

    B, N, _ = desc.shape
    base_graph = ensure_batched_graph(base_graph, B, device=desc.device, dtype=desc.dtype)

    if base_graph.shape[-2:] != (N, N):
        raise ValueError(f"base_graph shape {tuple(base_graph.shape)} does not match desc N={N}")

    dist2 = torch.cdist(desc, desc, p=2) ** 2
    dyn = base_graph * torch.exp(-dist2 / max(tau, eps))
    return row_normalize(dyn, eps=eps)


def calibrate_prior(
    base_graph: torch.Tensor,
    dyn_affinity: torch.Tensor,
    rho: float = 0.5,
    eps: float = 1e-6,
) -> torch.Tensor:
    """
    Ag = RowNorm((1-rho)Bg + rho Dg).
    """
    rho = float(rho)
    if not 0.0 <= rho <= 1.0:
        raise ValueError(f"rho must be in [0,1], got {rho}")

    B = dyn_affinity.size(0)
    base_graph = ensure_batched_graph(base_graph, B, device=dyn_affinity.device, dtype=dyn_affinity.dtype)
    A = (1.0 - rho) * base_graph + rho * dyn_affinity
    return row_normalize(A, eps=eps)


def prior_motion_energy(desc: torch.Tensor, A: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """
    Compute a compact reliability scalar for a structural prior.

    Args:
        desc: [B,N,4], last descriptor dimension includes normalized magnitude at index 3.
        A: [B,N,N]

    Returns:
        energy: [B,1]
    """
    B, N, _ = desc.shape
    A = ensure_batched_graph(A, B, device=desc.device, dtype=desc.dtype)
    mag = desc[..., 3].abs().unsqueeze(-1)  # [B,N,1]
    weighted = torch.bmm(A, mag)            # [B,N,1]
    return weighted.mean(dim=1)             # [B,1]


def build_motion_calibrated_priors(
    flow: torch.Tensor,
    A_com_base: torch.Tensor,
    A_reg_base: torch.Tensor,
    rho_com: float = 0.5,
    rho_reg: float = 0.5,
    tau: float = 0.7,
) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, torch.Tensor]]:
    """
    Construct motion-calibrated component-level and region-level priors.

    Args:
        flow: [B,3,H,W]
        A_com_base: [B,N,N] or [N,N]
        A_reg_base: [B,N,N] or [N,N]

    Returns:
        A_com: [B,N,N]
        A_reg: [B,N,N]
        aux: descriptors, dynamic affinities, and energy scalars.
    """
    if A_com_base.dim() == 2:
        N = A_com_base.size(0)
    else:
        N = A_com_base.size(-1)

    desc = patch_motion_descriptor(flow, num_tokens=N)

    A_com_base = ensure_batched_graph(A_com_base, flow.size(0), device=flow.device, dtype=flow.dtype)
    A_reg_base = ensure_batched_graph(A_reg_base, flow.size(0), device=flow.device, dtype=flow.dtype)

    D_com = motion_affinity(desc, A_com_base, tau=tau)
    D_reg = motion_affinity(desc, A_reg_base, tau=tau)

    A_com = calibrate_prior(A_com_base, D_com, rho=rho_com)
    A_reg = calibrate_prior(A_reg_base, D_reg, rho=rho_reg)

    e_com = prior_motion_energy(desc, A_com)
    e_reg = prior_motion_energy(desc, A_reg)

    aux = {
        "motion_desc": desc,
        "D_com": D_com,
        "D_reg": D_reg,
        "e_com": e_com,
        "e_reg": e_reg,
    }
    return A_com, A_reg, aux
