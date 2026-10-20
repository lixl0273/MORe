from __future__ import annotations

from typing import Dict, Iterable

import torch
import torch.nn.functional as F


def entropy_from_logits(logits: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    p = F.softmax(logits, dim=-1).clamp_min(eps)
    return -(p * torch.log(p)).sum(dim=-1)


def kl_probs(p: torch.Tensor, q: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    p = p.clamp_min(eps)
    q = q.clamp_min(eps)
    return (p * (torch.log(p) - torch.log(q))).sum(dim=-1)


def js_divergence_from_logits(
    logits_a: torch.Tensor,
    logits_b: torch.Tensor,
    eps: float = 1e-8,
) -> torch.Tensor:
    p = F.softmax(logits_a, dim=-1).clamp_min(eps)
    q = F.softmax(logits_b, dim=-1).clamp_min(eps)
    m = 0.5 * (p + q)
    return 0.5 * kl_probs(p, m, eps=eps) + 0.5 * kl_probs(q, m, eps=eps)


def attention_kl(Pg: torch.Tensor, P0: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """
    KL(Pg || P0), averaged over batch/head/token dimensions.

    Accepts:
        [B,H,N,N]
        [B,N,N]
    """
    if Pg.dim() == 3:
        Pg = Pg.unsqueeze(1)
    if P0.dim() == 3:
        P0 = P0.unsqueeze(1)

    Pg = Pg.clamp_min(eps)
    P0 = P0.clamp_min(eps)
    kl = Pg * (torch.log(Pg) - torch.log(P0))
    return kl.sum(dim=-1).mean()


def structural_compression_loss(
    P0_list: Iterable[torch.Tensor],
    Pcom_list: Iterable[torch.Tensor],
    Preg_list: Iterable[torch.Tensor],
) -> torch.Tensor:
    losses = []
    for P0, Pcom, Preg in zip(P0_list, Pcom_list, Preg_list):
        losses.append(attention_kl(Pcom, P0) + attention_kl(Preg, P0))

    if not losses:
        raise ValueError("Empty attention list for structural compression loss.")

    return torch.stack(losses).mean()


def structural_information_gain_loss(
    logits_structural: torch.Tensor,
    logits_standard: torch.Tensor,
) -> torch.Tensor:
    """
    Lgain = H(pA) - H(p0).
    Minimizing Lgain encourages structural branch to reduce entropy.
    """
    return (entropy_from_logits(logits_structural) - entropy_from_logits(logits_standard)).mean()


def prediction_consistency_loss(
    logits: torch.Tensor,
    logits_enhanced: torch.Tensor,
) -> torch.Tensor:
    return js_divergence_from_logits(logits, logits_enhanced).mean()


def compute_full_information_loss(
    ce_loss: torch.Tensor,
    enhanced_loss: torch.Tensor | None,
    aux: Dict,
    aux_enhanced: Dict | None,
    alpha: float = 1.0,
    beta: float = 0.05,
    mu: float = 0.05,
    nu: float = 0.1,
) -> Dict[str, torch.Tensor]:
    """
    Compose final information-theoretic objective.

    Expected aux keys for full mode:
        logits_standard
        logits_structural
        P0_list
        Pcom_list
        Preg_list

    Expected aux_enhanced:
        logits
    """
    device = ce_loss.device
    zero = torch.zeros([], device=device, dtype=ce_loss.dtype)

    L_enh = enhanced_loss if enhanced_loss is not None else zero

    if "logits_standard" in aux and "logits_structural" in aux:
        L_gain = structural_information_gain_loss(
            aux["logits_structural"],
            aux["logits_standard"],
        )
    else:
        L_gain = zero

    if all(k in aux for k in ["P0_list", "Pcom_list", "Preg_list"]):
        L_comp = structural_compression_loss(
            aux["P0_list"],
            aux["Pcom_list"],
            aux["Preg_list"],
        )
    else:
        L_comp = zero

    if aux_enhanced is not None and "logits" in aux and "logits" in aux_enhanced:
        L_cons = prediction_consistency_loss(aux["logits"], aux_enhanced["logits"])
    else:
        L_cons = zero

    total = ce_loss + alpha * L_enh + beta * L_comp + mu * L_gain + nu * L_cons

    return {
        "loss": total,
        "Lce": ce_loss.detach(),
        "Lenh": L_enh.detach(),
        "Lcomp": L_comp.detach(),
        "Lgain": L_gain.detach(),
        "Lcons": L_cons.detach(),
    }
