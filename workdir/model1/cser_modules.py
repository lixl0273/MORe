from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def msmer_token_classifier(base_model: nn.Module, tokens: torch.Tensor) -> torch.Tensor:
    """Reuse the original MSMER classifier path.

    Original MyNet_FRAT uses:
        token_feat [B,N,D] -> [B,D,H,W] -> conv1x1_pc -> flatten -> fc.
    CSER variants must reuse this path so E1-E5 differ only in the token representation.
    """
    if not hasattr(base_model, "conv1x1_pc") or not hasattr(base_model, "fc"):
        raise RuntimeError("Original MSMER classifier path conv1x1_pc/fc was not found.")

    if tokens.dim() != 3:
        raise RuntimeError(f"Expected tokens [B,N,D], got shape={tuple(tokens.shape)}")

    bsz, num_tokens, dim = tokens.shape
    feature_size = int(getattr(base_model, "feature_size", round(num_tokens ** 0.5)))
    if feature_size * feature_size != num_tokens:
        raise RuntimeError(
            f"Token count mismatch: N={num_tokens}, expected feature_size^2={feature_size ** 2}."
        )

    feat_map = tokens.transpose(1, 2).contiguous().view(bsz, dim, feature_size, feature_size)
    feat_map = base_model.conv1x1_pc(feat_map)
    feat_flat = feat_map.flatten(1)
    return base_model.fc(feat_flat)


class CSERTokenReasoner(nn.Module):
    """Causal Structural Evidence Reasoning over MSMER token features.

    Inputs:
        tokens: [B, N, D]
        A_Component: [B, 8, N] or [B, 1, 8, N]
        A_Region: [B, 4, N] or [B, 1, 4, N]

    The module builds a token graph from facial component/region memberships, mixes it
    with a learnable graph, performs one graph-message-passing step, and returns a
    token-level structural evidence representation.
    """

    def __init__(
        self,
        embed_dim: int,
        num_tokens: int = 196,
        graph_type: str = "soft",
        prior_beta: float = 0.7,
        use_reliability: bool = False,
        reliability_mode: str = "naive",
        reliability_gamma: float = 0.5,
        reliability_min: float = 0.5,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.embed_dim = int(embed_dim)
        self.num_tokens = int(num_tokens)
        self.graph_type = str(graph_type)
        self.prior_beta = float(prior_beta)
        self.use_reliability = bool(use_reliability)
        self.reliability_mode = str(reliability_mode)
        self.reliability_gamma = float(reliability_gamma)
        self.reliability_min = float(reliability_min)

        init_graph = torch.eye(self.num_tokens) * 2.0
        self.learn_adj_logits = nn.Parameter(init_graph)

        hidden = max(64, self.embed_dim // 4)
        self.pre_norm = nn.LayerNorm(self.embed_dim)
        self.msg_proj = nn.Linear(self.embed_dim, self.embed_dim)
        self.out_proj = nn.Linear(self.embed_dim, self.embed_dim)
        self.dropout = nn.Dropout(dropout)
        self.act = nn.GELU()
        self.post_norm = nn.LayerNorm(self.embed_dim)
        self.rel_mlp = nn.Sequential(
            nn.LayerNorm(self.embed_dim),
            nn.Linear(self.embed_dim, hidden),
            nn.GELU(),
            nn.Linear(hidden, 1),
        )
        self.score_mlp = nn.Sequential(
            nn.LayerNorm(self.embed_dim),
            nn.Linear(self.embed_dim, hidden),
            nn.GELU(),
            nn.Linear(hidden, 1),
        )

    @staticmethod
    def _normalize_membership(A, batch_size: int, num_tokens: int, device, dtype):
        if A is None or not torch.is_tensor(A):
            return None
        A = A.to(device=device, dtype=dtype)
        if A.dim() == 4 and A.size(1) == 1:
            A = A.squeeze(1)
        if A.dim() != 3:
            return None
        if A.size(0) != batch_size:
            if A.size(0) == 1:
                A = A.expand(batch_size, -1, -1)
            else:
                return None
        if A.size(-1) != num_tokens:
            return None
        A = torch.nan_to_num(A, nan=0.0, posinf=0.0, neginf=0.0).clamp_min(0.0)
        return A / A.sum(dim=-1, keepdim=True).clamp_min(1e-6)

    def _prior_graph(self, A_Component, A_Region, batch_size: int, num_tokens: int, device, dtype):
        memberships = []
        A_component = self._normalize_membership(A_Component, batch_size, num_tokens, device, dtype)
        A_region = self._normalize_membership(A_Region, batch_size, num_tokens, device, dtype)
        if A_component is not None:
            memberships.append(A_component)
        if A_region is not None:
            memberships.append(A_region)

        eye = torch.eye(num_tokens, device=device, dtype=dtype).unsqueeze(0)
        if not memberships:
            return eye.expand(batch_size, -1, -1)

        M = torch.cat(memberships, dim=1)  # [B,K,N]
        G = torch.bmm(M.transpose(1, 2), M) + eye
        G = G.clamp_min(0.0)
        return G / G.sum(dim=-1, keepdim=True).clamp_min(1e-6)

    def _learn_graph(self, batch_size: int, num_tokens: int, device, dtype):
        logits = self.learn_adj_logits[:num_tokens, :num_tokens].to(device=device, dtype=dtype)
        logits = 0.5 * (logits + logits.transpose(0, 1))
        G = torch.softmax(logits, dim=-1)
        return G.unsqueeze(0).expand(batch_size, -1, -1)

    def _graph(self, tokens: torch.Tensor, A_Component=None, A_Region=None):
        batch_size, num_tokens, _ = tokens.shape
        G_prior = self._prior_graph(A_Component, A_Region, batch_size, num_tokens, tokens.device, tokens.dtype)
        G_learn = self._learn_graph(batch_size, num_tokens, tokens.device, tokens.dtype)

        if self.graph_type == "prior":
            G = G_prior
        elif self.graph_type == "learn":
            G = G_learn
        else:
            beta = max(0.0, min(1.0, self.prior_beta))
            G = beta * G_prior + (1.0 - beta) * G_learn
            G = G / G.sum(dim=-1, keepdim=True).clamp_min(1e-6)
        return G

    def forward(self, tokens: torch.Tensor, A_Component=None, A_Region=None):
        x = self.pre_norm(tokens)
        G = self._graph(x, A_Component, A_Region)

        msg = torch.bmm(G, x)
        delta = self.msg_proj(msg)
        delta = self.act(delta)
        delta = self.dropout(delta)
        delta = self.out_proj(delta)
        delta = self.post_norm(delta)

        reliability = torch.sigmoid(self.rel_mlp(x))
        if self.use_reliability:
            if self.reliability_mode == "residual":
                delta = delta * (1.0 + self.reliability_gamma * reliability)
            elif self.reliability_mode == "bounded":
                reliability_bounded = self.reliability_min + (1.0 - self.reliability_min) * reliability
                delta = delta * reliability_bounded
            else:
                # Naive reliability is intentionally kept as an ablation setting.
                delta = delta * reliability

        evidence = torch.sigmoid(self.score_mlp(delta))
        aux = {
            "graph": G,
            "reliability": reliability,
            "evidence": evidence,
        }
        return delta, aux


class MSMERWithCSER(nn.Module):
    """Minimal wrapper around original MyNet_FRAT.

    The wrapper does not replace MPT/MMTE, FRAT/MSPA, or the original classifier.
    It only modifies the FRAT token representation immediately before the original
    conv1x1_pc + fc head.
    """

    def __init__(
        self,
        base_model: nn.Module,
        embed_dim: int,
        num_tokens: int,
        num_classes: int,
        mode: str = "residual",
        graph_type: str = "soft",
        prior_beta: float = 0.7,
        residual_alpha: float = 0.1,
        use_reliability: bool = False,
        reliability_mode: str = "naive",
        reliability_gamma: float = 0.5,
        reliability_min: float = 0.5,
        relative_update_bound: float = 0.0,
        use_counterfactual: bool = False,
        cf_margin: float = 0.02,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.base_model = base_model
        self.vit_pos = base_model.vit_pos
        self.mode = str(mode)
        self.residual_alpha = float(residual_alpha)
        self.relative_update_bound = float(relative_update_bound)
        self.use_counterfactual = bool(use_counterfactual)
        self.cf_margin = float(cf_margin)
        self.cser_aux_loss = None

        self.cser = CSERTokenReasoner(
            embed_dim=embed_dim,
            num_tokens=num_tokens,
            graph_type=graph_type,
            prior_beta=prior_beta,
            use_reliability=use_reliability,
            reliability_mode=reliability_mode,
            reliability_gamma=reliability_gamma,
            reliability_min=reliability_min,
            dropout=dropout,
        )

    def _extract_tokens(self, flow, A_Component=None, A_Region=None):
        if bool(getattr(self.base_model, "use_mpt", False)):
            flow = self.base_model.mpt(flow)
        out = self.base_model.vit_pos(flow, A_Component, A_Region, return_aux=True)
        if isinstance(out, tuple):
            return out
        return out, {}

    def _compose(self, tokens: torch.Tensor, delta: torch.Tensor) -> torch.Tensor:
        if self.mode == 'direct':
            return delta

        update = self.residual_alpha * delta
        if self.relative_update_bound > 0:
            token_norm = tokens.norm(dim=-1, keepdim=True).clamp_min(1e-6)
            update_norm = update.norm(dim=-1, keepdim=True)
            ratio = update_norm / token_norm
            scale = torch.clamp(
                self.relative_update_bound / (ratio + 1e-12),
                max=1.0,
            )
            update = update * scale

        return tokens + update

    def forward(self, flow, A_Component=None, A_Region=None, return_aux: bool = False):
        self.cser_aux_loss = None
        base_tokens, base_aux = self._extract_tokens(flow, A_Component, A_Region)
        delta, cser_aux = self.cser(base_tokens, A_Component, A_Region)
        out_tokens = self._compose(base_tokens, delta)
        logits = msmer_token_classifier(self.base_model, out_tokens)

        if self.training and self.use_counterfactual:
            evidence = cser_aux["evidence"].detach()
            cf_tokens = base_tokens * (1.0 - evidence)
            cf_delta, _ = self.cser(cf_tokens, A_Component, A_Region)
            cf_logits = msmer_token_classifier(self.base_model, self._compose(cf_tokens, cf_delta))
            target = logits.detach().argmax(dim=1, keepdim=True)
            p = F.softmax(logits, dim=1).gather(1, target)
            p_cf = F.softmax(cf_logits, dim=1).gather(1, target)
            self.cser_aux_loss = F.relu(p_cf - p + self.cf_margin).mean()

        if return_aux:
            aux = dict(base_aux) if isinstance(base_aux, dict) else {}
            aux["cser"] = cser_aux
            if self.cser_aux_loss is not None:
                aux["cser_aux_loss"] = self.cser_aux_loss
            aux["logits"] = logits
            return logits, aux
        return logits
