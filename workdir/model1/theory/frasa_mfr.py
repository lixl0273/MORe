from __future__ import annotations

import torch
import torch.nn as nn

from model1.theory.motion_calibrated_prior import row_normalize


def membership_or_graph_to_graph(A: torch.Tensor) -> torch.Tensor:
    """
    Convert structural matrix to token-token graph.

    Supported:
        [B, M, N] membership matrix, e.g. component/region assignment
        [B, N, N] token-token graph
        [M, N]
        [N, N]
    """
    if A is None:
        return None

    if A.dim() == 2:
        A = A.unsqueeze(0)

    if A.dim() != 3:
        raise ValueError(f"Expected A to be [B,M,N] or [B,N,N], got {tuple(A.shape)}")

    # token-token graph already
    if A.size(-2) == A.size(-1):
        return row_normalize(A)

    # membership matrix [B,M,N] -> graph [B,N,N]
    G = torch.matmul(A.transpose(1, 2), A)
    return row_normalize(G)


class FRASA(nn.Module):
    """
    Motion-calibrated multi-granularity FRASA.

    Attention score:
        S = QK^T + gamma_com QK_com^T + gamma_reg QK_reg^T

    gamma_com and gamma_reg are independent sigmoid gates.
    """

    def __init__(
        self,
        dim,
        num_heads=8,
        qkv_bias=False,
        qk_scale=None,
        attn_drop=0.,
        proj_drop=0.,
        M_R=4,
        N=196,
    ):
        super().__init__()

        self.num_heads = num_heads
        self.dim = dim
        self.head_dim = dim // num_heads
        self.scale = qk_scale or self.head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.w_c = nn.Linear(dim, dim)
        self.w_r = nn.Linear(dim, dim)

        self.gate_com = nn.Sequential(
            nn.Linear(dim * 3 + 2, max(dim // 4, 64)),
            nn.ReLU(inplace=True),
            nn.Linear(max(dim // 4, 64), 1),
        )
        self.gate_reg = nn.Sequential(
            nn.Linear(dim * 3 + 2, max(dim // 4, 64)),
            nn.ReLU(inplace=True),
            nn.Linear(max(dim // 4, 64), 1),
        )

        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        self.gate_type = "sigmoid"
        self.prior_type = "nonparam"  # static | nonparam | learned_relation
        self.rho_com = 0.5
        self.rho_reg = 0.5

        # Learned relation predictor psi_g.
        # We project high-dimensional motion tokens to a compact descriptor
        # before building pairwise [d_i, d_j, |d_i-d_j|, d_i*d_j].
        self.rel_dim = 16
        self.rel_proj = nn.Linear(dim, self.rel_dim)
        self.rel_mlp_com = nn.Sequential(
            nn.Linear(self.rel_dim * 4, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 1),
        )
        self.rel_mlp_reg = nn.Sequential(
            nn.Linear(self.rel_dim * 4, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 1),
        )
        self.last_aux = {}

    def _reshape_heads(self, x: torch.Tensor) -> torch.Tensor:
        B, N, C = x.shape
        return x.reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)

    def _learned_relation_prior(self, x: torch.Tensor, A_base: torch.Tensor, mlp: nn.Module, rho: float):
        """
        Learned relation prior:
            R(n,m) = sigmoid(psi([d_n, d_m, |d_n-d_m|, d_n*d_m]))
            A = RowNorm((1-rho)B + rho * (B * R))

        x is the motion token representation [B,N,C].
        A_base is a static token-token support [B,N,N] or [1,N,N].
        """
        B, N, _ = x.shape

        A_base = membership_or_graph_to_graph(A_base)
        if A_base is None:
            A_base = torch.eye(N, device=x.device, dtype=x.dtype).unsqueeze(0).expand(B, -1, -1)

        A_base = A_base.to(device=x.device, dtype=x.dtype)
        if A_base.size(0) == 1 and B > 1:
            A_base = A_base.expand(B, -1, -1)

        z = self.rel_proj(x)  # [B,N,R]
        zi = z.unsqueeze(2).expand(-1, -1, N, -1)
        zj = z.unsqueeze(1).expand(-1, N, -1, -1)

        pair = torch.cat([zi, zj, torch.abs(zi - zj), zi * zj], dim=-1)
        rel = torch.sigmoid(mlp(pair).squeeze(-1))  # [B,N,N]

        D = A_base * rel
        A = row_normalize((1.0 - rho) * A_base + rho * D)
        return A, D

    def forward(self, x, A_Component=None, A_Region=None, e_com=None, e_reg=None):
        B, N, C = x.shape

        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        A_com = membership_or_graph_to_graph(A_Component)
        A_reg = membership_or_graph_to_graph(A_Region)

        if A_com is None:
            A_com = torch.eye(N, device=x.device, dtype=x.dtype).unsqueeze(0).expand(B, -1, -1)
        if A_reg is None:
            A_reg = torch.eye(N, device=x.device, dtype=x.dtype).unsqueeze(0).expand(B, -1, -1)

        A_com = A_com.to(device=x.device, dtype=x.dtype)
        A_reg = A_reg.to(device=x.device, dtype=x.dtype)

        prior_type = getattr(self, "prior_type", "nonparam")
        D_com = None
        D_reg = None

        if prior_type == "learned_relation":
            A_com, D_com = self._learned_relation_prior(
                x,
                A_com,
                self.rel_mlp_com,
                rho=float(getattr(self, "rho_com", 0.5)),
            )
            A_reg, D_reg = self._learned_relation_prior(
                x,
                A_reg,
                self.rel_mlp_reg,
                rho=float(getattr(self, "rho_reg", 0.5)),
            )
        elif prior_type == "static":
            A_com = row_normalize(A_com)
            A_reg = row_normalize(A_reg)
        elif prior_type == "nonparam":
            # A_com/A_reg are already calibrated upstream.
            A_com = row_normalize(A_com)
            A_reg = row_normalize(A_reg)
        else:
            raise ValueError(f"Unsupported prior_type: {prior_type}")

        # Structural contexts: Xcom = Acom X, Xreg = Areg X
        X_com = torch.bmm(A_com, x)
        X_reg = torch.bmm(A_reg, x)

        K_com = self._reshape_heads(self.w_c(X_com))
        K_reg = self._reshape_heads(self.w_r(X_reg))

        if e_com is None:
            e_com = torch.zeros(B, 1, device=x.device, dtype=x.dtype)
        if e_reg is None:
            e_reg = torch.zeros(B, 1, device=x.device, dtype=x.dtype)

        if e_com.dim() == 1:
            e_com = e_com.view(B, 1)
        if e_reg.dim() == 1:
            e_reg = e_reg.view(B, 1)

        # Reliability descriptor r = [GAP(X), GAP(Xcom), GAP(Xreg), ecom, ereg]
        r = torch.cat(
            [
                x.mean(dim=1),
                X_com.mean(dim=1),
                X_reg.mean(dim=1),
                e_com.to(device=x.device, dtype=x.dtype),
                e_reg.to(device=x.device, dtype=x.dtype),
            ],
            dim=-1,
        )

        gate_logit_com = self.gate_com(r)  # [B,1]
        gate_logit_reg = self.gate_reg(r)  # [B,1]

        gate_type = getattr(self, "gate_type", "sigmoid")
        if gate_type == "softmax":
            gamma_pair = torch.softmax(torch.cat([gate_logit_com, gate_logit_reg], dim=-1), dim=-1)
            gamma_com = gamma_pair[:, 0:1]
            gamma_reg = gamma_pair[:, 1:2]
        elif gate_type == "sigmoid":
            gamma_com = torch.sigmoid(gate_logit_com)
            gamma_reg = torch.sigmoid(gate_logit_reg)
        else:
            raise ValueError(f"Unsupported gate_type: {gate_type}")

        score0 = (q @ k.transpose(-2, -1)) * self.scale
        score_com = (q @ K_com.transpose(-2, -1)) * self.scale
        score_reg = (q @ K_reg.transpose(-2, -1)) * self.scale

        score = (
            score0
            + gamma_com.view(B, 1, 1, 1) * score_com
            + gamma_reg.view(B, 1, 1, 1) * score_reg
        )

        P0 = score0.softmax(dim=-1)
        Pcom = score_com.softmax(dim=-1)
        Preg = score_reg.softmax(dim=-1)

        attn = score.softmax(dim=-1)
        attn = self.attn_drop(attn)

        out = (attn @ v).transpose(1, 2).reshape(B, N, C)
        out = self.proj(out)
        out = self.proj_drop(out)

        self.last_aux = {
            "gamma_com": gamma_com,
            "gamma_reg": gamma_reg,
            "P0": P0,
            "Pcom": Pcom,
            "Preg": Preg,
            "prior_type": prior_type,
            "D_com": D_com,
            "D_reg": D_reg,
        }

        return out
