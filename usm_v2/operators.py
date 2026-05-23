"""
USM v2 — Compositional Operator and Relation Maps.

Relation-typed operations in tangent space at the origin, parameterized by
the learnable curvature.
"""

import torch
import torch.nn as nn
from typing import Optional

from .manifold import (
    LearnablePoincareBall, logmap0, expmap0, clamp_to_ball, EPS
)


RELATIONS = ("IS_A", "CAUSES", "PART_OF", "SIMILAR_TO", "ANTONYM", "CAPABLE_OF")
REL2IDX = {r: i for i, r in enumerate(RELATIONS)}
N_RELATIONS = len(RELATIONS)

_SYMW = {
    "IS_A": 0.0, "CAUSES": 0.0, "PART_OF": 0.0,
    "SIMILAR_TO": 1.0, "ANTONYM": 0.1, "CAPABLE_OF": 0.0,
}
SYM_WEIGHTS = torch.tensor([_SYMW[r] for r in RELATIONS], dtype=torch.float32)


class CompositionalOperator(nn.Module):
    """
    Typed relational composition: z_A ⊕_r z_B.

    Operates in tangent space at origin (via logmap0), applies a bilinear
    relation-specific transform, then maps back to the ball (via expmap0).
    Symmetry weights blend U toward W for symmetric relations.
    """

    def __init__(self, manifold: LearnablePoincareBall, d: int = 1024,
                 n_rel: int = N_RELATIONS, hyperbolic: bool = True):
        super().__init__()
        self.d = d
        self.hyperbolic = hyperbolic
        self.manifold = manifold

        init = lambda: (
            torch.eye(d).unsqueeze(0).repeat(n_rel, 1, 1)
            + 0.01 * torch.randn(n_rel, d, d)
        )
        self.W = nn.Parameter(init())
        self.U_raw = nn.Parameter(init())
        self.V = nn.Parameter(0.01 * torch.randn(n_rel, d, d))
        self.b = nn.Parameter(torch.zeros(n_rel, d))
        self.register_buffer("sym_w", SYM_WEIGHTS)

    def effective_U(self):
        s = self.sym_w.view(-1, 1, 1)
        return s * self.W + (1.0 - s) * self.U_raw

    def forward(self, z_A: torch.Tensor, rel_idx: torch.Tensor,
                z_B: torch.Tensor, c: Optional[torch.Tensor] = None) -> torch.Tensor:
        if c is None:
            c = self.manifold.c

        if self.hyperbolic and c.item() > EPS:
            a, b = logmap0(z_A, c), logmap0(z_B, c)
        else:
            a, b = z_A, z_B

        U_eff = self.effective_U()
        W_r = self.W[rel_idx]
        U_r = U_eff[rel_idx]
        V_r = self.V[rel_idx]
        b_r = self.b[rel_idx]

        out = torch.tanh(
            torch.einsum("bij,bj->bi", W_r, a)
            + torch.einsum("bij,bj->bi", U_r, b)
            + torch.einsum("bij,bj->bi", V_r, a * b)
            + b_r
        )

        if self.hyperbolic and c.item() > EPS:
            return clamp_to_ball(expmap0(out, c), c)
        return out


class RelationMaps(nn.Module):
    """
    TransE-style relation translation: z_h --r--> z_pred.

    Per-relation linear map in tangent space.
    """

    def __init__(self, manifold: LearnablePoincareBall, d: int = 1024,
                 n_rel: int = N_RELATIONS, hyperbolic: bool = True):
        super().__init__()
        self.hyperbolic = hyperbolic
        self.manifold = manifold
        self.W = nn.Parameter(
            torch.eye(d).unsqueeze(0).repeat(n_rel, 1, 1)
            + 0.01 * torch.randn(n_rel, d, d)
        )
        self.b = nn.Parameter(torch.zeros(n_rel, d))

    def forward(self, z_h: torch.Tensor, rel_idx: torch.Tensor,
                c: Optional[torch.Tensor] = None) -> torch.Tensor:
        if c is None:
            c = self.manifold.c

        if self.hyperbolic and c.item() > EPS:
            h = logmap0(z_h, c)
        else:
            h = z_h

        W_r = self.W[rel_idx]
        b_r = self.b[rel_idx]
        out = torch.einsum("bij,bj->bi", W_r, h) + b_r

        if self.hyperbolic and c.item() > EPS:
            return clamp_to_ball(expmap0(out, c), c)
        return out
