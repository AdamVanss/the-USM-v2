"""
USM v2 — Loss functions parameterized by curvature.

All losses accept an explicit curvature tensor `c` so they work correctly
during burn-in (c_eff ≈ 0 → Euclidean) and full hyperbolic training.
"""

import torch
import torch.nn.functional as F
from typing import Optional

from .manifold import (
    poincare_dist, poincare_cdist, dist0, logmap0,
    eucl_dist, eucl_cdist, EPS
)


def _is_hyperbolic(c: torch.Tensor) -> bool:
    return c.item() > EPS


def loss_cl(z_src: torch.Tensor, z_tgt: torch.Tensor, c: torch.Tensor,
            tau: float = 1.0) -> torch.Tensor:
    """Symmetric InfoNCE with negative geodesic (or Euclidean) distance as similarity."""
    B = z_src.shape[0]
    if B == 0:
        return z_src.new_zeros(())

    cdist_fn = poincare_cdist if _is_hyperbolic(c) else eucl_cdist
    if _is_hyperbolic(c):
        dists = cdist_fn(z_src, z_tgt, c)
    else:
        dists = cdist_fn(z_src, z_tgt)

    sim = -dists / tau
    labels = torch.arange(B, device=z_src.device)
    return 0.5 * (F.cross_entropy(sim, labels) + F.cross_entropy(sim.T, labels))


def loss_rel(z_pred: torch.Tensor, z_t: torch.Tensor, z_t_neg: torch.Tensor,
             c: torch.Tensor, margin: float = 2.0) -> torch.Tensor:
    """TransE-style margin loss."""
    if _is_hyperbolic(c):
        d_pos = poincare_dist(z_pred, z_t, c)
        d_neg = poincare_dist(z_pred, z_t_neg, c)
    else:
        d_pos = eucl_dist(z_pred, z_t)
        d_neg = eucl_dist(z_pred, z_t_neg)
    return F.relu(margin - d_neg + d_pos).mean()


def loss_entailment(z_A: torch.Tensor, z_B: torch.Tensor, labels: torch.Tensor,
                    c: torch.Tensor, delta_c: float = 2.0) -> torch.Tensor:
    """
    Partial order + contradiction margin loss from SNLI.
    Labels: 0 = entailment, 1 = neutral (ignored), 2 = contradiction.
    """
    total, n = z_A.new_zeros(()), 0

    ent_mask = (labels == 0)
    contra_mask = (labels == 2)

    if _is_hyperbolic(c):
        lm = lambda x: logmap0(x, c)
        dist_fn = lambda a, b: poincare_dist(a, b, c)
    else:
        lm = lambda x: x
        dist_fn = eucl_dist

    if ent_mask.any():
        a_e, b_e = lm(z_A[ent_mask]), lm(z_B[ent_mask])
        total = total + F.relu(b_e - a_e).pow(2).sum(-1).mean()
        n += 1

    if contra_mask.any():
        d_c = dist_fn(z_A[contra_mask], z_B[contra_mask])
        total = total + F.relu(delta_c - d_c).pow(2).mean()
        n += 1

    return total / max(n, 1)


def loss_crossmodal(z_text: torch.Tensor, z_image: torch.Tensor,
                    c: torch.Tensor, tau: float = 1.0) -> torch.Tensor:
    """Cross-modal InfoNCE: pull text-image pairs together."""
    return loss_cl(z_text, z_image, c, tau=tau)


def loss_hierarchy(z_fine: torch.Tensor, z_coarse: torch.Tensor,
                   c: torch.Tensor, margin: float = 1.0) -> torch.Tensor:
    """
    Fine-class embeddings should be deeper on the ball (higher depth) than coarse.
    Combines margin ranking + entailment proximity penalty.
    """
    if _is_hyperbolic(c):
        depth_fine = dist0(z_fine, c)
        depth_coarse = dist0(z_coarse, c)
    else:
        depth_fine = z_fine.norm(dim=-1)
        depth_coarse = z_coarse.norm(dim=-1)

    L_rank = F.margin_ranking_loss(
        depth_fine, depth_coarse,
        target=torch.ones_like(depth_fine),
        margin=margin,
    )

    if _is_hyperbolic(c):
        L_close = poincare_dist(z_fine, z_coarse, c).mean() * 0.1
    else:
        L_close = (z_fine - z_coarse).norm(dim=-1).mean() * 0.1

    return L_rank + L_close
