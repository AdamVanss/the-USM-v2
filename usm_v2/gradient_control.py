"""
USM v2 — Riemannian gradient control.

Three mechanisms that together prevent the radial collapse observed at scale:
  1. Conformal factor-aware gradient scaling
  2. Adaptive Riemannian gradient clipping
  3. Euclidean burn-in with smooth curvature ramp
"""

import torch
from .manifold import conformal_factor, EPS


# ---------------------------------------------------------------------------
# 1. Conformal-factor gradient scaling
# ---------------------------------------------------------------------------

def scale_riemannian_grads(params, c: torch.Tensor):
    """
    Rescale Euclidean gradients of manifold parameters by the inverse
    squared conformal factor.

    In the Poincaré ball, the Riemannian metric at x is g_x = λ_x² · I,
    so the Riemannian gradient is (1/λ²) times the Euclidean gradient.
    Without this rescaling, points near the boundary receive explosively
    large effective updates — the root cause of radial collapse.
    """
    if c.item() < EPS:
        return

    for p in params:
        if p.grad is None or p.dim() < 1:
            continue
        lam = conformal_factor(p.data, c)  # [..., 1]
        p.grad.data.div_(lam.pow(2).clamp(max=1e6))


# ---------------------------------------------------------------------------
# 2. Riemannian-aware gradient clipping
# ---------------------------------------------------------------------------

def clip_riemannian_grad_norm(params, max_norm: float, c: torch.Tensor):
    """
    Clip gradients using Riemannian norm rather than Euclidean norm.

    The Riemannian norm of a tangent vector v at point x is ||v||_x = ||v|| / λ_x.
    Clipping in this metric prevents over-correction near the boundary while
    allowing healthy updates near the origin.
    """
    if c.item() < EPS:
        torch.nn.utils.clip_grad_norm_(params, max_norm)
        return

    total_norm_sq = 0.0
    grads = []
    for p in params:
        if p.grad is None:
            continue
        if p.dim() >= 1:
            lam = conformal_factor(p.data, c)
            riem_grad = p.grad.data / lam
            riem_norm = riem_grad.norm()
        else:
            riem_norm = p.grad.data.norm()
        total_norm_sq += riem_norm.item() ** 2
        grads.append(p.grad)

    total_norm = total_norm_sq ** 0.5
    clip_coef = max_norm / (total_norm + 1e-6)
    if clip_coef < 1.0:
        for g in grads:
            g.data.mul_(clip_coef)


# ---------------------------------------------------------------------------
# 3. Euclidean burn-in scheduler
# ---------------------------------------------------------------------------

class BurninScheduler:
    """
    Controls a smooth transition from Euclidean (c_eff=0) to full hyperbolic
    geometry over training.

    During burn-in epochs the curvature scale is 0 (pure Euclidean), then it
    linearly ramps to 1.0 over the transition window. This lets the model
    learn a reasonable initial layout before hyperbolic geometry amplifies
    gradients near the boundary.

    Inspired by Nickel & Kiela (2017) who found burn-in critical for
    Poincaré embedding stability.
    """

    def __init__(self, burnin_epochs: int = 10, transition_epochs: int = 5):
        self.burnin_epochs = burnin_epochs
        self.transition_epochs = transition_epochs

    def get_curvature_scale(self, epoch: int) -> float:
        """Returns curvature multiplier in [0, 1]."""
        if epoch < self.burnin_epochs:
            return 0.0
        elif epoch < self.burnin_epochs + self.transition_epochs:
            progress = (epoch - self.burnin_epochs) / max(self.transition_epochs, 1)
            return min(progress, 1.0)
        return 1.0

    def is_hyperbolic(self, epoch: int) -> bool:
        return self.get_curvature_scale(epoch) > 0.0
