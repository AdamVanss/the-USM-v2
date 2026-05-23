"""
USM v2 — Learnable Poincaré ball with curvature-parameterized primitives.

All hyperbolic operations are parameterized by the curvature c, which can be
learned end-to-end. When c -> 0 the ball flattens to Euclidean space, enabling
a smooth burn-in transition.
"""

import torch
import torch.nn as nn
import geoopt

EPS = 1e-5


class LearnablePoincareBall(nn.Module):
    """
    Poincaré ball with per-relation learnable curvature.

    Each relation type gets its own curvature c_r. The base curvature
    (used for projection) is max(c_r) so all points fit in every
    relation-specific ball. This lets IS_A develop strong curvature for
    tree structure while SIMILAR_TO stays near-Euclidean.
    """

    def __init__(self, c_init: float = 0.001, c_min: float = 0.0001, c_max: float = 10.0,
                 learnable: bool = True, n_relations: int = 6):
        super().__init__()
        self.c_min = c_min
        self.c_max = c_max
        self.n_relations = n_relations
        if learnable:
            self._c_params = nn.Parameter(torch.full((n_relations,), float(c_init)))
        else:
            self.register_buffer("_c_params", torch.full((n_relations,), float(c_init)))

    @property
    def c(self) -> torch.Tensor:
        """Base curvature for projection (max over all relations)."""
        return torch.clamp(self._c_params.max(), self.c_min, self.c_max)

    def c_rel(self, rel_idx: int) -> torch.Tensor:
        """Curvature for a specific relation index (scalar)."""
        return torch.clamp(self._c_params[rel_idx], self.c_min, self.c_max)

    def effective_c(self, scale: float = 1.0) -> torch.Tensor:
        """Base curvature modulated by burn-in scale in [0, 1]."""
        if scale <= 0.0:
            return torch.tensor(0.0, device=self._c_params.device)
        return self.c * scale

    def effective_c_rel(self, rel_idx: int, scale: float = 1.0) -> torch.Tensor:
        """Per-relation curvature modulated by burn-in scale."""
        if scale <= 0.0:
            return torch.tensor(0.0, device=self._c_params.device)
        return self.c_rel(rel_idx) * scale

    @property
    def ball(self) -> geoopt.PoincareBall:
        return geoopt.PoincareBall(c=self.c)

    def max_norm(self, c: torch.Tensor) -> torch.Tensor:
        """Maximum allowed norm for points: 1/sqrt(c) - eps."""
        safe_c = c.clamp(min=EPS)
        return (1.0 / safe_c.sqrt()) - EPS


def clamp_to_ball(z: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
    """Project z inside the Poincaré ball of curvature c."""
    safe_c = c.clamp(min=EPS)
    max_n = (1.0 / safe_c.sqrt()) - EPS
    norms = z.norm(dim=-1, keepdim=True).clamp(min=1e-10)
    scale = torch.where(norms >= max_n, max_n / norms, torch.ones_like(norms))
    return z * scale


def conformal_factor(z: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
    """
    Conformal factor lambda_x = 2 / (1 - c * ||x||^2).
    Shape: same as z but with last dim squeezed.
    """
    safe_c = c.clamp(min=EPS)
    norm_sq = z.pow(2).sum(dim=-1, keepdim=True)
    return 2.0 / (1.0 - safe_c * norm_sq).clamp(min=EPS)


# ---------------------------------------------------------------------------
# Hyperbolic primitives parameterized by curvature c
# ---------------------------------------------------------------------------

def _mobius_add(x: torch.Tensor, y: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
    """Möbius addition in the Poincaré ball with curvature c."""
    x_sq = (x * x).sum(dim=-1, keepdim=True)
    y_sq = (y * y).sum(dim=-1, keepdim=True)
    xy = (x * y).sum(dim=-1, keepdim=True)
    num = (1 + 2 * c * xy + c * y_sq) * x + (1 - c * x_sq) * y
    denom = (1 + 2 * c * xy + c * c * x_sq * y_sq).clamp(min=EPS)
    return num / denom


def expmap0(v: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
    """Exponential map from the origin."""
    safe_c = c.clamp(min=EPS)
    sqrt_c = safe_c.sqrt()
    v_norm = v.norm(dim=-1, keepdim=True).clamp(min=1e-10)
    return clamp_to_ball(
        torch.tanh(sqrt_c * v_norm) * v / (sqrt_c * v_norm),
        c
    )


def logmap0(z: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
    """Logarithmic map to the origin (inverse of expmap0)."""
    z = clamp_to_ball(z, c)
    safe_c = c.clamp(min=EPS)
    sqrt_c = safe_c.sqrt()
    z_norm = z.norm(dim=-1, keepdim=True).clamp(min=1e-10)
    return torch.atanh(sqrt_c * z_norm).clamp(max=10.0) * z / (sqrt_c * z_norm)


def expmap(x: torch.Tensor, v: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
    """Exponential map from point x in direction v."""
    safe_c = c.clamp(min=EPS)
    sqrt_c = safe_c.sqrt()
    x = clamp_to_ball(x, c)
    lam = conformal_factor(x, c)
    v_norm = v.norm(dim=-1, keepdim=True).clamp(min=1e-10)
    second = torch.tanh(sqrt_c * lam * v_norm / 2.0) * v / (sqrt_c * v_norm)
    return clamp_to_ball(_mobius_add(x, second, c), c)


def poincare_dist(z1: torch.Tensor, z2: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
    """Geodesic distance between z1 and z2 on the Poincaré ball."""
    safe_c = c.clamp(min=EPS)
    sqrt_c = safe_c.sqrt()
    z1 = clamp_to_ball(z1, c)
    z2 = clamp_to_ball(z2, c)
    diff_sq = (z1 - z2).pow(2).sum(dim=-1)
    n1_sq = z1.pow(2).sum(dim=-1)
    n2_sq = z2.pow(2).sum(dim=-1)
    denom = ((1.0 - safe_c * n1_sq) * (1.0 - safe_c * n2_sq)).clamp(min=EPS)
    arg = 1.0 + 2.0 * safe_c * diff_sq / denom
    return (1.0 / sqrt_c) * torch.acosh(arg.clamp(min=1.0 + 1e-6))


def poincare_cdist(z1: torch.Tensor, z2: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
    """Batched pairwise geodesic distance matrix [B1, B2]."""
    safe_c = c.clamp(min=EPS)
    sqrt_c = safe_c.sqrt()
    z1 = clamp_to_ball(z1, c)
    z2 = clamp_to_ball(z2, c)
    n1_sq = z1.pow(2).sum(dim=-1, keepdim=True)
    n2_sq = z2.pow(2).sum(dim=-1, keepdim=True).T
    dot = z1 @ z2.T
    diff_sq = (n1_sq + n2_sq - 2.0 * dot).clamp(min=0.0)
    denom = ((1.0 - safe_c * n1_sq) * (1.0 - safe_c * n2_sq)).clamp(min=EPS)
    arg = 1.0 + 2.0 * safe_c * diff_sq / denom
    return (1.0 / sqrt_c) * torch.acosh(arg.clamp(min=1.0 + 1e-6))


def dist0(z: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
    """Geodesic distance from the origin."""
    z = clamp_to_ball(z, c)
    safe_c = c.clamp(min=EPS)
    sqrt_c = safe_c.sqrt()
    n_sq = z.pow(2).sum(dim=-1)
    arg = 1.0 + 2.0 * safe_c * n_sq / (1.0 - safe_c * n_sq).clamp(min=EPS)
    return (1.0 / sqrt_c) * torch.acosh(arg.clamp(min=1.0))


# ---------------------------------------------------------------------------
# Euclidean equivalents (baseline)
# ---------------------------------------------------------------------------

def eucl_dist(z1: torch.Tensor, z2: torch.Tensor) -> torch.Tensor:
    return (z1 - z2).pow(2).sum(dim=-1).sqrt()


def eucl_cdist(z1: torch.Tensor, z2: torch.Tensor) -> torch.Tensor:
    return torch.cdist(z1, z2, p=2)
