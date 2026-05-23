"""
USM v2 — Universal Semantic Manifold with learnable curvature,
Riemannian gradient control, and curriculum training.

Package structure:
    config.py           Unified configuration
    manifold.py         LearnablePoincareBall + hyperbolic primitives
    encoders.py         ConceptEncoder, VisionEncoder
    operators.py        CompositionalOperator, RelationMaps
    losses.py           All loss functions
    gradient_control.py Conformal factor scaling, Riemannian clipping, burn-in
    curriculum.py       Difficulty scoring, progressive sampling, loss scheduling
    data.py             Data loading (ConceptNet, SNLI, CIFAR-100, cross-lingual)
    training.py         Phase 1 & Phase 2 training loops
    evaluation.py       Link prediction, cross-modal retrieval, hierarchy depth
"""

from .config import USMConfig
from .manifold import LearnablePoincareBall

__version__ = "2.0.0"
