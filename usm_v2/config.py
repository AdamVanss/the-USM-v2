"""
USM v2 — Unified configuration.

VRAM-adaptive settings: detects GPU memory and selects scale profile.
"""

import torch
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class USMConfig:
    seed: int = 42

    # --- Manifold ---
    c_init: float = 1.0
    c_min: float = 0.01
    c_max: float = 10.0
    c_lr: float = 1e-3
    learnable_curvature: bool = True

    # --- Dimensions ---
    d: int = 1024
    d_backbone: int = 384  # MiniLM output
    d_clip: int = 768

    # --- Backbones ---
    text_backbone: str = "paraphrase-multilingual-MiniLM-L12-v2"
    clip_model: str = "openai/clip-vit-large-patch14"

    # --- Relations ---
    relations: tuple = ("IS_A", "CAUSES", "PART_OF", "SIMILAR_TO", "ANTONYM", "CAPABLE_OF")

    # --- Training Phase 1 ---
    n_epochs_p1: int = 60
    batch_size: int = 256
    lr_p1: float = 5e-4
    weight_decay: float = 1e-4
    grad_clip_norm: float = 1.0

    # --- Training Phase 2 ---
    n_epochs_p2: int = 60
    lr_p2_vis: float = 3e-4
    lr_p2_text: float = 1e-5
    p2_batch: int = 1024

    # --- Data ---
    max_cn_triples: int = 400_000
    max_cl_per_lang: int = 20_000
    max_snli_pairs: int = 20_000

    # --- Gradient control ---
    burnin_epochs: int = 10
    transition_epochs: int = 5
    use_riemannian_clipping: bool = True

    # --- Curriculum ---
    curriculum_enabled: bool = True
    curriculum_full_data_pct: float = 0.7
    hard_neg_max_ratio: float = 0.8

    # --- Loss weights (base values, curriculum may modulate) ---
    w_rel: float = 1.0
    w_comp: float = 0.1
    w_cl: float = 1.0
    w_ent: float = 0.5
    w_hier: float = 0.5

    # --- Loss margins ---
    margin_rel: float = 2.0
    margin_hier: float = 1.0
    delta_contradiction: float = 2.0

    # --- Infrastructure ---
    ckpt_dir: str = "/tmp/usm_v2_checkpoints"
    ckpt_every: int = 5
    encode_batch: int = 512
    grad_accum: int = 2

    # --- Computed at runtime ---
    device: Optional[torch.device] = field(default=None, repr=False)
    use_bf16: bool = False
    large_gpu: bool = False

    def __post_init__(self):
        if self.device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        if torch.cuda.is_available():
            vram_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
            self.large_gpu = vram_gb >= 24
            self.use_bf16 = vram_gb >= 38
        else:
            self.large_gpu = False
            self.use_bf16 = False

        if not self.large_gpu:
            self._apply_medium_scale()

    def _apply_medium_scale(self):
        """Downscale for T4 / consumer GPU (12-16 GB)."""
        self.d = 512
        self.d_clip = 512
        self.clip_model = "openai/clip-vit-base-patch32"
        self.n_epochs_p1 = 30
        self.n_epochs_p2 = 30
        self.batch_size = 128
        self.max_cn_triples = 100_000
        self.encode_batch = 256
        self.grad_accum = 1
