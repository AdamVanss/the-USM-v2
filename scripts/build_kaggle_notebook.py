"""Build a single self-contained Kaggle notebook from usm_v2 source modules."""
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MODULES = [
    "config.py", "manifold.py", "gradient_control.py", "operators.py",
    "losses.py", "encoders.py", "curriculum.py", "data.py",
    "training.py", "evaluation.py",
]


def strip_relative_imports(src: str) -> str:
    lines = src.splitlines()
    out = []
    skip = False
    paren_depth = 0
    for line in lines:
        stripped = line.lstrip()
        if not skip and re.match(r"^from\s+\.", stripped):
            skip = True
            paren_depth = line.count("(") - line.count(")")
            if paren_depth <= 0:
                skip = False
            continue
        if skip:
            paren_depth += line.count("(") - line.count(")")
            if paren_depth <= 0:
                skip = False
            continue
        out.append(line)
    return "\n".join(out)


def read_module(name: str) -> str:
    text = (ROOT / "usm_v2" / name).read_text(encoding="utf-8")
    text = re.sub(r'^"""[\s\S]*?"""\n+', "", text, count=1)
    return strip_relative_imports(text).strip()


def cell_md(text: str) -> dict:
    return {"cell_type": "markdown", "metadata": {}, "source": [text]}


def cell_code(text: str) -> dict:
    return {
        "cell_type": "code",
        "metadata": {},
        "source": [line + "\n" for line in text.splitlines()],
        "execution_count": None,
        "outputs": [],
    }


def main():
    library_header = '''# =============================================================================
# USM v2 — inlined library (single-notebook build for Kaggle)
# =============================================================================
import os, re, random, time, math
from collections import defaultdict, Counter
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from tqdm.auto import tqdm
import geoopt
'''

    parts = [library_header]
    for mod in MODULES:
        parts.append(f"\n\n# ----- {mod} -----\n")
        parts.append(read_module(mod))
    library_code = "".join(parts)

    cells = [
        cell_md(
            "# Universal Semantic Manifold — v2 (Single Notebook)\n\n"
            "Self-contained notebook for **Kaggle** (no repo clone needed).\n\n"
            "**Before running:**\n"
            "1. **Settings → Accelerator → GPU** (T4 or P100)\n"
            "2. **Settings → Internet → ON** (downloads ConceptNet, SNLI, models)\n"
            "3. Keep `VALIDATION_MODE = True` for a ~5–10 min smoke test\n\n"
            "Pipeline: learnable Poincaré curvature · Riemannian gradient control · "
            "curriculum training · cross-modal alignment."
        ),
        cell_code(
            "!pip install -q geoopt sentence-transformers transformers datasets umap-learn torchvision\n"
            "print('Dependencies installed.')"
        ),
        cell_md("## Library (all usm_v2 code inlined)"),
        cell_code(library_code),
        cell_md("## Configuration"),
        cell_code(
            "VALIDATION_MODE = True  # False = full training scale\n\n"
            "CKPT_DIR = '/kaggle/working/checkpoints'\n"
            "DATA_ROOT = '/kaggle/working/data'\n"
            "os.makedirs(CKPT_DIR, exist_ok=True)\n"
            "os.makedirs(DATA_ROOT, exist_ok=True)\n\n"
            "cfg = USMConfig(validation_mode=VALIDATION_MODE)\n"
            "cfg.ckpt_dir = CKPT_DIR\n\n"
            "torch.manual_seed(cfg.seed)\n"
            "random.seed(cfg.seed)\n"
            "np.random.seed(cfg.seed)\n\n"
            "mode = 'VALIDATION (smoke test)' if cfg.validation_mode else 'TRAINING (full scale)'\n"
            "print(f'Mode:              {mode}')\n"
            "print(f'Dimension:         {cfg.d}')\n"
            "print(f'P1 epochs:         {cfg.n_epochs_p1}  |  P2 epochs: {cfg.n_epochs_p2}')\n"
            "print(f'ConceptNet cap:    {cfg.max_cn_triples:,} triples')\n"
            "print(f'Device:            {cfg.device}')\n"
            "print(f'Large GPU:         {cfg.large_gpu}')\n"
            "if torch.cuda.is_available():\n"
            "    print(f'GPU:               {torch.cuda.get_device_name(0)}')"
        ),
        cell_md("## 1. Build Manifold & Models"),
        cell_code(
            "manifold = LearnablePoincareBall(\n"
            "    c_init=cfg.c_init, c_min=cfg.c_min, c_max=cfg.c_max,\n"
            "    learnable=cfg.learnable_curvature,\n"
            ").to(cfg.device)\n\n"
            "encoder = ConceptEncoder(\n"
            "    manifold, d_out=cfg.d, backbone=cfg.text_backbone,\n"
            "    hyperbolic=True, device=cfg.device,\n"
            ").to(cfg.device)\n\n"
            "vis_encoder = VisionEncoder(\n"
            "    manifold, d_out=cfg.d, clip_model=cfg.clip_model,\n"
            "    d_clip=cfg.d_clip, hyperbolic=True, device=cfg.device,\n"
            ").to(cfg.device)\n\n"
            "comp_op = CompositionalOperator(manifold, d=cfg.d, hyperbolic=True).to(cfg.device)\n"
            "rel_maps = RelationMaps(manifold, d=cfg.d, hyperbolic=True).to(cfg.device)\n\n"
            "n_params = sum(p.numel() for p in [\n"
            "    *manifold.parameters(), *encoder.parameters(), *vis_encoder.parameters(),\n"
            "    *comp_op.parameters(), *rel_maps.parameters(),\n"
            "] if p.requires_grad)\n"
            "print(f'Trainable parameters: {n_params:,}')\n"
            "print(f'Initial curvature c = {manifold.c.item():.4f}')"
        ),
        cell_md("## 2. Load Data"),
        cell_code(
            "print('--- ConceptNet ---')\n"
            "triples = load_conceptnet(max_triples=cfg.max_cn_triples)\n\n"
            "print('\\n--- Cross-lingual ---')\n"
            "cl_pairs = load_crosslingual(max_per_lang=cfg.max_cl_per_lang)\n\n"
            "print('\\n--- SNLI ---')\n"
            "snli_pairs = load_snli(max_pairs=cfg.max_snli_pairs)\n\n"
            "vocab_set = set()\n"
            "for h, _, t in triples:\n"
            "    vocab_set.add(h)\n"
            "    vocab_set.add(t)\n"
            "vocab_set.update(CIFAR100_FINE)\n"
            "vocab_set.update(CIFAR100_COARSE)\n"
            "vocab_list = sorted(vocab_set)\n"
            "concept2bb = {c: i for i, c in enumerate(vocab_list)}\n\n"
            "print(f'\\nVocabulary: {len(vocab_list):,} concepts')\n"
            "print(f'Triples:   {len(triples):,}')\n"
            "print(f'CL pairs:  {len(cl_pairs):,}')\n"
            "print(f'SNLI:      {len(snli_pairs):,}')"
        ),
        cell_md("## 3. Pre-cache Embeddings"),
        cell_code(
            "vocab_bb = _precache_text_embeddings(encoder, vocab_list, batch_size=cfg.encode_batch, device=cfg.device)\n"
            "print(f'Vocab backbone cache: {vocab_bb.shape}')\n\n"
            "print('Pre-caching cross-lingual pairs...')\n"
            "if cl_pairs:\n"
            "    cl_src_texts, cl_tgt_texts = zip(*cl_pairs)\n"
            "    cl_bb_src = _precache_text_embeddings(encoder, list(cl_src_texts), batch_size=cfg.encode_batch, device=cfg.device)\n"
            "    cl_bb_tgt = _precache_text_embeddings(encoder, list(cl_tgt_texts), batch_size=cfg.encode_batch, device=cfg.device)\n"
            "    print(f'  CL cache: {cl_bb_src.shape}')\n"
            "else:\n"
            "    cl_bb_src = cl_bb_tgt = None\n\n"
            "print('Pre-caching SNLI pairs...')\n"
            "if snli_pairs:\n"
            "    snli_a, snli_b, snli_labels = zip(*snli_pairs)\n"
            "    snli_bb_a = _precache_text_embeddings(encoder, list(snli_a), batch_size=cfg.encode_batch, device=cfg.device)\n"
            "    snli_bb_b = _precache_text_embeddings(encoder, list(snli_b), batch_size=cfg.encode_batch, device=cfg.device)\n"
            "    snli_labels_t = torch.tensor(snli_labels, device=cfg.device)\n"
            "    print(f'  SNLI cache: {snli_bb_a.shape}')\n"
            "else:\n"
            "    snli_bb_a = snli_bb_b = snli_labels_t = None\n\n"
            "fine_bb = encoder.encode_backbone(CIFAR100_FINE).to(cfg.device)\n"
            "coarse_bb = encoder.encode_backbone(CIFAR100_COARSE).to(cfg.device)\n"
            "fine2coarse_tensor = torch.tensor(\n"
            "    [CIFAR100_FINE2COARSE[i] for i in range(len(CIFAR100_FINE))],\n"
            "    device=cfg.device,\n"
            ")\n\n"
            "random.shuffle(triples)\n"
            "n_test = max(500, len(triples) // 20)\n"
            "test_triples = triples[:n_test]\n"
            "train_triples = triples[n_test:]\n"
            "print(f'Train triples: {len(train_triples):,}  |  Test triples: {len(test_triples):,}')"
        ),
        cell_md("## 4. Phase 1 — Text-only Training"),
        cell_code(
            "history_p1 = train_phase1(\n"
            "    cfg, manifold, encoder, comp_op, rel_maps,\n"
            "    train_triples, cl_pairs, snli_pairs,\n"
            "    vocab_list, vocab_bb, concept2bb,\n"
            "    cl_bb_src=cl_bb_src, cl_bb_tgt=cl_bb_tgt,\n"
            "    snli_bb_a=snli_bb_a, snli_bb_b=snli_bb_b, snli_labels_t=snli_labels_t,\n"
            ")\n"
            "print(f'\\nFinal curvature c = {manifold.c.item():.4f}')"
        ),
        cell_md("## 5. Phase 1 Evaluation"),
        cell_code(
            "_eval_cap = 200 if cfg.validation_mode else 2000\n"
            "lp_results = evaluate_link_prediction(\n"
            "    encoder, rel_maps, test_triples, vocab_list,\n"
            "    vocab_bb, concept2bb, manifold.c,\n"
            "    hyperbolic=True, tag='P1', device=cfg.device,\n"
            "    max_eval=_eval_cap,\n"
            ")\n"
            "print('Link Prediction (Phase 1):')\n"
            "for k, v in lp_results.items():\n"
            "    print(f'  {k}: {v:.4f}' if isinstance(v, float) else f'  {k}: {v}')"
        ),
        cell_md("## 6. Phase 2 — Multimodal Training"),
        cell_code(
            "print('Loading CIFAR-100...')\n"
            "cifar_train, cifar_test = load_cifar100(data_root=DATA_ROOT)\n\n"
            "cifar_dl_train = DataLoader(cifar_train, batch_size=256, shuffle=False, num_workers=2)\n"
            "cifar_dl_test = DataLoader(cifar_test, batch_size=256, shuffle=False, num_workers=2)\n\n"
            "print('Pre-caching CLIP embeddings...')\n"
            "clip_train, clip_labels_train = _precache_clip_embeddings(\n"
            "    vis_encoder, cifar_dl_train, device=cfg.device, max_images=cfg.max_clip_images,\n"
            ")\n"
            "clip_test, clip_labels_test = _precache_clip_embeddings(\n"
            "    vis_encoder, cifar_dl_test, device=cfg.device, max_images=cfg.max_clip_images,\n"
            ")\n"
            "print(f'Train CLIP: {clip_train.shape}, Test CLIP: {clip_test.shape}')\n\n"
            "history_p2 = train_phase2(\n"
            "    cfg, manifold, encoder, vis_encoder,\n"
            "    clip_train, clip_labels_train,\n"
            "    fine_bb, coarse_bb, fine2coarse_tensor,\n"
            ")\n"
            "print(f'\\nFinal curvature c = {manifold.c.item():.4f}')"
        ),
        cell_md("## 7. Phase 2 Evaluation"),
        cell_code(
            "fine_z = encoder.project(fine_bb, c=manifold.c)\n\n"
            "xm_results = evaluate_crossmodal(\n"
            "    vis_encoder, fine_z, clip_test, clip_labels_test,\n"
            "    manifold.c, hyperbolic=True, tag='P2', device=cfg.device,\n"
            ")\n"
            "print('Cross-modal Retrieval (Phase 2):')\n"
            "for k, v in xm_results.items():\n"
            "    print(f'  {k}: {v:.4f}' if isinstance(v, float) else f'  {k}: {v}')\n\n"
            "hier_results = evaluate_hierarchy(\n"
            "    vis_encoder, encoder, fine_bb, coarse_bb,\n"
            "    fine2coarse_tensor, manifold.c, hyperbolic=True, tag='P2',\n"
            ")\n"
            "print(f'\\nHierarchy Accuracy: {hier_results[\"accuracy\"]:.2%}')\n"
            "print(f'Mean depth diff (fine - coarse): {hier_results[\"mean_diff\"]:.4f}')"
        ),
        cell_md("## 8. Visualization & Save"),
        cell_code(
            "import matplotlib.pyplot as plt\n\n"
            "fig, axes = plt.subplots(2, 3, figsize=(18, 10))\n\n"
            "axes[0, 0].plot(history_p1['loss_total'], label='Total')\n"
            "axes[0, 0].plot(history_p1['loss_rel'], label='L_rel', alpha=0.7)\n"
            "axes[0, 0].plot(history_p1['loss_cl'], label='L_cl', alpha=0.7)\n"
            "axes[0, 0].set_title('Phase 1 Losses')\n"
            "axes[0, 0].legend()\n\n"
            "axes[0, 1].plot(history_p1['curvature'], label='Phase 1', color='tab:blue')\n"
            "if history_p2['curvature']:\n"
            "    offset = len(history_p1['curvature'])\n"
            "    axes[0, 1].plot(range(offset, offset + len(history_p2['curvature'])),\n"
            "                    history_p2['curvature'], label='Phase 2', color='tab:orange')\n"
            "axes[0, 1].set_title('Learned Curvature c')\n"
            "axes[0, 1].axhline(y=1.0, color='gray', linestyle='--', alpha=0.5)\n"
            "axes[0, 1].legend()\n\n"
            "axes[0, 2].plot(history_p1['curriculum_pct'], color='tab:green')\n"
            "axes[0, 2].set_title('Curriculum Progress')\n\n"
            "if history_p2['loss_total']:\n"
            "    axes[1, 0].plot(history_p2['loss_total'])\n"
            "    axes[1, 0].set_title('Phase 2 Loss')\n\n"
            "axes[1, 1].hist(hier_results['fine_depths'], bins=30, alpha=0.6, label='Fine')\n"
            "axes[1, 1].hist(hier_results['coarse_depths'], bins=30, alpha=0.6, label='Coarse')\n"
            "axes[1, 1].set_title(f'Hierarchy (acc={hier_results[\"accuracy\"]:.0%})')\n"
            "axes[1, 1].legend()\n\n"
            "try:\n"
            "    import umap\n"
            "    with torch.no_grad():\n"
            "        sample_idx = torch.randperm(len(vocab_list))[:500]\n"
            "        z_sample = encoder.project(vocab_bb[sample_idx], c=manifold.c)\n"
            "        tangent = logmap0(z_sample, manifold.c).cpu().numpy()\n"
            "    umap_2d = umap.UMAP(n_components=2, random_state=42).fit_transform(tangent)\n"
            "    norms = np.linalg.norm(tangent, axis=1)\n"
            "    sc = axes[1, 2].scatter(umap_2d[:, 0], umap_2d[:, 1], c=norms, cmap='viridis', s=8, alpha=0.7)\n"
            "    plt.colorbar(sc, ax=axes[1, 2], label='depth')\n"
            "    axes[1, 2].set_title('UMAP of Concepts')\n"
            "except ImportError:\n"
            "    axes[1, 2].text(0.5, 0.5, 'umap unavailable', ha='center', va='center', transform=axes[1, 2].transAxes)\n\n"
            "plt.tight_layout()\n"
            "out_png = os.path.join(CKPT_DIR, 'usm_v2_results.png')\n"
            "plt.savefig(out_png, dpi=150, bbox_inches='tight')\n"
            "plt.show()\n"
            "print(f'Saved plot: {out_png}')\n\n"
            "final_path = os.path.join(CKPT_DIR, 'usm_v2_final.pt')\n"
            "torch.save({\n"
            "    'manifold': manifold.state_dict(),\n"
            "    'encoder': encoder.state_dict(),\n"
            "    'vis_encoder': vis_encoder.state_dict(),\n"
            "    'comp_op': comp_op.state_dict(),\n"
            "    'rel_maps': rel_maps.state_dict(),\n"
            "    'config': cfg,\n"
            "    'history_p1': history_p1,\n"
            "    'history_p2': history_p2,\n"
            "    'lp_results': lp_results,\n"
            "    'xm_results': xm_results,\n"
            "    'hier_results': hier_results,\n"
            "    'final_curvature': manifold.c.item(),\n"
            "}, final_path)\n"
            "print(f'Saved model: {final_path}')\n"
            "print(f'\\n=== SUMMARY ===')\n"
            "print(f'MRR:          {lp_results[\"MRR\"]:.4f}')\n"
            "print(f'Hits@10:      {lp_results[\"Hits@10\"]:.4f}')\n"
            "print(f'Hierarchy:    {hier_results[\"accuracy\"]:.2%}')\n"
            "print(f'Cross-modal:  R@5={xm_results[\"R@5\"]:.4f}')\n"
            "print(f'Curvature c:  {manifold.c.item():.4f}')"
        ),
    ]

    nb = {
        "nbformat": 4,
        "nbformat_minor": 5,
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3",
            },
            "language_info": {"name": "python", "version": "3.10.0"},
            "accelerator": "GPU",
        },
        "cells": cells,
    }

    out_path = ROOT / "notebooks" / "usm_v2_kaggle.ipynb"
    out_path.write_text(json.dumps(nb, indent=1), encoding="utf-8")
    print(f"Wrote {out_path} ({out_path.stat().st_size // 1024} KB, {len(cells)} cells)")
    print(f"Library cell: {len(library_code.splitlines())} lines")


if __name__ == "__main__":
    main()
