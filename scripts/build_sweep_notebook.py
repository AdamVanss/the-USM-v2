"""Build dimensionality-sweep notebook: Hyperbolic vs Euclidean at d=[32,64,128,256,512].

Pre-caches backbone embeddings ONCE, then loops over dimensions and geometry types.
Generates a comparison table + matplotlib plots proving hyperbolic advantage at low d.
"""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MODULES = [
    "config.py", "manifold.py", "gradient_control.py", "operators.py",
    "losses.py", "encoders.py", "curriculum.py", "data.py",
    "training.py", "evaluation.py",
]


def strip_relative_imports(src: str) -> str:
    import re
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
    import re
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


# ── Sweep-specific cells ────────────────────────────────────────────────

TITLE_MD = (
    "# USM v2 — Dimensionality Sweep: Hyperbolic vs Euclidean\n\n"
    "Trains **both** geometries at `d = [32, 64, 128, 256, 512]` on "
    "the same data and architecture, proving that **hyperbolic embeddings "
    "achieve equivalent quality in far fewer dimensions**.\n\n"
    "**Before running:**\n"
    "1. **Settings → Accelerator → GPU T4 x2**\n"
    "2. **Settings → Internet → ON**\n\n"
    "Backbone embeddings are pre-cached once and reused across all runs.\n\n"
    "**Estimated time:** ~2.5–3 hours on T4 x2."
)

INSTALL_CODE = (
    "!pip install -q geoopt sentence-transformers transformers datasets umap-learn torchvision\n"
    "print('Dependencies installed.')"
)

CONFIG_CODE = """\
import time as _time, gc, copy as _copy

CKPT_DIR = '/kaggle/working/checkpoints'
DATA_ROOT = '/kaggle/working/data'
import os
os.makedirs(CKPT_DIR, exist_ok=True)
os.makedirs(DATA_ROOT, exist_ok=True)

cfg = USMConfig(validation_mode=False)
cfg.ckpt_dir = CKPT_DIR

torch.manual_seed(cfg.seed)
random.seed(cfg.seed)
np.random.seed(cfg.seed)

print(f'Base config: d={cfg.d}, batch={cfg.batch_size}, GPUs={cfg.n_gpus}')
if torch.cuda.is_available():
    for i in range(cfg.n_gpus):
        props = torch.cuda.get_device_properties(i)
        print(f'  GPU {i}: {torch.cuda.get_device_name(i)} ({props.total_memory / 1e9:.1f} GB)')
print(f'Training device:  {cfg.device}')
print(f'Backbone device:  {cfg.backbone_device}')
"""

DATA_CODE = """\
print('--- ConceptNet ---')
triples = load_conceptnet(max_triples=cfg.max_cn_triples)

print('\\n--- Cross-lingual ---')
cl_pairs = load_crosslingual(max_per_lang=cfg.max_cl_per_lang)

print('\\n--- SNLI ---')
snli_pairs = load_snli(max_pairs=cfg.max_snli_pairs)

vocab_set = set()
for h, _, t in triples:
    vocab_set.add(h)
    vocab_set.add(t)
vocab_set.update(CIFAR100_FINE)
vocab_set.update(CIFAR100_COARSE)
vocab_list = sorted(vocab_set)
concept2bb = {c: i for i, c in enumerate(vocab_list)}

print(f'\\nVocabulary: {len(vocab_list):,} concepts')
print(f'Triples:   {len(triples):,}')
print(f'CL pairs:  {len(cl_pairs):,}')
print(f'SNLI:      {len(snli_pairs):,}')

random.shuffle(triples)
n_test = max(500, len(triples) // 20)
test_triples = triples[:n_test]
train_triples = triples[n_test:]
print(f'Train: {len(train_triples):,}  |  Test: {len(test_triples):,}')
"""

PRECACHE_CODE = """\
_run_start = _time.time()

_tmp_manifold = LearnablePoincareBall(
    c_init=cfg.c_init, c_min=cfg.c_min, c_max=cfg.c_max,
    learnable=False, n_relations=cfg.n_relations,
).to(cfg.device)
_tmp_encoder = ConceptEncoder(
    _tmp_manifold, d_out=cfg.d, backbone=cfg.text_backbone,
    hyperbolic=True, device=cfg.device, backbone_device=cfg.backbone_device,
).to(cfg.device)

print('Pre-caching text backbone embeddings (384-dim, reused for all dims)...')
vocab_bb = _precache_text_embeddings(_tmp_encoder, vocab_list, batch_size=cfg.encode_batch, device=cfg.device)
print(f'  vocab_bb: {vocab_bb.shape}')

if cl_pairs:
    cl_src_texts, cl_tgt_texts = zip(*cl_pairs)
    cl_bb_src = _precache_text_embeddings(_tmp_encoder, list(cl_src_texts), batch_size=cfg.encode_batch, device=cfg.device)
    cl_bb_tgt = _precache_text_embeddings(_tmp_encoder, list(cl_tgt_texts), batch_size=cfg.encode_batch, device=cfg.device)
    print(f'  cl_bb: {cl_bb_src.shape}')
else:
    cl_bb_src = cl_bb_tgt = None

if snli_pairs:
    snli_a, snli_b, snli_labels = zip(*snli_pairs)
    snli_bb_a = _precache_text_embeddings(_tmp_encoder, list(snli_a), batch_size=cfg.encode_batch, device=cfg.device)
    snli_bb_b = _precache_text_embeddings(_tmp_encoder, list(snli_b), batch_size=cfg.encode_batch, device=cfg.device)
    snli_labels_t = torch.tensor(snli_labels, device=cfg.device)
    print(f'  snli_bb: {snli_bb_a.shape}')
else:
    snli_bb_a = snli_bb_b = snli_labels_t = None

fine_bb = _tmp_encoder.encode_backbone(CIFAR100_FINE).to(cfg.device)
coarse_bb = _tmp_encoder.encode_backbone(CIFAR100_COARSE).to(cfg.device)
fine2coarse_tensor = torch.tensor(
    [CIFAR100_FINE2COARSE[i] for i in range(len(CIFAR100_FINE))],
    device=cfg.device,
)
print(f'  fine_bb: {fine_bb.shape}  coarse_bb: {coarse_bb.shape}')

del _tmp_encoder, _tmp_manifold
gc.collect()
torch.cuda.empty_cache()

print(f'\\nPre-cache time: {_time.time() - _run_start:.1f}s')
if torch.cuda.is_available():
    for i in range(cfg.n_gpus):
        alloc = torch.cuda.memory_allocated(i) / 1e9
        total = torch.cuda.get_device_properties(i).total_memory / 1e9
        print(f'  GPU {i}: {alloc:.2f} / {total:.1f} GB')
"""

CLIP_CODE = """\
print('Loading CIFAR-100...')
cifar_train, cifar_test = load_cifar100(data_root=DATA_ROOT)
cifar_dl_train = DataLoader(cifar_train, batch_size=256, shuffle=False, num_workers=2)
cifar_dl_test = DataLoader(cifar_test, batch_size=256, shuffle=False, num_workers=2)

_tmp_manifold = LearnablePoincareBall(
    c_init=cfg.c_init, c_min=cfg.c_min, c_max=cfg.c_max,
    learnable=False, n_relations=cfg.n_relations,
).to(cfg.device)
_tmp_vis = VisionEncoder(
    _tmp_manifold, d_out=cfg.d, clip_model=cfg.clip_model,
    d_clip=cfg.d_clip, hyperbolic=True, device=cfg.device,
    backbone_device=cfg.backbone_device,
).to(cfg.device)

print('Pre-caching CLIP embeddings (512-dim, reused for all dims)...')
clip_train, clip_labels_train = _precache_clip_embeddings(
    _tmp_vis, cifar_dl_train, device=cfg.device, max_images=cfg.max_clip_images,
)
clip_test, clip_labels_test = _precache_clip_embeddings(
    _tmp_vis, cifar_dl_test, device=cfg.device, max_images=cfg.max_clip_images,
)
print(f'  CLIP train: {clip_train.shape}, test: {clip_test.shape}')

del _tmp_vis, _tmp_manifold
gc.collect()
torch.cuda.empty_cache()
"""

SWEEP_CODE = """\
DIMS = [32, 64, 128, 256, 512]
SWEEP_P1_EPOCHS = 35
SWEEP_P2_EPOCHS = 20
EVAL_CAP = 1000

C_TARGETS_HYP = (0.05, 0.01, 0.03, 0.003, 0.003, 0.01)
C_WARMUP_START = 5
C_WARMUP_END = 28

all_results = {}
_sweep_start = _time.time()

for d_sweep in DIMS:
    for is_hyp in [True, False]:
        mode = "HYP" if is_hyp else "EUCL"
        key = f"d{d_sweep}_{mode}"

        print(f'\\n{"=" * 65}')
        print(f'  {"HYPERBOLIC" if is_hyp else "EUCLIDEAN":>12s}  d = {d_sweep}')
        print(f'{"=" * 65}')

        torch.manual_seed(cfg.seed)
        random.seed(cfg.seed)
        np.random.seed(cfg.seed)

        cfg_s = _copy.deepcopy(cfg)
        cfg_s.d = d_sweep
        cfg_s.n_epochs_p1 = SWEEP_P1_EPOCHS
        cfg_s.n_epochs_p2 = SWEEP_P2_EPOCHS
        cfg_s.learnable_curvature = False

        if is_hyp:
            cfg_s.c_init = 0.001
            cfg_s.c_min = 0.0001
            cfg_s.c_targets = C_TARGETS_HYP
            cfg_s.c_warmup_start = C_WARMUP_START
            cfg_s.c_warmup_end = C_WARMUP_END
        else:
            cfg_s.c_init = 0.0
            cfg_s.c_min = 0.0
            cfg_s.c_max = 0.0
            cfg_s.c_targets = (0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
            cfg_s.c_warmup_start = 999
            cfg_s.c_warmup_end = 999

        manifold_s = LearnablePoincareBall(
            c_init=cfg_s.c_init, c_min=cfg_s.c_min, c_max=cfg_s.c_max,
            learnable=False, n_relations=cfg_s.n_relations,
        ).to(cfg_s.device)

        encoder_s = ConceptEncoder(
            manifold_s, d_out=d_sweep, backbone=cfg_s.text_backbone,
            hyperbolic=is_hyp, device=cfg_s.device,
            backbone_device=cfg_s.backbone_device,
        ).to(cfg_s.device)

        vis_encoder_s = VisionEncoder(
            manifold_s, d_out=d_sweep, clip_model=cfg_s.clip_model,
            d_clip=cfg_s.d_clip, hyperbolic=is_hyp, device=cfg_s.device,
            backbone_device=cfg_s.backbone_device,
        ).to(cfg_s.device)

        comp_op_s = CompositionalOperator(manifold_s, d=d_sweep, hyperbolic=is_hyp).to(cfg_s.device)
        rel_maps_s = RelationMaps(manifold_s, d=d_sweep, hyperbolic=is_hyp).to(cfg_s.device)

        n_params = sum(p.numel() for p in [
            *manifold_s.parameters(), *encoder_s.parameters(), *vis_encoder_s.parameters(),
            *comp_op_s.parameters(), *rel_maps_s.parameters(),
        ] if p.requires_grad)
        print(f'  Trainable params: {n_params:,}')

        # ── Phase 1 ──────────────────────────────────────
        _t0 = _time.time()
        history_p1_s = train_phase1(
            cfg_s, manifold_s, encoder_s, comp_op_s, rel_maps_s,
            train_triples, cl_pairs, snli_pairs,
            vocab_list, vocab_bb, concept2bb,
            cl_bb_src=cl_bb_src, cl_bb_tgt=cl_bb_tgt,
            snli_bb_a=snli_bb_a, snli_bb_b=snli_bb_b,
            snli_labels_t=snli_labels_t,
        )
        p1_time = _time.time() - _t0

        c_eval = manifold_s.c
        lp = evaluate_link_prediction(
            encoder_s, rel_maps_s, test_triples, vocab_list,
            vocab_bb, concept2bb, c_eval,
            hyperbolic=is_hyp, manifold=manifold_s if is_hyp else None,
            tag=key, device=cfg_s.device, max_eval=EVAL_CAP,
        )

        # ── Phase 2 ──────────────────────────────────────
        _t1 = _time.time()
        history_p2_s = train_phase2(
            cfg_s, manifold_s, encoder_s, vis_encoder_s,
            clip_train, clip_labels_train,
            fine_bb, coarse_bb, fine2coarse_tensor,
        )
        p2_time = _time.time() - _t1

        with torch.no_grad():
            fine_z_s = encoder_s.project(fine_bb, c=c_eval)
        xm = evaluate_crossmodal(
            vis_encoder_s, fine_z_s, clip_test, clip_labels_test,
            c_eval, hyperbolic=is_hyp, tag=key, device=cfg_s.device,
        )
        hier = evaluate_hierarchy(
            vis_encoder_s, encoder_s, fine_bb, coarse_bb,
            fine2coarse_tensor, c_eval, hyperbolic=is_hyp, tag=key,
        )

        all_results[key] = {
            'd': d_sweep, 'mode': mode,
            'MRR': lp['MRR'], 'Hits@1': lp['Hits@1'], 'Hits@10': lp['Hits@10'],
            'R@1': xm['R@1'], 'R@5': xm['R@5'], 'R@10': xm['R@10'],
            'Hierarchy': hier['accuracy'], 'MeanDepth': hier['mean_diff'],
            'P1_loss': history_p1_s['loss_total'][-1],
            'P1_time': p1_time, 'P2_time': p2_time,
            'params': n_params,
        }

        print(f'\\n  d={d_sweep} {mode}: MRR={lp["MRR"]:.4f}  H@10={lp["Hits@10"]:.4f}  '
              f'R@5={xm["R@5"]:.4f}  Hier={hier["accuracy"]:.0%}  '
              f'P1={p1_time/60:.1f}m  P2={p2_time/60:.1f}m')

        del manifold_s, encoder_s, vis_encoder_s, comp_op_s, rel_maps_s
        del history_p1_s, history_p2_s, fine_z_s
        gc.collect()
        torch.cuda.empty_cache()

_sweep_total = _time.time() - _sweep_start
print(f'\\n\\nSweep complete in {_sweep_total/60:.1f} min ({_sweep_total/3600:.1f} hrs)')
"""

RESULTS_CODE = """\
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

DIMS = [32, 64, 128, 256, 512]
METRICS = [
    ('MRR', 'MRR (Link Prediction)', True),
    ('Hits@10', 'Hits@10 (Link Prediction)', True),
    ('R@5', 'R@5 (Cross-modal Retrieval)', True),
    ('Hierarchy', 'Hierarchy Accuracy', True),
    ('MeanDepth', 'Mean Depth Diff (fine - coarse)', True),
    ('P1_loss', 'Final P1 Loss', False),
]

# ── Table ──
print('=' * 80)
print('  DIMENSIONALITY SWEEP — HYPERBOLIC vs EUCLIDEAN')
print('=' * 80)
header = f'{\"d\":>5s}  {\"Geo\":>5s}  {\"Params\":>10s}  {\"MRR\":>7s}  {\"H@10\":>7s}  {\"R@5\":>7s}  {\"Hier\":>7s}  {\"Depth\":>7s}  {\"P1loss\":>7s}'
print(header)
print('-' * 80)
for d in DIMS:
    for m in ['HYP', 'EUCL']:
        r = all_results[f'd{d}_{m}']
        print(f'{d:5d}  {m:>5s}  {r["params"]:10,d}  '
              f'{r["MRR"]:7.4f}  {r["Hits@10"]:7.4f}  '
              f'{r["R@5"]:7.4f}  {r["Hierarchy"]:7.2%}  '
              f'{r["MeanDepth"]:7.4f}  {r["P1_loss"]:7.4f}')
    delta_mrr = all_results[f'd{d}_HYP']['MRR'] - all_results[f'd{d}_EUCL']['MRR']
    marker = '<<< HYP wins' if delta_mrr > 0.005 else ('>>> EUCL wins' if delta_mrr < -0.005 else '    ~tie')
    print(f'       delta MRR = {delta_mrr:+.4f}  {marker}')

# ── Per-dim winner summary ──
print('\\n' + '=' * 80)
print('  PER-DIMENSION WINNER (by MRR)')
print('=' * 80)
for d in DIMS:
    h = all_results[f'd{d}_HYP']['MRR']
    e = all_results[f'd{d}_EUCL']['MRR']
    winner = 'HYPERBOLIC' if h > e else 'EUCLIDEAN'
    pct = abs(h - e) / max(e, 1e-8) * 100
    print(f'  d={d:3d}:  {winner:12s}  (MRR {max(h,e):.4f} vs {min(h,e):.4f},  +{pct:.1f}%)')

# ── Plots ──
fig, axes = plt.subplots(2, 3, figsize=(18, 10))

for ax, (metric_key, title, higher_better) in zip(axes.flat, METRICS):
    hyp_vals = [all_results[f'd{d}_HYP'][metric_key] for d in DIMS]
    eucl_vals = [all_results[f'd{d}_EUCL'][metric_key] for d in DIMS]

    ax.plot(DIMS, hyp_vals, 'o-', linewidth=2, markersize=8,
            label='Hyperbolic', color='#2196F3')
    ax.plot(DIMS, eucl_vals, 's--', linewidth=2, markersize=8,
            label='Euclidean', color='#FF9800')

    for i, d in enumerate(DIMS):
        better = hyp_vals[i] > eucl_vals[i] if higher_better else hyp_vals[i] < eucl_vals[i]
        if better:
            ax.annotate('*', (d, hyp_vals[i]), fontsize=16, ha='center',
                        va='bottom', color='#2196F3', fontweight='bold')

    ax.set_xlabel('Embedding Dimension d')
    ax.set_ylabel(metric_key)
    ax.set_title(title, fontsize=11, fontweight='bold')
    ax.set_xscale('log', base=2)
    ax.set_xticks(DIMS)
    ax.set_xticklabels([str(d) for d in DIMS])
    ax.legend()
    ax.grid(alpha=0.3)

fig.suptitle('USM v2: Hyperbolic vs Euclidean across Dimensions',
             fontsize=14, fontweight='bold', y=1.02)
plt.tight_layout()
out_png = os.path.join(CKPT_DIR, 'usm_v2_dim_sweep.png')
plt.savefig(out_png, dpi=150, bbox_inches='tight')
plt.show()
print(f'\\nPlot saved: {out_png}')
"""

SAVE_CODE = """\
import json as _json

sweep_path = os.path.join(CKPT_DIR, 'usm_v2_sweep_results.json')
_serializable = {}
for k, v in all_results.items():
    _serializable[k] = {sk: float(sv) if isinstance(sv, (float, int)) else sv for sk, sv in v.items()}
with open(sweep_path, 'w') as f:
    _json.dump(_serializable, f, indent=2)
print(f'Results saved: {sweep_path}')

print(f'\\nTotal sweep time: {(_time.time() - _run_start)/3600:.1f} hours')
"""


def main():
    library_header = """\
# =============================================================================
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
"""

    parts = [library_header]
    for mod in MODULES:
        parts.append(f"\n\n# ----- {mod} -----\n")
        parts.append(read_module(mod))
    library_code = "".join(parts)

    cells = [
        cell_md(TITLE_MD),
        cell_code(INSTALL_CODE),
        cell_md("## Library (all usm_v2 code inlined)"),
        cell_code(library_code),
        cell_md("## Configuration"),
        cell_code(CONFIG_CODE),
        cell_md("## 1. Load Data"),
        cell_code(DATA_CODE),
        cell_md(
            "## 2. Pre-cache Backbone Embeddings\n\n"
            "Text backbone (MiniLM, 384-dim) and CLIP (512-dim) are frozen — "
            "their outputs are dimension-independent and reused for **all** sweep runs."
        ),
        cell_code(PRECACHE_CODE),
        cell_md("## 3. Pre-cache CLIP (CIFAR-100)"),
        cell_code(CLIP_CODE),
        cell_md(
            "## 4. Dimensionality Sweep\n\n"
            "For each `d ∈ {32, 64, 128, 256, 512}`, trains:\n"
            "- **Hyperbolic**: scheduled per-relation curvature warmup\n"
            "- **Euclidean**: identical architecture, flat geometry (c=0)\n\n"
            "35 P1 epochs + 20 P2 epochs per run. "
            "Both share the same pre-cached backbone embeddings."
        ),
        cell_code(SWEEP_CODE),
        cell_md(
            "## 5. Results: Hyperbolic vs Euclidean\n\n"
            "The thesis: **hyperbolic geometry achieves the same quality in "
            "far fewer dimensions**. At low d, the exponential volume of "
            "hyperbolic space gives it a decisive advantage for encoding "
            "hierarchical structure. At high d, both converge because "
            "Euclidean space has enough capacity."
        ),
        cell_code(RESULTS_CODE),
        cell_md("## 6. Save Results"),
        cell_code(SAVE_CODE),
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

    out_dir = ROOT / "notebooks"
    out_dir.mkdir(exist_ok=True)
    out_path = out_dir / "usm_v2_sweep.ipynb"
    out_path.write_text(json.dumps(nb, indent=1), encoding="utf-8")
    print(f"Wrote {out_path} ({out_path.stat().st_size // 1024} KB, {len(cells)} cells)")
    print(f"Library cell: {len(library_code.splitlines())} lines")


if __name__ == "__main__":
    main()
