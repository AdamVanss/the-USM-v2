"""
USM v2 — Evaluation functions.

All metrics accept an explicit curvature tensor for consistency with
the learnable manifold.
"""

import numpy as np
import torch
from tqdm.auto import tqdm
from typing import List, Tuple, Dict, Optional

from .manifold import poincare_cdist, eucl_cdist, dist0, EPS
from .operators import REL2IDX


def evaluate_link_prediction(
    encoder,
    rel_maps,
    test_triples: List[Tuple[str, str, str]],
    vocab_list: List[str],
    vocab_bb: torch.Tensor,
    concept2bb: Dict[str, int],
    c: torch.Tensor,
    hyperbolic: bool = True,
    manifold=None,
    tag: str = "",
    max_eval: int = 2000,
    device: torch.device = torch.device("cpu"),
) -> Dict[str, float]:
    """KG link prediction with per-relation curvature when manifold is provided."""
    encoder.eval()
    rel_maps.eval()

    all_z = []
    with torch.no_grad():
        for i in range(0, len(vocab_list), 1024):
            batch_idx = torch.tensor(
                [concept2bb[c_name] for c_name in vocab_list[i:i + 1024]],
                device=device, dtype=torch.long
            )
            z_b = encoder.project(vocab_bb[batch_idx], c=c)
            all_z.append(z_b)
    all_z = torch.cat(all_z, dim=0)
    concept2idx = {c_name: i for i, c_name in enumerate(vocab_list)}

    ranks = []
    for h, r, t in tqdm(test_triples[:max_eval], desc=f"LinkPred {tag}", leave=False):
        if h not in concept2idx or t not in concept2idx:
            continue
        r_idx_val = REL2IDX[r]
        r_idx = torch.tensor([r_idx_val], device=device)

        c_r = manifold.c_rel(r_idx_val) if (manifold is not None and hyperbolic) else c

        z_h = all_z[concept2idx[h]].unsqueeze(0)
        with torch.no_grad():
            z_pred = rel_maps(z_h, r_idx, c=c_r)
            if hyperbolic and c_r.item() > EPS:
                dists = poincare_cdist(z_pred, all_z, c_r).squeeze(0)
            else:
                dists = eucl_cdist(z_pred, all_z).squeeze(0)
        rank = (dists < dists[concept2idx[t]]).sum().item() + 1
        ranks.append(rank)

    if not ranks:
        return {"MRR": 0, "Hits@1": 0, "Hits@10": 0, "n": 0}
    ranks = np.array(ranks)
    return {
        "MRR": float(np.mean(1.0 / ranks)),
        "Hits@1": float(np.mean(ranks <= 1)),
        "Hits@10": float(np.mean(ranks <= 10)),
        "n": len(ranks),
    }


def evaluate_crossmodal(
    vis_encoder,
    text_z: torch.Tensor,
    precomp_clip: torch.Tensor,
    precomp_lbl: torch.Tensor,
    c: torch.Tensor,
    hyperbolic: bool = True,
    tag: str = "",
    max_images: int = 2000,
    device: torch.device = torch.device("cpu"),
) -> Dict[str, float]:
    """Cross-modal retrieval: Image -> Text on CIFAR-100 test set."""
    vis_encoder.eval()

    cdist_fn = poincare_cdist if (hyperbolic and c.item() > EPS) else eucl_cdist

    n_eval = min(max_images, precomp_clip.shape[0])
    perm = torch.randperm(precomp_clip.shape[0])[:n_eval]

    with torch.no_grad():
        z_img = vis_encoder.project(precomp_clip[perm], c=c)
        labels = precomp_lbl[perm]

    ranks = []
    with torch.no_grad():
        if hyperbolic and c.item() > EPS:
            dists = cdist_fn(z_img, text_z, c)
        else:
            dists = cdist_fn(z_img, text_z)

        for i in range(n_eval):
            target = labels[i].item()
            rank = (dists[i] < dists[i, target]).sum().item() + 1
            ranks.append(rank)

    ranks = np.array(ranks)
    return {
        "R@1": float(np.mean(ranks <= 1)),
        "R@5": float(np.mean(ranks <= 5)),
        "R@10": float(np.mean(ranks <= 10)),
        "MedR": float(np.median(ranks)),
    }


def evaluate_hierarchy(
    vis_encoder,
    text_encoder,
    fine_bb: torch.Tensor,
    coarse_bb: torch.Tensor,
    fine2coarse: torch.Tensor,
    c: torch.Tensor,
    hyperbolic: bool = True,
    tag: str = "",
) -> Dict[str, object]:
    """
    Hierarchy depth test: fine-class embeddings should be deeper than coarse.
    """
    vis_encoder.eval()
    text_encoder.eval()

    if hyperbolic and c.item() > EPS:
        dist0_fn = lambda z: dist0(z, c)
    else:
        dist0_fn = lambda z: z.norm(dim=-1)

    with torch.no_grad():
        fine_z = text_encoder.project(fine_bb, c=c)
        coarse_z = text_encoder.project(coarse_bb, c=c)

    correct = 0
    total = 0
    depth_diffs = []
    for fine_idx in range(fine_z.shape[0]):
        coarse_idx = fine2coarse[fine_idx].item() if isinstance(fine2coarse, torch.Tensor) else fine2coarse[fine_idx]
        d_fine = dist0_fn(fine_z[fine_idx].unsqueeze(0)).item()
        d_coarse = dist0_fn(coarse_z[coarse_idx].unsqueeze(0)).item()
        if d_fine > d_coarse:
            correct += 1
        depth_diffs.append(d_fine - d_coarse)
        total += 1

    return {
        "accuracy": correct / max(total, 1),
        "mean_diff": float(np.mean(depth_diffs)) if depth_diffs else 0.0,
        "fine_depths": [dist0_fn(fine_z[i].unsqueeze(0)).item() for i in range(fine_z.shape[0])],
        "coarse_depths": [dist0_fn(coarse_z[i].unsqueeze(0)).item() for i in range(coarse_z.shape[0])],
    }
