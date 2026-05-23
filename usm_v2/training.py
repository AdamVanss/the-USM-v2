"""
USM v2 — Training loops.

Phase 1: text-only (ConceptNet + cross-lingual + SNLI) with full curriculum,
         learnable curvature, and Riemannian gradient control.
Phase 2: multimodal (CIFAR-100 vision ↔ text) with hierarchy objectives.

The epoch loop is structured as:
  1. Curriculum sampler selects available triples
  2. Burn-in scheduler sets effective curvature
  3. Loss weights are scheduled by curriculum
  4. Gradients are rescaled by conformal factor
  5. Riemannian clipping prevents boundary explosion
"""

import os
import time
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from typing import Dict, List, Optional

import geoopt

from .config import USMConfig
from .manifold import LearnablePoincareBall, logmap0, expmap0, clamp_to_ball, dist0, EPS
from .encoders import ConceptEncoder, VisionEncoder
from .operators import CompositionalOperator, RelationMaps, REL2IDX, N_RELATIONS, RELATIONS
from .losses import loss_cl, loss_rel, loss_entailment, loss_crossmodal, loss_hierarchy
from .gradient_control import (
    scale_riemannian_grads,
    clip_riemannian_grad_norm,
    BurninScheduler,
)
from .curriculum import (
    compute_difficulty_scores,
    CurriculumSampler,
    LossWeightScheduler,
    sample_negatives_gpu,
)
from .data import (
    CIFAR100_FINE, CIFAR100_COARSE, CIFAR100_FINE2COARSE,
)


def _precache_text_embeddings(
    encoder: ConceptEncoder,
    texts: List[str],
    batch_size: int = 512,
    device: torch.device = torch.device("cpu"),
) -> torch.Tensor:
    """Pre-encode texts through the frozen backbone (run once, reuse)."""
    all_embs = []
    for i in tqdm(range(0, len(texts), batch_size), desc="Pre-caching text", leave=False):
        emb = encoder.encode_backbone(texts[i:i + batch_size])
        all_embs.append(emb.cpu())
    return torch.cat(all_embs, dim=0).to(device)


def _precache_clip_embeddings(
    vis_encoder: VisionEncoder,
    dataloader: DataLoader,
    device: torch.device = torch.device("cpu"),
    max_images: int = 50000,
) -> tuple:
    """Pre-encode CIFAR images through the frozen CLIP backbone."""
    all_embs, all_labels = [], []
    n = 0
    for imgs, labels in tqdm(dataloader, desc="Pre-caching CLIP", leave=False):
        emb = vis_encoder.encode_images(imgs.to(device))
        all_embs.append(emb.cpu())
        all_labels.append(labels)
        n += imgs.shape[0]
        if n >= max_images:
            break
    return torch.cat(all_embs, dim=0).to(device), torch.cat(all_labels, dim=0).to(device)


# ---------------------------------------------------------------------------
# Phase 1: Text-only training
# ---------------------------------------------------------------------------

def train_phase1(
    cfg: USMConfig,
    manifold: LearnablePoincareBall,
    encoder: ConceptEncoder,
    comp_op: CompositionalOperator,
    rel_maps: RelationMaps,
    triples: List,
    cl_pairs: List,
    snli_pairs: List,
    vocab_list: List[str],
    vocab_bb: torch.Tensor,
    concept2bb: Dict[str, int],
    cl_bb_src: Optional[torch.Tensor] = None,
    cl_bb_tgt: Optional[torch.Tensor] = None,
    snli_bb_a: Optional[torch.Tensor] = None,
    snli_bb_b: Optional[torch.Tensor] = None,
    snli_labels_t: Optional[torch.Tensor] = None,
) -> Dict[str, List]:
    """
    Phase 1 — GPU-native training loop.

    All triple data is pre-converted to integer index tensors on GPU.
    No string operations, dict lookups, or CPU-side DataLoader in the hot loop.
    """
    device = cfg.device
    history = {
        "loss_total": [], "loss_rel": [], "loss_comp": [],
        "loss_cl": [], "loss_ent": [], "loss_hier": [],
        "curvature": [], "curvature_per_rel": [], "lr": [], "curriculum_pct": [],
    }

    # ---- Pre-convert ALL triples to GPU index tensors (once) ----
    all_h_idx = torch.tensor([concept2bb[h] for h, r, t in triples], device=device)
    all_r_idx = torch.tensor([REL2IDX[r] for h, r, t in triples], device=device)
    all_t_idx = torch.tensor([concept2bb[t] for h, r, t in triples], device=device)
    n_vocab = len(vocab_list)

    difficulty_scores = compute_difficulty_scores(triples, vocab_list)
    curriculum = CurriculumSampler(
        triples, difficulty_scores, cfg.n_epochs_p1,
        full_data_pct=cfg.curriculum_full_data_pct,
        enabled=cfg.curriculum_enabled,
    )
    loss_scheduler = LossWeightScheduler(cfg.n_epochs_p1, enabled=cfg.curriculum_enabled)
    burnin = BurninScheduler(cfg.burnin_epochs, cfg.transition_epochs)

    n_cl = cl_bb_src.shape[0] if cl_bb_src is not None else 0
    n_snli = snli_bb_a.shape[0] if snli_bb_a is not None else 0

    # Curvature is scheduled, not learned — prevents optimizer collapse
    manifold._c_params.requires_grad_(False)
    c_targets = torch.tensor(
        cfg.c_targets[:N_RELATIONS] if hasattr(cfg, 'c_targets') else [cfg.c_init] * N_RELATIONS,
        device=device, dtype=torch.float32,
    )
    c_ws = getattr(cfg, 'c_warmup_start', 10)
    c_we = getattr(cfg, 'c_warmup_end', 40)

    manifold_ids = {id(p) for p in manifold.parameters()}
    model_params = []
    seen = set(manifold_ids)
    for p in (
        list(encoder.proj.parameters())
        + ([encoder.mu0] if hasattr(encoder, "mu0") else [])
        + list(comp_op.parameters())
        + list(rel_maps.parameters())
    ):
        pid = id(p)
        if pid not in seen:
            seen.add(pid)
            model_params.append(p)

    optimizer = geoopt.optim.RiemannianAdam(
        [{"params": model_params, "lr": cfg.lr_p1}],
        weight_decay=cfg.weight_decay,
    )

    base_weights = {
        "L_rel": cfg.w_rel, "L_comp": cfg.w_comp,
        "L_cl": cfg.w_cl, "L_ent": cfg.w_ent, "L_hier": cfg.w_hier,
    }

    ISA_IDX = REL2IDX.get("IS_A", 0)
    PARTOF_IDX = REL2IDX.get("PART_OF", 2)

    e0_idx = curriculum.get_epoch_indices(0)
    e0_batches = len(e0_idx) // cfg.batch_size
    tgt_str = " ".join(f"{v:.4f}" for v in c_targets.tolist())
    print(
        f'Phase 1 starting: {cfg.n_epochs_p1} epochs | {len(triples):,} triples | '
        f'batch={cfg.batch_size} | d={cfg.d} | device={device}\n'
        f'  Curvature targets: [{tgt_str}]\n'
        f'  Relations:         {" ".join(RELATIONS)}\n'
        f'  Warmup: E{c_ws}→E{c_we} (linear ramp per relation)\n'
        f'  Epoch 0: {len(e0_idx):,} triples (~{e0_batches} batches)\n'
        f'  GPU-native loop + cosine LR schedule',
        flush=True,
    )

    for epoch in range(cfg.n_epochs_p1):
        t0 = time.time()
        encoder.train()
        comp_op.train()
        rel_maps.train()

        # Scheduled curvature warmup: linear ramp from c_init to c_targets
        with torch.no_grad():
            if epoch < c_ws:
                progress = 0.0
            elif epoch >= c_we:
                progress = 1.0
            else:
                progress = (epoch - c_ws) / max(c_we - c_ws, 1)
            manifold._c_params.copy_(cfg.c_init + (c_targets - cfg.c_init) * progress)

        # Cosine LR schedule
        cos_lr = cfg.lr_p1 * 0.5 * (1 + math.cos(math.pi * epoch / cfg.n_epochs_p1))
        optimizer.param_groups[0]["lr"] = cos_lr

        c_scale = burnin.get_curvature_scale(epoch)

        epoch_idx = curriculum.get_epoch_indices(epoch).to(device)
        n_epoch = len(epoch_idx)
        weights = loss_scheduler.get_weights(epoch, base_weights)
        cpct = curriculum.get_progress(epoch)

        # Shuffle this epoch's triples on GPU
        perm = torch.randperm(n_epoch, device=device)
        ep_h = all_h_idx[epoch_idx[perm]]
        ep_r = all_r_idx[epoch_idx[perm]]
        ep_t = all_t_idx[epoch_idx[perm]]

        c_base_detached = manifold.c.detach()
        all_z = (
            encoder.project(vocab_bb, c=c_base_detached).detach()
            if epoch > 5 else None
        )

        epoch_loss = 0.0
        epoch_losses = {k: 0.0 for k in history if k.startswith("loss_")}
        n_batches = 0
        BS = cfg.batch_size
        n_total_batches = n_epoch // BS

        batch_iter = tqdm(
            range(0, n_epoch - BS + 1, BS),
            desc=f'P1 E{epoch:02d}',
            total=n_total_batches,
            leave=(epoch == cfg.n_epochs_p1 - 1),
        )
        for start in batch_iter:
            c_base = manifold.effective_c(c_scale)

            h_idx = ep_h[start:start + BS]
            t_idx = ep_t[start:start + BS]
            rels = ep_r[start:start + BS]

            z_h = encoder.project(vocab_bb[h_idx], c=c_base)
            z_t = encoder.project(vocab_bb[t_idx], c=c_base)

            neg_idx = sample_negatives_gpu(
                BS, n_vocab, epoch, cfg.n_epochs_p1, device,
                all_z=all_z,
                z_pred=None,
                t_idx=t_idx,
                hard_neg_max_ratio=cfg.hard_neg_max_ratio,
            )
            z_neg = encoder.project(vocab_bb[neg_idx], c=c_base)

            # --- Per-relation loss: each relation uses its own curvature ---
            L_rel_val = z_h.new_zeros(())
            L_hier_val = z_h.new_zeros(())
            rel_count = 0
            hier_count = 0
            for r in range(N_RELATIONS):
                mask = (rels == r)
                n_r = mask.sum().item()
                if n_r < 2:
                    continue
                c_r = manifold.effective_c_rel(r, c_scale)
                z_pred_r = rel_maps(z_h[mask], rels[mask], c=c_r)
                L_r = loss_rel(z_pred_r, z_t[mask], z_neg[mask], c_r, margin=cfg.margin_rel)
                L_rel_val = L_rel_val + L_r * n_r
                rel_count += n_r

                if r in (ISA_IDX, PARTOF_IDX) and c_r.item() > EPS and weights.get("L_hier", 0) > 0:
                    depth_h_r = dist0(z_h[mask], c_r)
                    depth_t_r = dist0(z_t[mask], c_r)
                    L_hier_val = L_hier_val + F.relu(depth_t_r - depth_h_r + cfg.margin_hier).mean() * n_r
                    hier_count += n_r

            if rel_count > 0:
                L_rel_val = L_rel_val / rel_count
            if hier_count > 0:
                L_hier_val = L_hier_val / hier_count

            z_comp = comp_op(z_h, rels, z_t, c=c_base)
            if c_base.item() > EPS:
                L_comp_val = logmap0(z_comp, c_base).pow(2).mean()
            else:
                L_comp_val = z_comp.pow(2).mean()

            L_cl_val = z_h.new_zeros(())
            if n_cl > 0 and weights.get("L_cl", 0) > 0:
                n_cl_batch = min(BS // 2, n_cl)
                idx_cl = torch.randint(0, n_cl, (n_cl_batch,), device=device)
                z_src = encoder.project(cl_bb_src[idx_cl], c=c_base)
                z_tgt = encoder.project(cl_bb_tgt[idx_cl], c=c_base)
                L_cl_val = loss_cl(z_src, z_tgt, c_base)

            L_ent_val = z_h.new_zeros(())
            if n_snli > 0 and weights.get("L_ent", 0) > 0:
                n_ent_batch = min(BS // 2, n_snli)
                idx_sn = torch.randint(0, n_snli, (n_ent_batch,), device=device)
                z_a_ent = encoder.project(snli_bb_a[idx_sn], c=c_base)
                z_b_ent = encoder.project(snli_bb_b[idx_sn], c=c_base)
                L_ent_val = loss_entailment(
                    z_a_ent, z_b_ent,
                    snli_labels_t[idx_sn],
                    c_base, delta_c=cfg.delta_contradiction,
                )

            total = (
                weights["L_rel"] * L_rel_val
                + weights["L_comp"] * L_comp_val
                + weights.get("L_cl", 0) * L_cl_val
                + weights.get("L_ent", 0) * L_ent_val
                + weights.get("L_hier", 0) * L_hier_val
            )

            total.backward()

            if c_base.item() > EPS and cfg.use_riemannian_clipping:
                manifold_aware = [p for p in model_params if p.grad is not None]
                scale_riemannian_grads(manifold_aware, c_base)
                clip_riemannian_grad_norm(manifold_aware, cfg.grad_clip_norm, c_base)
            else:
                torch.nn.utils.clip_grad_norm_(model_params, cfg.grad_clip_norm)

            optimizer.step()
            optimizer.zero_grad()

            epoch_loss += total.item()
            epoch_losses["loss_rel"] += L_rel_val.item()
            epoch_losses["loss_comp"] += L_comp_val.item()
            epoch_losses["loss_cl"] += L_cl_val.item()
            epoch_losses["loss_ent"] += L_ent_val.item()
            epoch_losses["loss_hier"] += L_hier_val.item()
            n_batches += 1

            if n_batches == 1 or n_batches % 25 == 0:
                batch_iter.set_postfix(
                    loss=f'{total.item():.3f}',
                    c_base=f'{c_base.item():.3f}',
                )

        nb = max(n_batches, 1)
        history["loss_total"].append(epoch_loss / nb)
        history["loss_rel"].append(epoch_losses["loss_rel"] / nb)
        history["loss_comp"].append(epoch_losses["loss_comp"] / nb)
        history["loss_cl"].append(epoch_losses["loss_cl"] / nb)
        history["loss_ent"].append(epoch_losses["loss_ent"] / nb)
        history["loss_hier"].append(epoch_losses["loss_hier"] / nb)
        c_per_rel = [manifold.c_rel(r).item() for r in range(N_RELATIONS)]
        history["curvature"].append(max(c_per_rel))
        history["curvature_per_rel"].append(c_per_rel)
        history["lr"].append(cos_lr)
        history["curriculum_pct"].append(cpct)

        dt = time.time() - t0
        c_rel_str = " ".join(f"{v:.3f}" for v in c_per_rel)
        print(
            f"[P1 E{epoch:02d}] "
            f"loss={epoch_loss / nb:.4f}  "
            f"c=[{c_rel_str}]  "
            f"lr={cos_lr:.1e}  "
            f"data={n_epoch:,}/{len(triples):,}  "
            f"curriculum={cpct:.0%}  "
            f"({dt:.1f}s)",
            flush=True,
        )

        if cfg.ckpt_dir and (epoch + 1) % cfg.ckpt_every == 0:
            os.makedirs(cfg.ckpt_dir, exist_ok=True)
            torch.save({
                "epoch": epoch,
                "encoder": encoder.state_dict(),
                "comp_op": comp_op.state_dict(),
                "rel_maps": rel_maps.state_dict(),
                "manifold": manifold.state_dict(),
                "optimizer": optimizer.state_dict(),
                "config": cfg,
            }, os.path.join(cfg.ckpt_dir, f"p1_epoch_{epoch:03d}.pt"))

    return history


# ---------------------------------------------------------------------------
# Phase 2: Multimodal training
# ---------------------------------------------------------------------------

def train_phase2(
    cfg: USMConfig,
    manifold: LearnablePoincareBall,
    encoder: ConceptEncoder,
    vis_encoder: VisionEncoder,
    precomp_clip: torch.Tensor,
    precomp_labels: torch.Tensor,
    fine_bb: torch.Tensor,
    coarse_bb: torch.Tensor,
    fine2coarse_tensor: torch.Tensor,
) -> Dict[str, List]:
    """
    Phase 2: align vision with frozen text geometry + hierarchy.

    Curvature continues from Phase 1 (already ramped up).
    Gradient control remains active.
    """
    device = cfg.device
    history = {
        "loss_total": [], "loss_xmodal": [], "loss_hier": [],
        "curvature": [],
    }

    manifold_ids = {id(p) for p in manifold.parameters()}
    manifold_params = [p for p in manifold.parameters() if p.requires_grad] if cfg.learnable_curvature else []

    seen = set(id(p) for p in manifold_params) | manifold_ids
    vis_params = []
    for p in list(vis_encoder.proj.parameters()) + ([vis_encoder.mu0] if hasattr(vis_encoder, "mu0") else []):
        if id(p) not in seen:
            seen.add(id(p))
            vis_params.append(p)

    text_params = []
    for p in list(encoder.proj.parameters()) + ([encoder.mu0] if hasattr(encoder, "mu0") else []):
        if id(p) not in seen:
            seen.add(id(p))
            text_params.append(p)

    param_groups = [
        {"params": vis_params, "lr": cfg.lr_p2_vis},
        {"params": text_params, "lr": cfg.lr_p2_text},
    ]
    if manifold_params:
        param_groups.append({"params": manifold_params, "lr": cfg.c_lr * 0.1})

    optimizer = geoopt.optim.RiemannianAdam(param_groups, weight_decay=cfg.weight_decay)

    N = precomp_clip.shape[0]
    p2_batches = (N + cfg.p2_batch - 1) // cfg.p2_batch
    print(
        f'Phase 2 starting: {cfg.n_epochs_p2} epochs | {N:,} images | '
        f'batch={cfg.p2_batch} (~{p2_batches} batches/epoch) | device={device}',
        flush=True,
    )

    for epoch in range(cfg.n_epochs_p2):
        t0 = time.time()
        vis_encoder.train()
        encoder.train()

        perm = torch.randperm(N, device=device)
        epoch_loss = 0.0
        n_batches = 0

        batch_starts = range(0, N, cfg.p2_batch)
        batch_iter = tqdm(batch_starts, desc=f'P2 E{epoch:02d}', leave=(epoch == cfg.n_epochs_p2 - 1))
        for start in batch_iter:
            c_eff = manifold.c

            idx = perm[start:start + cfg.p2_batch]
            clip_emb = precomp_clip[idx]
            labels = precomp_labels[idx]

            z_img = vis_encoder.project(clip_emb, c=c_eff)

            fine_labels = labels
            coarse_labels = fine2coarse_tensor[labels]
            z_fine_text = encoder.project(fine_bb[fine_labels], c=c_eff)
            z_coarse_text = encoder.project(coarse_bb[coarse_labels], c=c_eff)

            L_xm = loss_crossmodal(z_fine_text, z_img, c_eff)
            L_hier = loss_hierarchy(z_fine_text, z_coarse_text, c_eff, margin=cfg.margin_hier)

            total = L_xm + cfg.w_hier * L_hier
            total.backward()

            all_params = vis_params + text_params
            if c_eff.item() > EPS and cfg.use_riemannian_clipping:
                manifold_aware = [p for p in all_params if p.grad is not None]
                scale_riemannian_grads(manifold_aware, c_eff)
                clip_riemannian_grad_norm(manifold_aware, cfg.grad_clip_norm, c_eff)
            else:
                torch.nn.utils.clip_grad_norm_(all_params, cfg.grad_clip_norm)

            optimizer.step()
            optimizer.zero_grad()

            epoch_loss += total.item()
            n_batches += 1

            if n_batches == 1 or n_batches % 10 == 0:
                batch_iter.set_postfix(loss=f'{total.item():.3f}')

        nb = max(n_batches, 1)
        history["loss_total"].append(epoch_loss / nb)
        history["loss_xmodal"].append(0.0)
        history["loss_hier"].append(0.0)
        with torch.no_grad():
            c_val = manifold.c.item()
        history["curvature"].append(c_val)

        dt = time.time() - t0
        print(
            f"[P2 E{epoch:02d}] "
            f"loss={epoch_loss / nb:.4f}  "
            f"c={c_val:.4f}  "
            f"({dt:.1f}s)",
            flush=True,
        )

        if cfg.ckpt_dir and (epoch + 1) % cfg.ckpt_every == 0:
            os.makedirs(cfg.ckpt_dir, exist_ok=True)
            torch.save({
                "epoch": epoch,
                "encoder": encoder.state_dict(),
                "vis_encoder": vis_encoder.state_dict(),
                "manifold": manifold.state_dict(),
                "optimizer": optimizer.state_dict(),
                "config": cfg,
            }, os.path.join(cfg.ckpt_dir, f"p2_epoch_{epoch:03d}.pt"))

    return history
