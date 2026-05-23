"""
USM v2 — Curriculum training.

Four mechanisms that work together to progressively increase difficulty:
  1. Concept difficulty scoring from IS_A hierarchy depth
  2. Progressive triple sampling (easy → hard)
  3. Loss weight scheduling (ramp losses in over training)
  4. Hard negative mining with increasing ratio
"""

import random
import torch
from collections import defaultdict
from typing import List, Tuple, Dict, Optional


# ---------------------------------------------------------------------------
# 1. Difficulty scoring
# ---------------------------------------------------------------------------

def compute_difficulty_scores(
    triples: List[Tuple[str, str, str]],
    vocab_list: List[str],
) -> Dict[str, float]:
    """
    Score each concept by its depth in the IS_A hierarchy.

    Root concepts (e.g., 'animal', 'object') get low scores (easy).
    Leaf concepts (e.g., 'golden retriever') get high scores (hard).
    Concepts not in the IS_A tree get a default mid-range score.

    Returns:
        dict mapping concept -> difficulty in [0, 1]
    """
    children = defaultdict(set)
    parents = defaultdict(set)

    for h, r, t in triples:
        if r == "IS_A":
            children[t].add(h)
            parents[h].add(t)

    roots = set()
    for concept in set(children.keys()) | set(parents.keys()):
        if concept not in parents:
            roots.add(concept)

    depths: Dict[str, int] = {}
    queue = list(roots)
    for r in roots:
        depths[r] = 0

    while queue:
        node = queue.pop(0)
        d = depths[node]
        for child in children.get(node, []):
            if child not in depths or depths[child] < d + 1:
                depths[child] = d + 1
                queue.append(child)

    if not depths:
        return {c: 0.5 for c in vocab_list}

    max_depth = max(depths.values()) if depths else 1
    max_depth = max(max_depth, 1)

    scores = {}
    for c in vocab_list:
        if c in depths:
            scores[c] = depths[c] / max_depth
        else:
            scores[c] = 0.5

    return scores


def score_triple(triple: Tuple[str, str, str],
                 concept_scores: Dict[str, float]) -> float:
    """Difficulty of a triple = max difficulty of its head and tail."""
    h, _, t = triple
    return max(concept_scores.get(h, 0.5), concept_scores.get(t, 0.5))


# ---------------------------------------------------------------------------
# 2. Progressive triple sampling
# ---------------------------------------------------------------------------

class CurriculumSampler:
    """
    Controls which triples are available each epoch.

    Early epochs: only low-difficulty triples (broad, abstract concepts).
    Later epochs: progressively include harder/more specific triples.
    By `full_data_pct` of total epochs, all triples are included.
    """

    def __init__(self, triples: List[Tuple[str, str, str]],
                 concept_scores: Dict[str, float],
                 total_epochs: int,
                 full_data_pct: float = 0.7,
                 enabled: bool = True):
        self.enabled = enabled
        self.total_epochs = total_epochs
        self.full_data_pct = full_data_pct
        self.n_triples = len(triples)

        scored = [
            (i, t, score_triple(t, concept_scores))
            for i, t in enumerate(triples)
        ]
        scored.sort(key=lambda x: x[2])

        self._sorted_orig_idx = [i for i, t, s in scored]
        self._sorted_scores = [s for i, t, s in scored]
        self.scored_triples = [(t, s) for i, t, s in scored]

    def get_epoch_triples(self, epoch: int) -> List[Tuple[str, str, str]]:
        if not self.enabled:
            return [t for t, _ in self.scored_triples]

        threshold = min(1.0, (epoch + 1) / max(self.total_epochs * self.full_data_pct, 1))
        result = [t for t, s in self.scored_triples if s <= threshold]

        if len(result) < max(100, len(self.scored_triples) // 10):
            result = [t for t, _ in self.scored_triples[:max(100, len(self.scored_triples) // 10)]]

        return result

    def get_epoch_indices(self, epoch: int) -> torch.Tensor:
        """Return original triple indices for this epoch as a LongTensor."""
        if not self.enabled:
            return torch.arange(self.n_triples, dtype=torch.long)

        threshold = min(1.0, (epoch + 1) / max(self.total_epochs * self.full_data_pct, 1))
        selected = [
            self._sorted_orig_idx[i]
            for i, s in enumerate(self._sorted_scores)
            if s <= threshold
        ]

        min_count = max(100, self.n_triples // 10)
        if len(selected) < min_count:
            selected = self._sorted_orig_idx[:min_count]

        return torch.tensor(selected, dtype=torch.long)

    def get_progress(self, epoch: int) -> float:
        """Returns curriculum progress in [0, 1]."""
        return min(1.0, (epoch + 1) / max(self.total_epochs * self.full_data_pct, 1))


# ---------------------------------------------------------------------------
# 3. Loss weight scheduling
# ---------------------------------------------------------------------------

class LossWeightScheduler:
    """
    Gradually introduces harder losses over training.

    L_rel: always active (foundational KG structure)
    L_comp: ramps in over first 30% of training
    L_cl: ramps in over first 50%
    L_ent: starts at 20%, full at 60%
    L_hier: starts at 30%, full at 70%

    Base weights from config are multiplied by the schedule factor.
    """

    def __init__(self, total_epochs: int, enabled: bool = True):
        self.total_epochs = max(total_epochs, 1)
        self.enabled = enabled

    def get_weights(self, epoch: int, base_weights: Dict[str, float]) -> Dict[str, float]:
        if not self.enabled:
            return dict(base_weights)

        progress = epoch / self.total_epochs

        schedule = {
            "L_rel": 1.0,
            "L_comp": min(1.0, progress / 0.3) if progress > 0 else 0.0,
            "L_cl": min(1.0, progress / 0.5) if progress > 0 else 0.0,
            "L_ent": min(1.0, max(0, (progress - 0.2) / 0.4)),
            "L_hier": min(1.0, max(0, (progress - 0.3) / 0.4)),
        }

        return {
            k: base_weights.get(k, 1.0) * schedule.get(k, 1.0)
            for k in base_weights
        }


# ---------------------------------------------------------------------------
# 4. Hard negative mining
# ---------------------------------------------------------------------------

def sample_negatives(
    batch_size: int,
    vocab_list: List[str],
    epoch: int,
    total_epochs: int,
    all_z: Optional[torch.Tensor] = None,
    z_pred: Optional[torch.Tensor] = None,
    vocab2idx: Optional[Dict[str, int]] = None,
    exclude_idx: Optional[List[int]] = None,
    hard_neg_max_ratio: float = 0.8,
) -> List[str]:
    """
    Sample negative tails with increasing hard-negative ratio.

    Easy negatives: random vocab samples.
    Hard negatives: nearest non-target entities in current embedding space.
    """
    hard_ratio = min(hard_neg_max_ratio, epoch / max(total_epochs * 0.6, 1))
    n_hard = int(batch_size * hard_ratio)
    n_easy = batch_size - n_hard

    easy_negs = random.choices(vocab_list, k=n_easy)

    if n_hard > 0 and all_z is not None and z_pred is not None and vocab2idx is not None:
        with torch.no_grad():
            dists = torch.cdist(z_pred[:n_hard].float(), all_z.float(), p=2)

            if exclude_idx is not None:
                for i, eidx in enumerate(exclude_idx[:n_hard]):
                    if eidx is not None:
                        dists[i, eidx] = float("inf")

            hard_indices = dists.argmin(dim=-1).tolist()
            hard_negs = [vocab_list[idx] for idx in hard_indices]
    else:
        hard_negs = random.choices(vocab_list, k=n_hard)

    return easy_negs + hard_negs


def sample_negatives_gpu(
    batch_size: int,
    n_vocab: int,
    epoch: int,
    total_epochs: int,
    device: torch.device,
    all_z: Optional[torch.Tensor] = None,
    z_pred: Optional[torch.Tensor] = None,
    t_idx: Optional[torch.Tensor] = None,
    hard_neg_max_ratio: float = 0.8,
) -> torch.Tensor:
    """
    GPU-native negative sampling — returns vocab index tensor directly.

    No string operations, no CPU round-trips.
    """
    hard_ratio = min(hard_neg_max_ratio, epoch / max(total_epochs * 0.6, 1))
    n_hard = int(batch_size * hard_ratio)
    n_easy = batch_size - n_hard

    neg_idx = torch.randint(0, n_vocab, (batch_size,), device=device)

    if n_hard > 0 and all_z is not None and z_pred is not None:
        with torch.no_grad():
            dists = torch.cdist(z_pred[:n_hard].float(), all_z.float(), p=2)
            if t_idx is not None:
                dists[torch.arange(n_hard, device=device), t_idx[:n_hard]] = float("inf")
            neg_idx[:n_hard] = dists.argmin(dim=-1)

    return neg_idx
