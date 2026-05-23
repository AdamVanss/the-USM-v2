"""
USM v2 — Data loading.

Handles ConceptNet (S3 streaming + HuggingFace fallback + hardcoded fallback),
cross-lingual pairs (Tatoeba), SNLI, and CIFAR-100.
"""

import re
import random
from collections import defaultdict, Counter
from typing import List, Tuple, Dict

import torch
from torch.utils.data import Dataset, DataLoader

from .operators import RELATIONS, REL2IDX


# ---------------------------------------------------------------------------
# ConceptNet relation mapping
# ---------------------------------------------------------------------------

_CN_MAP = {
    "/r/IsA": "IS_A", "IsA": "IS_A", "is_a": "IS_A",
    "/r/Causes": "CAUSES", "Causes": "CAUSES", "causes": "CAUSES",
    "/r/PartOf": "PART_OF", "PartOf": "PART_OF", "part_of": "PART_OF",
    "/r/SimilarTo": "SIMILAR_TO", "SimilarTo": "SIMILAR_TO", "similar_to": "SIMILAR_TO",
    "/r/Antonym": "ANTONYM", "Antonym": "ANTONYM", "antonym": "ANTONYM",
    "/r/CapableOf": "CAPABLE_OF", "CapableOf": "CAPABLE_OF", "capable_of": "CAPABLE_OF",
}


def _norm_concept(s: str):
    s = str(s).strip()
    if "/c/en/" in s:
        s = s.split("/c/en/")[-1]
        s = s.split("/")[0]
    elif "/" in s:
        s = s.split("/")[-1]
    s = re.sub(r"[^a-z ]", "", s.replace("_", " ").lower()).strip()
    return s if len(s) > 1 else None


# ---------------------------------------------------------------------------
# Hardcoded fallback triples
# ---------------------------------------------------------------------------

_FALLBACK_TRIPLES = [
    ("dog", "IS_A", "animal"), ("cat", "IS_A", "animal"), ("wolf", "IS_A", "animal"),
    ("eagle", "IS_A", "bird"), ("robin", "IS_A", "bird"), ("penguin", "IS_A", "bird"),
    ("salmon", "IS_A", "fish"), ("shark", "IS_A", "fish"),
    ("rose", "IS_A", "flower"), ("tulip", "IS_A", "flower"), ("daisy", "IS_A", "flower"),
    ("oak", "IS_A", "tree"), ("pine", "IS_A", "tree"), ("maple", "IS_A", "tree"),
    ("hammer", "IS_A", "tool"), ("wrench", "IS_A", "tool"), ("saw", "IS_A", "tool"),
    ("violin", "IS_A", "instrument"), ("piano", "IS_A", "instrument"),
    ("fire", "CAUSES", "smoke"), ("fire", "CAUSES", "heat"),
    ("rain", "CAUSES", "flooding"), ("exercise", "CAUSES", "fatigue"),
    ("wheel", "PART_OF", "car"), ("engine", "PART_OF", "car"),
    ("leaf", "PART_OF", "tree"), ("branch", "PART_OF", "tree"),
    ("happy", "SIMILAR_TO", "joyful"), ("sad", "SIMILAR_TO", "unhappy"),
    ("fast", "SIMILAR_TO", "quick"), ("big", "SIMILAR_TO", "large"),
    ("hot", "ANTONYM", "cold"), ("fast", "ANTONYM", "slow"),
    ("love", "ANTONYM", "hate"), ("light", "ANTONYM", "dark"),
    ("dog", "CAPABLE_OF", "barking"), ("bird", "CAPABLE_OF", "flying"),
    ("fish", "CAPABLE_OF", "swimming"), ("human", "CAPABLE_OF", "thinking"),
    ("lion", "IS_A", "animal"), ("bear", "IS_A", "animal"), ("horse", "IS_A", "animal"),
    ("sparrow", "IS_A", "bird"), ("hawk", "IS_A", "bird"), ("owl", "IS_A", "bird"),
    ("lily", "IS_A", "flower"), ("orchid", "IS_A", "flower"),
    ("birch", "IS_A", "tree"), ("willow", "IS_A", "tree"),
    ("drum", "IS_A", "instrument"), ("flute", "IS_A", "instrument"),
    ("chair", "IS_A", "furniture"), ("table", "IS_A", "furniture"),
    ("soccer", "IS_A", "sport"), ("tennis", "IS_A", "sport"),
    ("lightning", "CAUSES", "fire"), ("virus", "CAUSES", "disease"),
    ("roof", "PART_OF", "house"), ("keyboard", "PART_OF", "computer"),
    ("angry", "SIMILAR_TO", "furious"), ("tiny", "SIMILAR_TO", "small"),
    ("rich", "ANTONYM", "poor"), ("strong", "ANTONYM", "weak"),
    ("horse", "CAPABLE_OF", "galloping"), ("eagle", "CAPABLE_OF", "soaring"),
]


def _augment_triples(triples):
    augmented = list(triples)
    for h, r, t in triples:
        if r in ("SIMILAR_TO", "ANTONYM"):
            augmented.append((t, r, h))
    isa_groups = defaultdict(list)
    for h, r, t in triples:
        if r == "IS_A":
            isa_groups[t].append(h)
    for parent, children in isa_groups.items():
        for i, c1 in enumerate(children):
            for c2 in children[i + 1:]:
                augmented.append((c1, "SIMILAR_TO", c2))
                augmented.append((c2, "SIMILAR_TO", c1))
    return augmented


# ---------------------------------------------------------------------------
# ConceptNet loading
# ---------------------------------------------------------------------------

def _load_conceptnet_s3(max_triples: int = 100_000):
    import urllib.request
    import gzip
    import io as _io

    url = "https://s3.amazonaws.com/conceptnet/downloads/2019/edges/conceptnet-assertions-5.7.0.csv.gz"
    print("  Streaming ConceptNet 5.7 from S3 ...")
    resp = urllib.request.urlopen(url, timeout=30)
    reader = _io.TextIOWrapper(gzip.GzipFile(fileobj=resp), encoding="utf-8")
    triples = []
    scanned = 0
    for line in reader:
        scanned += 1
        parts = line.strip().split("\t")
        if len(parts) < 5:
            continue
        rel, head, tail = parts[1], parts[2], parts[3]
        if not head.startswith("/c/en/") or not tail.startswith("/c/en/"):
            continue
        r = _CN_MAP.get(rel)
        if not r:
            continue
        h = _norm_concept(head)
        t = _norm_concept(tail)
        if h and t and h != t:
            triples.append((h, r, t))
        if len(triples) >= max_triples:
            break
        if scanned % 2_000_000 == 0:
            print(f"    ...{scanned // 1_000_000}M rows, {len(triples):,} triples so far")
    reader.close()
    return triples


def load_conceptnet(max_triples: int = 100_000) -> List[Tuple[str, str, str]]:
    try:
        triples = _load_conceptnet_s3(max_triples)
        if triples:
            rc = Counter(r for _, r, _ in triples)
            print(f"  OK  {len(triples):,} triples from S3 ({dict(rc)})")
            return triples
    except Exception as e:
        print(f"  FAIL  S3 streaming: {e}")

    try:
        from datasets import load_dataset
        print("  Trying conceptnet5/conceptnet5 (HuggingFace streaming)...")
        ds = load_dataset("conceptnet5/conceptnet5", split="train", streaming=True)
        triples = []
        for row in ds:
            if row.get("lang") != "en":
                continue
            r = _CN_MAP.get(row.get("rel"))
            if not r:
                continue
            h = _norm_concept(row["arg1"])
            t = _norm_concept(row["arg2"])
            if h and t and h != t:
                triples.append((h, r, t))
            if len(triples) >= max_triples:
                break
        if triples:
            print(f"  OK  {len(triples):,} triples from HuggingFace")
            return triples
    except Exception as e:
        print(f"  FAIL  HuggingFace: {e}")

    augmented = _augment_triples(_FALLBACK_TRIPLES)
    print(f"  Using hardcoded fallback: {len(_FALLBACK_TRIPLES)} base + "
          f"{len(augmented) - len(_FALLBACK_TRIPLES)} augmented = {len(augmented)} triples")
    return augmented


# ---------------------------------------------------------------------------
# Cross-lingual
# ---------------------------------------------------------------------------

def load_crosslingual(max_per_lang: int = 20_000) -> List[Tuple[str, str]]:
    from datasets import load_dataset
    pairs = []
    for lang in ["en-fr", "en-de", "en-es"]:
        try:
            print(f"  Trying tatoeba {lang}...")
            ds = load_dataset("sentence-transformers/parallel-sentences-tatoeba",
                              lang, split="train")
            lp = [(r["english"].strip(), r["non_english"].strip())
                  for r in ds if r["english"].strip() and r["non_english"].strip()]
            pairs.extend(lp[:max_per_lang])
            print(f"  OK  {min(len(lp), max_per_lang)} pairs from tatoeba {lang}")
        except Exception as e:
            print(f"  FAIL  tatoeba {lang}: {e}")

    if not pairs:
        print("  Using synthetic fallback (EN-FR/DE/ES)")
        pairs = [
            ("A dog is a domesticated animal.", "Un chien est un animal domestiqué."),
            ("Fire causes smoke.", "Le feu cause de la fumée."),
            ("A dog is a domesticated animal.", "Ein Hund ist ein domestiziertes Tier."),
            ("Fire causes smoke.", "Feuer verursacht Rauch."),
            ("A dog is a domesticated animal.", "Un perro es un animal domesticado."),
            ("Fire causes smoke.", "El fuego causa humo."),
        ] * 20
    return pairs


# ---------------------------------------------------------------------------
# SNLI
# ---------------------------------------------------------------------------

def load_snli(max_pairs: int = 20_000) -> List[Tuple[str, str, int]]:
    try:
        from datasets import load_dataset
        ds = load_dataset("snli", split=f"train[:{max_pairs}]")
        pairs = [(r["premise"], r["hypothesis"], r["label"])
                 for r in ds if r["label"] in (0, 1, 2)]
        print(f"  OK  {len(pairs)} SNLI pairs")
        return pairs
    except Exception as e:
        print(f"  FAIL  SNLI: {e}")

    print("  Building SNLI-style fallback from hardcoded ConceptNet...")
    pairs = []
    for h, r, t in _FALLBACK_TRIPLES:
        if r == "IS_A":
            pairs.append((f"A {h} is an example.", f"A {t} is an example.", 0))
            pairs.append((f"A {h} is an example.", "Something unrelated happened.", 1))
        elif r == "ANTONYM":
            pairs.append((f"{h} is the topic.", f"{t} is the topic.", 2))
    print(f"  OK  {len(pairs)} fallback SNLI-style pairs")
    return pairs


# ---------------------------------------------------------------------------
# CIFAR-100 constants
# ---------------------------------------------------------------------------

CIFAR100_COARSE = [
    "aquatic mammals", "fish", "flowers", "food containers", "fruit and vegetables",
    "household electrical devices", "household furniture", "insects", "large carnivores",
    "large man-made outdoor things", "large natural outdoor scenes",
    "large omnivores and herbivores", "medium-sized mammals", "non-insect invertebrates",
    "people", "reptiles", "small mammals", "trees", "vehicles 1", "vehicles 2",
]

CIFAR100_FINE = [
    "apple", "aquarium fish", "baby", "bear", "beaver", "bed", "bee", "beetle",
    "bicycle", "bottle", "bowl", "boy", "bridge", "bus", "butterfly", "camel",
    "can", "castle", "caterpillar", "cattle", "chair", "chimpanzee", "clock",
    "cloud", "cockroach", "couch", "crab", "crocodile", "cup", "dinosaur",
    "dolphin", "elephant", "flatfish", "forest", "fox", "girl", "hamster",
    "house", "kangaroo", "keyboard", "lamp", "lawn mower", "leopard", "lion",
    "lizard", "lobster", "man", "maple tree", "motorcycle", "mountain", "mouse",
    "mushroom", "oak tree", "orange", "orchid", "otter", "palm tree", "pear",
    "pickup truck", "pine tree", "plain", "plate", "poppy", "porcupine",
    "possum", "rabbit", "raccoon", "ray", "road", "rocket", "rose", "sea",
    "seal", "shark", "shrew", "skunk", "skyscraper", "snail", "snake",
    "spider", "squirrel", "streetcar", "sunflower", "sweet pepper", "table",
    "tank", "telephone", "television", "tiger", "tractor", "train", "trout",
    "tulip", "turtle", "wardrobe", "whale", "willow tree", "wolf", "woman", "worm",
]

CIFAR100_FINE2COARSE = {
    0: 4, 1: 1, 2: 14, 3: 8, 4: 0, 5: 6, 6: 7, 7: 7, 8: 18, 9: 3, 10: 3, 11: 14,
    12: 9, 13: 18, 14: 7, 15: 11, 16: 3, 17: 9, 18: 7, 19: 11, 20: 6, 21: 11, 22: 5,
    23: 10, 24: 7, 25: 6, 26: 13, 27: 15, 28: 3, 29: 15, 30: 0, 31: 11, 32: 1, 33: 10,
    34: 12, 35: 14, 36: 16, 37: 9, 38: 11, 39: 5, 40: 5, 41: 19, 42: 8, 43: 8, 44: 15,
    45: 13, 46: 14, 47: 17, 48: 18, 49: 10, 50: 16, 51: 4, 52: 17, 53: 4, 54: 2, 55: 0,
    56: 17, 57: 4, 58: 18, 59: 17, 60: 10, 61: 3, 62: 2, 63: 12, 64: 12, 65: 16, 66: 12,
    67: 1, 68: 9, 69: 19, 70: 2, 71: 10, 72: 0, 73: 1, 74: 16, 75: 12, 76: 9, 77: 13,
    78: 15, 79: 13, 80: 16, 81: 19, 82: 2, 83: 4, 84: 6, 85: 19, 86: 5, 87: 5, 88: 8,
    89: 19, 90: 18, 91: 1, 92: 2, 93: 15, 94: 6, 95: 0, 96: 17, 97: 8, 98: 14, 99: 13,
}


def load_cifar100(data_root: str = "./data"):
    import torchvision
    import torchvision.transforms as T

    transform = T.Compose([
        T.Resize(224), T.CenterCrop(224), T.ToTensor(),
        T.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])
    train = torchvision.datasets.CIFAR100(
        root=data_root, train=True, download=True, transform=transform)
    test = torchvision.datasets.CIFAR100(
        root=data_root, train=False, download=True, transform=transform)
    return train, test


# ---------------------------------------------------------------------------
# KG Dataset for DataLoader
# ---------------------------------------------------------------------------

class KGDataset(Dataset):
    def __init__(self, triples: List[Tuple[str, str, str]]):
        self.triples = triples

    def __len__(self):
        return len(self.triples)

    def __getitem__(self, idx):
        h, r, t = self.triples[idx]
        return h, REL2IDX[r], t


def kg_collate_fn(batch):
    heads, rels, tails = zip(*batch)
    return list(heads), torch.tensor(rels, dtype=torch.long), list(tails)
