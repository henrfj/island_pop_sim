"""Clan membership assignment and lineage-readable branch naming."""
import numpy as np

from .config import CultureTraits, TribeConfig


TRAIT_WORDS = {
    "warlike": ("Spear", "Ember", "Fang"),
    "peaceful": ("Harbor", "Calm", "Dawn"),
    "strength_in_numbers": ("Host", "Many", "Union"),
    "strong_individuals": ("Free", "Bold", "Rift"),
    "xenophile": ("Open", "Bridge", "Welcome"),
    "xenophobic": ("Ward", "Gate", "Bound"),
    "syncretic": ("Braided", "Blend", "Mosaic"),
    "insular": ("Keep", "Holdfast", "Closed"),
    "harsh_discipline": ("Law", "Order", "Chain"),
    "nurturing": ("Kin", "Nest", "Bloom"),
    "agrarian": ("Field", "Harvest", "Loam"),
    "hunter_gatherer": ("Trail", "Forage", "Reed"),
    "migratory": ("Drift", "Wake", "Roam"),
    "stationary": ("Hearth", "Stone", "Root"),
}


def allocate_mixed_children(birth_counts: np.ndarray, first_parent_weight: float,
                            local_first_share: float, cfg: TribeConfig,
                            rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    parental = min(max(first_parent_weight, 0.0), 1.0)
    local = min(max(local_first_share, 0.0), 1.0)
    first_probability = ((1.0 - cfg.local_child_influence) * parental
                         + cfg.local_child_influence * local)
    first = rng.binomial(np.asarray(birth_counts, dtype=np.int64), first_probability)
    return first, np.asarray(birth_counts, dtype=np.int64) - first


def unique_branch_name(parent: str, existing: set[str], cfg: TribeConfig) -> str:
    root = parent.split(" ", 1)[0]
    for suffix in cfg.secession_suffixes:
        candidate = f"{root} {suffix}"
        if candidate not in existing:
            return candidate
    index = 2
    while f"{root} {index}" in existing:
        index += 1
    return f"{root} {index}"


def _lineage_roots(name: str) -> list[str]:
    roots = []
    for part in name.split(" ", 1)[0].split("-"):
        if part and part not in roots:
            roots.append(part)
    return roots


def _compact_root_label(root: str) -> str:
    root = root.strip()
    lowered = root.lower()
    if lowered.endswith("folk"):
        root = root[:-4]
    elif lowered.endswith("born"):
        root = root[:-4]
    elif lowered.endswith("clan"):
        root = root[:-4]
    elif lowered.endswith("kin"):
        root = root[:-3]
    return root or "Clan"


def _trait_word(culture: CultureTraits, rng: np.random.Generator) -> str:
    ranked = sorted(culture.__dict__.items(), key=lambda item: (-item[1], item[0]))
    top_trait, top_value = ranked[0]
    if top_value <= 0:
        return "New"
    return str(rng.choice(TRAIT_WORDS.get(top_trait, ("New",))))


def unique_mixed_name(first: str, second: str, dominant_parent: str,
                      culture: CultureTraits, existing: set[str],
                      rng: np.random.Generator) -> str:
    root = f"{_trait_word(culture, rng)} {_compact_root_label(dominant_parent)}"
    candidate = root
    index = 2
    while candidate in existing:
        candidate = f"{root} {index}"
        index += 1
    return candidate


def mixed_culture(first: CultureTraits, second: CultureTraits, mutation_band: float,
                  rng: np.random.Generator) -> CultureTraits:
    """Blend parent cultures once, then freeze the resulting culture on the new clan."""
    values = []
    for first_value, second_value in zip(first.__dict__.values(), second.__dict__.values()):
        mean = (first_value + second_value) / 2.0
        span = mutation_band * max(abs(first_value), abs(second_value), 0.10)
        values.append(float(np.clip(mean + rng.uniform(-span, span), 0.0, 1.0)))
    return CultureTraits(*values)