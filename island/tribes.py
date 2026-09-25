"""Clan membership assignment and lineage-readable branch naming."""
import numpy as np

from .config import CultureTraits, TribeConfig


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


def unique_mixed_name(first: str, second: str, existing: set[str]) -> str:
    roots = sorted({first.split(" ", 1)[0], second.split(" ", 1)[0]})
    root = "-".join(roots)
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