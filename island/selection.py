"""Translates SelectionConfig's human-readable targets ("bb", "brown", ...) into
per-genotype weight multipliers, used at birth (fertility / aesthetic preference) and at
culling (survival / genetic advantage).

Both knobs are independent and constant for now (as requested) -- but since they're just
functions of a Valley-less SelectionConfig, it's a small step later to make them functions
of a valley's own state (e.g. genetic advantage only kicking in the generation after an
eruption, when ash-adapted genes would actually matter).
"""
from typing import Tuple
import numpy as np

from .config import SelectionConfig

_INDEX = {"BB": 0, "Bb": 1, "bb": 2}


def genotype_weights(target: str, strength: float) -> Tuple[float, float, float]:
    weights = [1.0, 1.0, 1.0]
    if target in _INDEX:
        weights[_INDEX[target]] = strength
    elif target == "brown":
        weights[0] = strength
        weights[1] = strength
    elif target == "blue":
        weights[2] = strength
    else:
        raise ValueError(f"Unknown selection target {target!r} (use BB/Bb/bb/brown/blue)")
    return tuple(weights)


def fertility_weights(cfg: SelectionConfig) -> Tuple[float, float, float]:
    """'A visible feature is preferred for aesthetic reasons': skews who gets chosen as a
    parent, applied when the next generation's births are drawn."""
    return genotype_weights(cfg.aesthetic_target, cfg.aesthetic_strength)


def survival_weights(cfg: SelectionConfig) -> Tuple[float, float, float]:
    """'Carrying a gene gives a distinct advantage': skews who survives when a valley is
    culled back down to capacity."""
    return genotype_weights(cfg.advantage_target, cfg.advantage_strength)


def survival_weights_flat(cfg: SelectionConfig) -> np.ndarray:
    """Same as survival_weights, but expanded to the 6-category (origin x genotype) vector
    used at culling time: [native_BB, native_Bb, native_bb, arrived_BB, arrived_Bb, arrived_bb].
    `resident_survival_bonus` gives locally-born individuals a survival edge over newcomers,
    on top of (multiplicatively combined with) the genetic-advantage weight per genotype."""
    genotype_w = np.array(genotype_weights(cfg.advantage_target, cfg.advantage_strength))
    return np.concatenate([genotype_w * cfg.resident_survival_bonus, genotype_w * 1.0])
