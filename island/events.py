"""Disruptive events. Four for now (volcanic eruption, war, emigration, storm), written as
plain functions over a Valley's flat category-count vector so more event types can be
added later without touching world.py's pipeline.

All of these operate on `valley.flat_counts` -- a generic vector of category counts (in
this sim: 2 origins [native, arrived] x 3 genotypes [BB, Bb, bb], see world.py) -- and are
oblivious to what the categories actually mean. Binomial thinning (eruption), a random
multinomial faction split (war), and hypergeometric subsampling (emigration) all
automatically preserve genotype *and* origin proportions correctly, with no special-casing
needed here.
"""
from typing import Optional
import numpy as np

from . import sampling
from .config import EmigrationConfig, VolcanoConfig, WarConfig


def scheduled_eruption_index(gen_index: int, n_valleys: int, cfg: VolcanoConfig) -> Optional[int]:
    """Which valley (by ring index) erupts this generation, if any -- a fixed clockwise
    rotation rather than random chance (see VolcanoConfig). Returns None on generations
    that aren't a multiple of eruption_interval_gens, or at generation 0 (Landfall's
    eruption is treated as having already happened before the sim starts)."""
    interval = cfg.eruption_interval_gens
    if interval <= 0 or gen_index <= 0 or gen_index % interval != 0:
        return None
    return (gen_index // interval) % n_valleys


def erupt(valley, cfg: VolcanoConfig, rng: np.random.Generator) -> None:
    """Apply an eruption: moderate immediate death (a random fraction between
    death_fraction_min and death_fraction_max), then devastation_gens generations of
    near-zero capacity (see World.current_capacity) before the ash-fertility bonus begins
    its decay."""
    death_fraction = rng.uniform(cfg.death_fraction_min, cfg.death_fraction_max)
    valley.set_flat_counts(rng.binomial(valley.flat_counts, 1.0 - death_fraction))
    valley.eruption_this_gen = True
    valley.devastated_gens_remaining = cfg.devastation_gens


def storm_strikes(rng: np.random.Generator, chance: float) -> bool:
    """A single canoe crossing this generation: True if the whole party is lost at sea."""
    return rng.random() < chance


def war_probability(streak: int, cfg: WarConfig) -> float:
    """Genuinely near-0% for the first few generations a valley is over capacity, ramping
    -- quadratically, not linearly -- to ~100% if the overcrowding persists for
    `patience_gens` generations straight. A brief one- or two-generation overshoot barely
    registers; only a sustained crisis makes war likely."""
    if cfg.patience_gens <= 0:
        return 1.0 if streak > 0 else 0.0
    return min(1.0, (streak / cfg.patience_gens) ** 2)


def maybe_war(valley, cfg: WarConfig, rng: np.random.Generator) -> Optional[np.ndarray]:
    """If war breaks out: split the valley randomly into `n_factions`, pick a winner
    (bigger factions more likely to win, via `win_power_exponent`), apply casualties to
    everyone, and return the losing faction(s)' survivors as a refugee count-vector (to be
    expelled and migrated elsewhere) -- the winning faction's survivors become the valley's
    new population. Returns None if no war happens this generation."""
    p = war_probability(valley.over_capacity_streak, cfg)
    if rng.random() >= p:
        return None

    counts = valley.flat_counts
    n_cats = len(counts)
    faction_counts = np.zeros((cfg.n_factions, n_cats), dtype=np.int64)
    for cat, count in enumerate(counts):
        if count > 0:
            faction_counts[:, cat] = rng.multinomial(count, [1.0 / cfg.n_factions] * cfg.n_factions)

    sizes = faction_counts.sum(axis=1)
    if sizes.sum() == 0:
        return None
    weights = sizes.astype(float) ** cfg.win_power_exponent
    probs = weights / weights.sum()
    winner = rng.choice(cfg.n_factions, p=probs)

    new_home = np.zeros(n_cats, dtype=np.int64)
    refugees = np.zeros(n_cats, dtype=np.int64)
    for fi in range(cfg.n_factions):
        casualty_rate = cfg.winner_casualty_rate if fi == winner else cfg.loser_casualty_rate
        survivors = rng.binomial(faction_counts[fi], 1.0 - casualty_rate)
        if fi == winner:
            new_home += survivors
        else:
            refugees += survivors

    valley.set_flat_counts(new_home)
    valley.war_occurred_this_gen = True
    valley.over_capacity_streak = 0  # the crisis that caused it has been (violently) resolved
    return refugees


def emigration_probability(streak: int, cfg: EmigrationConfig) -> float:
    """A gentler, earlier-triggering release valve than war: the same quadratic ramp shape
    as war_probability, but capped at `max_chance` rather than approaching certainty."""
    if cfg.patience_gens <= 0:
        return cfg.max_chance if streak > 0 else 0.0
    return min(cfg.max_chance, (streak / cfg.patience_gens) ** 2 * cfg.max_chance)


def maybe_emigrate(valley, cfg: EmigrationConfig, rng: np.random.Generator) -> Optional[np.ndarray]:
    """If triggered: a random fraction (uniform between min_fraction and max_fraction) of
    the valley's population decides to leave the island entirely, looking for fortune
    elsewhere -- they are lost from the simulation, not redistributed. Returns the
    count-vector of who left (for logging), or None if nothing happened."""
    p = emigration_probability(valley.over_capacity_streak, cfg)
    if rng.random() >= p:
        return None

    counts = valley.flat_counts
    total = int(counts.sum())
    if total == 0:
        return None
    fraction = rng.uniform(cfg.min_fraction, cfg.max_fraction)
    n_leaving = min(total, int(round(total * fraction)))
    if n_leaving <= 0:
        return None

    leaving = sampling.weighted_sample_without_replacement_counts(counts, np.ones(len(counts)), n_leaving, rng)
    valley.set_flat_counts(counts - leaving)
    valley.emigration_this_gen = True
    valley.over_capacity_streak = 0  # the crisis has been (peacefully) resolved
    return leaving
