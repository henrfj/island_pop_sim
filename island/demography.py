"""Aggregate mortality, aging, and fertility for five-year cohorts."""
from dataclasses import dataclass
from typing import Tuple
import numpy as np

from .config import DemographyConfig

CHILD, ADULT, ELDER = range(3)
LIFE_STAGES = ("child", "adult", "elder")


@dataclass(frozen=True)
class CohortTransition:
    counts: np.ndarray
    baseline_deaths: int
    shortage_deaths: int


def initial_age_counts(genotypes: Tuple[int, int, int], cfg: DemographyConfig,
                       rng: np.random.Generator) -> np.ndarray:
    result = np.zeros((3, 3), dtype=np.int64)
    shares = np.array([cfg.child_share, cfg.adult_share,
                       max(0.0, 1.0 - cfg.child_share - cfg.adult_share)])
    shares /= shares.sum()
    for genotype, count in enumerate(genotypes):
        result[:, genotype] = rng.multinomial(int(count), shares)
    return result


def transition_cohorts(counts: np.ndarray, food_security: float, cfg: DemographyConfig,
                       rng: np.random.Generator) -> CohortTransition:
    counts = np.asarray(counts, dtype=np.int64)
    if counts.shape != (3, 3):
        raise ValueError("cohort counts must have shape (life_stage=3, genotype=3)")
    survival = np.array([cfg.child_survival, cfg.adult_survival, cfg.elder_survival])[:, None]
    baseline_survivors = rng.binomial(counts, survival)
    baseline_deaths = int(counts.sum() - baseline_survivors.sum())
    onset = cfg.shortage_mortality_onset
    shortage = max(0.0, onset - food_security) / max(onset, 1e-9)
    shortage_rates = shortage * np.array([
        cfg.shortage_mortality_child,
        cfg.shortage_mortality_adult,
        cfg.shortage_mortality_elder,
    ])[:, None]
    survivors = rng.binomial(baseline_survivors, 1.0 - shortage_rates)
    shortage_deaths = int(baseline_survivors.sum() - survivors.sum())
    aged_children = rng.binomial(survivors[CHILD], cfg.child_aging_fraction)
    aged_adults = rng.binomial(survivors[ADULT], cfg.adult_aging_fraction)
    transitioned = survivors.copy()
    transitioned[CHILD] -= aged_children
    transitioned[ADULT] += aged_children - aged_adults
    transitioned[ELDER] += aged_adults
    return CohortTransition(transitioned, baseline_deaths, shortage_deaths)


def expected_births(adult_population: int, food_security: float, cfg: DemographyConfig,
                    rng: np.random.Generator) -> int:
    relative_security = min(max(food_security / cfg.fertility_full_security, 0.0), 1.0)
    food_factor = cfg.fertility_food_floor + (1.0 - cfg.fertility_food_floor) * relative_security
    mean = adult_population * cfg.reproductive_fraction * cfg.births_per_adult * food_factor
    return int(rng.poisson(max(0.0, mean)))