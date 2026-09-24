"""All tunable parameters for the island simulation, in one place.

Every number here is a first-guess default meant to be tuned by experiment, not a
carefully-fit constant -- see run_sim.py for how to override them, and README.md for how
these pieces fit into the generation pipeline.
"""
from dataclasses import dataclass, field
from typing import List


@dataclass
class ValleySpec:
    name: str
    base_capacity: int
    init_BB: int
    init_Bb: int
    init_bb: int
    starts_with_ash_buff: bool = False  # already mid-decay from an eruption that happened
                                         # just before generation 0 (see VolcanoConfig)


@dataclass
class GrowthConfig:
    """Births are driven by couples, not a population-wide rate. Each generation, the
    previous generation's (post-cull) population is paired up into `population // 2`
    couples, and the total number of children is a random draw -- Poisson(couples *
    mean_kids(x)) -- rather than a fixed formula, so litter-size variance is part of the
    model, not just genotype variance. `mean_kids(x)` is a smooth decreasing function of
    fill fraction x = population/capacity: `kids_peak` as x -> 0 (an empty valley, food and
    space for everyone), through `kids_at_capacity` right at x=1 (deliberately < 2, so a
    valley sitting at capacity is already trending down), asymptoting to `kids_floor` for
    severe overcrowding (x >> 1). See World._mean_kids_per_couple / World._births_for.

    This naturally produces an S-curve in *total* births without an artificial floor or
    hump: couples scale linearly with population (few couples when nearly empty, even
    though each is very fertile) while mean_kids falls with fill fraction (many couples but
    conservative litters near capacity). A population thinned by war or an eruption
    recovers because it has fewer couples that generation, not because of a special-cased
    growth boost -- which is what caused the old rate-hump model to overshoot right after a
    disaster (fewer people happened to land near the hump's peak fill fraction).

    `fill_lag_gens` makes couples decide family size off a STALE fill fraction --
    `fill_lag_gens` generations old, not this generation's -- rather than reacting in real
    time. There's no population-planning office in these valleys: crowding (or a disaster
    emptying the place out) takes a few generations to actually change birth behavior. This
    is the classic "delayed logistic growth" trick from ecology (Hutchinson's delayed
    logistic equation) -- a population overshoots capacity and corrects afterward, rather
    than snapping straight to equilibrium every generation the way an instantaneous
    (unlagged) reaction does. Set to 0 to go back to instant, tightly self-regulating
    reactions.
    """
    kids_peak: float = 4.5         # mean kids/couple as population -> 0 (empty valley)
    kids_at_capacity: float = 1.9  # mean kids/couple exactly at capacity (x=1); < 2 => decline
    kids_floor: float = 1.0        # asymptotic mean kids/couple under severe overcrowding
    fill_lag_gens: int = 4          # how stale the crowding perception feeding mean_kids is


@dataclass
class MigrationConfig:
    """Ordinary migration works like the emigration/war events -- a single sized group
    decides to move this generation, not a continuous trickle of the entire surplus. Each
    generation, a valley with any surplus draws a random fraction of that surplus (a
    "family" at the small end, a "gathering" at the large end) as this generation's
    candidate movers; that group then rolls the usual stay-or-go-where lottery (weighted by
    destination room and travel cost)."""
    migration_min_fraction: float = 0.35  # smallest slice of the surplus that moves as one group ("a family")
    migration_max_fraction: float = 0.90  # largest slice ("a whole gathering")
    path_cost: float = 1.0      # mountain path between ring-neighbors: cheap
    water_cost: float = 4.0     # canoe route to any other valley: expensive (and risky, see StormConfig)
    mobility: float = 1.5       # overall eagerness of the candidate group to leave at all
    stay_weight: float = 5.0    # relative pull of "just stay put" for the candidate group


@dataclass
class StormConfig:
    storm_chance: float = 0.15  # probability a given canoe crossing this generation is lost entirely


@dataclass
class CullConfig:
    """Overcrowding is resolved gradually, not instantly: each generation, only a fraction
    of the excess-over-capacity is removed, so severe overcrowding can take several
    generations to fully resolve -- leaving time for emigration or war to intervene first.
    """
    cull_fraction_per_gen: float = 0.5


@dataclass
class VolcanoConfig:
    """Eruptions are a fixed, predictable rotation around the ring, not per-valley random
    chance -- deliberate, so the island's rhythm stays legible generation to generation.
    Every `eruption_interval_gens` generations, the next valley clockwise (by ring index)
    erupts, so any one valley is hit once every `n_valleys * eruption_interval_gens`
    generations (4 valleys x 20 = every 80). Landfall (index 0) counts as having just
    erupted right before generation 0 -- it starts already mid-recovery (see
    ValleySpec.starts_with_ash_buff) -- so the rotation's first live eruption is the next
    valley clockwise, at generation `eruption_interval_gens`, cycling back to Landfall at
    4x that.

    An eruption itself does only moderate immediate damage (a random fraction between
    death_fraction_min and death_fraction_max), but leaves the valley functionally
    uninhabitable (capacity floored to ~0, see World.current_capacity) for
    `devastation_gens` generations -- long enough for the ordinary migration/war/cull
    machinery to do the rest of the work: some residents flee right away as a normal
    migration group, others stay and either get gradually culled or end up fighting over
    the scraps. Once devastation ends, the ash-fertility bonus begins its
    `fertility_decay_gens`-generation decay from there (not from the eruption itself)."""
    eruption_interval_gens: int = 20
    death_fraction_min: float = 0.10  # immediate population loss on eruption (sampled per event)
    death_fraction_max: float = 0.20
    devastation_gens: int = 2         # generations the valley is ~uninhabitable after erupting
    fertility_bonus: float = 0.5      # peak capacity bonus once devastation ends
    fertility_decay_gens: int = 20    # bonus decays linearly to 0 over this many generations


@dataclass
class WarConfig:
    """War risk is driven by how long a valley has been stuck over capacity, not just how
    far over: genuinely near-0% for the first few generations of overcrowding, escalating
    sharply only if it persists for `patience_gens` generations straight, reaching ~100% at
    that point (see events.war_probability -- a quadratic ramp, not linear, so a brief
    one-or-two-generation overshoot barely registers). A gradual cull and the emigration
    release-valve both give the valley a chance to resolve things first."""
    patience_gens: int = 12
    n_factions: int = 2
    winner_casualty_rate: float = 0.10
    loser_casualty_rate: float = 0.40
    win_power_exponent: float = 1.3  # >1: bigger factions win more reliably (not guaranteed)


@dataclass
class EmigrationConfig:
    """A gentler, earlier release valve than war: checked first each generation, when
    enabled. A random fraction of an overcrowded valley's population decides to leave the
    island entirely (lost from the simulation, not redistributed) in search of fortune
    elsewhere. Ramps up faster than war but caps at `max_chance` rather than approaching
    certainty. Currently switched off (`enabled=False`) to cut down on event friction and
    let overcrowding pressure build toward ordinary migration and war instead -- the
    mechanism is untouched, just gated, so it's a one-line change to bring back."""
    enabled: bool = False
    patience_gens: int = 6
    max_chance: float = 0.30
    min_fraction: float = 0.15
    max_fraction: float = 0.50


@dataclass
class SelectionConfig:
    # All three knobs are OFF (neutral) by default so the model matches the pure
    # drift / Hardy-Weinberg baseline from Part 1 unless you deliberately switch one on.
    aesthetic_target: str = "bb"        # "BB" | "Bb" | "bb" | "brown" | "blue" -- who's fashionable
    aesthetic_strength: float = 1.0     # fertility-weight multiplier for the target (at birth)
    advantage_target: str = "bb"
    advantage_strength: float = 1.0     # survival-weight multiplier for the target (at culling)
    resident_survival_bonus: float = 1.0  # survival-weight multiplier for locally-born vs.
                                           # new arrivals at culling (1.0 = no home-field edge)


@dataclass
class SimulationConfig:
    generations: int = 100
    seed: int = 20260923
    growth: GrowthConfig = field(default_factory=GrowthConfig)
    migration: MigrationConfig = field(default_factory=MigrationConfig)
    storm: StormConfig = field(default_factory=StormConfig)
    cull: CullConfig = field(default_factory=CullConfig)
    volcano: VolcanoConfig = field(default_factory=VolcanoConfig)
    war: WarConfig = field(default_factory=WarConfig)
    emigration: EmigrationConfig = field(default_factory=EmigrationConfig)
    selection: SelectionConfig = field(default_factory=SelectionConfig)


def default_valleys() -> List[ValleySpec]:
    """200 blue-eyed founders land in Landfall, freshly recovered from an eruption of its
    own just before generation 0 (hence starts_with_ash_buff -- it begins with the
    ash-fertility bonus already active, not the devastation). The other three valleys start
    full of pure brown-eyed settlers. Capacities are generous (500) relative to the founding
    populations so the sim isn't dominated by noise around a tiny equilibrium."""
    return [
        ValleySpec("Landfall", 500, 0, 0, 200, starts_with_ash_buff=True),
        ValleySpec("Highreach", 500, 500, 0, 0),
        ValleySpec("Saltmarsh", 500, 500, 0, 0),
        ValleySpec("Emberfield", 500, 500, 0, 0),
    ]
