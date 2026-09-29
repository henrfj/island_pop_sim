"""Configuration for the five-year, food-mediated island simulation."""
from dataclasses import dataclass, field
from typing import Dict, List



@dataclass(frozen=True)
class ValleySpec:
    name: str
    tribe: str
    land_yield: float
    storage_limit: float
    initial_food: float
    init_BB: int
    init_Bb: int
    init_bb: int
    spoilage_rate: float = 1.0
    initial_land_health: float = 1.0
    initial_ash_bonus_turns: int = 0


@dataclass
class DemographyConfig:
    child_share: float = 0.32
    adult_share: float = 0.56
    child_survival: float = 0.965
    adult_survival: float = 0.975
    elder_survival: float = 0.72
    child_aging_fraction: float = 0.25
    adult_aging_fraction: float = 0.10
    births_per_adult: float = 0.36
    reproductive_fraction: float = 0.42
    fertility_food_floor: float = 0.10
    fertility_full_security: float = 1.05
    shortage_mortality_onset: float = 0.65
    shortage_mortality_child: float = 0.20
    shortage_mortality_adult: float = 0.10
    shortage_mortality_elder: float = 0.28
    surplus_fertility_bonus: float = 0.18


@dataclass
class FoodConfig:
    adult_labor_productivity: float = 1.70
    labor_saturation: float = 420.0
    child_consumption: float = 0.65
    adult_consumption: float = 1.0
    elder_consumption: float = 0.85
    spoilage_fraction: float = 0.12
    security_memory: float = 0.55
    comfortable_security: float = 1.15


@dataclass
class MigrationConfig:
    households_per_1000_people: float = 4.0
    shortage_wave_chance: float = 0.18
    shortage_wave_households: float = 3.0
    family_size_min: int = 3
    family_size_max: int = 12
    path_cost: float = 1.0
    water_cost: float = 3.0
    storm_chance: float = 0.08
    clan_destination_pull: float = 0.20
    surplus_emigration_bonus: float = 0.80
    surplus_stock_threshold: float = 0.55


@dataclass
class TribeConfig:
    clan_endogamy: float = 0.20
    clan_cohesion: float = 0.30
    shock_cohesion_bonus: float = 0.25
    shock_memory_decay: float = 0.82
    local_child_influence: float = 0.20
    mixed_clan_chance: float = 0.35
    culture_mutation_band: float = 1.0
    minimum_independent_population: int = 20
    political_absorption_chance: float = 0.28
    political_split_chance: float = 0.12
    diaspora_collapse_population: int = 18
    political_maturity_turns: int = 3
    secession_suffixes: tuple[str, ...] = (
        "West", "East", "Upper", "Lower", "River", "Ash",
    )


@dataclass(frozen=True)
class GeneTraits:
    """Ancestry-indexed effects in BB, Bb, bb order.

    `BB` leans island-rooted, `bb` leans newcomer-heavy, and `Bb` sits between.
    These signals are intentionally weaker than culture because the simulation spans
    only a short cultural timescale.
    """
    island_immunity: tuple[float, float, float] = (0.04, 0.01, -0.04)
    frontier_drive: tuple[float, float, float] = (-0.02, 0.01, 0.04)


@dataclass(frozen=True)
class CultureTraits:
    warlike: float = 0.0
    peaceful: float = 0.0
    strength_in_numbers: float = 0.0
    strong_individuals: float = 0.0
    xenophile: float = 0.0
    xenophobic: float = 0.0
    syncretic: float = 0.0
    insular: float = 0.0
    harsh_discipline: float = 0.0
    nurturing: float = 0.0
    agrarian: float = 0.0
    hunter_gatherer: float = 0.0
    migratory: float = 0.0
    stationary: float = 0.0

    def __post_init__(self) -> None:
        normalized = {
            "warlike": max(0.0, self.warlike),
            "peaceful": max(0.0, self.peaceful),
            "strength_in_numbers": max(0.0, self.strength_in_numbers),
            "strong_individuals": max(0.0, self.strong_individuals),
            "xenophile": max(0.0, self.xenophile),
            "xenophobic": max(0.0, self.xenophobic),
            "syncretic": max(0.0, self.syncretic),
            "insular": max(0.0, self.insular),
            "harsh_discipline": max(0.0, self.harsh_discipline),
            "nurturing": max(0.0, self.nurturing),
            "agrarian": max(0.0, self.agrarian),
            "hunter_gatherer": max(0.0, self.hunter_gatherer),
            "migratory": max(0.0, self.migratory),
            "stationary": max(0.0, self.stationary),
        }
        for positive, opposite in (
            ("warlike", "peaceful"),
            ("strength_in_numbers", "strong_individuals"),
            ("xenophile", "xenophobic"),
            ("syncretic", "insular"),
            ("harsh_discipline", "nurturing"),
            ("migratory", "stationary"),
        ):
            overlap = min(normalized[positive], normalized[opposite])
            normalized[positive] -= overlap
            normalized[opposite] -= overlap
        for name, value in normalized.items():
            object.__setattr__(self, name, float(value))


CULTURE_TRAIT_NAMES = tuple(CultureTraits.__dataclass_fields__)
_NEUTRAL_CULTURE = CultureTraits()
OPPOSING_CULTURE_TRAITS = {
    "warlike": "peaceful",
    "peaceful": "warlike",
    "strength_in_numbers": "strong_individuals",
    "strong_individuals": "strength_in_numbers",
    "xenophile": "xenophobic",
    "xenophobic": "xenophile",
    "syncretic": "insular",
    "insular": "syncretic",
    "harsh_discipline": "nurturing",
    "nurturing": "harsh_discipline",
    "migratory": "stationary",
    "stationary": "migratory",
}


@dataclass(frozen=True)
class ArrivalEventConfig:
    turn: int = 400
    valley: str = "Landfall"
    clan: str = "Blackwake"
    init_BB: int = 0
    init_Bb: int = 0
    init_bb: int = 220
    traits: CultureTraits = field(default_factory=lambda: CultureTraits(
        warlike=0.28,
        strong_individuals=0.24,
        xenophobic=0.55,
        harsh_discipline=0.14,
        migratory=0.22,
    ))
    dwell_min_turns: int = 10
    dwell_max_turns: int = 50
    land_drain_per_turn: float = 0.04
    food_security_floor: float = 0.72
    departure_fraction: float = 0.97
    mass_move_chance: float = 0.38
    mass_move_land_threshold: float = 0.78
    cohesion_bonus: float = 0.45

    @property
    def population(self) -> int:
        return self.init_BB + self.init_Bb + self.init_bb


@dataclass
class TraitConfig:
    genes: GeneTraits = field(default_factory=GeneTraits)
    clans: Dict[str, CultureTraits] = field(default_factory=lambda: {
        "Tideborn": CultureTraits(migratory=0.12),
        "Cloudfolk": CultureTraits(agrarian=0.12),
        "Reedkin": CultureTraits(nurturing=0.12),
        "Ashclan": CultureTraits(warlike=0.12),
    })
    randomize_founders: bool = True
    locked_founders: frozenset[str] = field(default_factory=frozenset)

    def culture(self, clan: str) -> CultureTraits:
        culture = self.clans.get(clan)
        return culture if culture is not None else _NEUTRAL_CULTURE

    def founder_signature(self, clan: str) -> str | None:
        for trait_name in CULTURE_TRAIT_NAMES:
            if getattr(self.culture(clan), trait_name) > 0:
                return trait_name
        return None

    def randomized_founder_culture(self, clan: str, rng) -> CultureTraits:
        signature = self.founder_signature(clan)
        if signature is None:
            return self.culture(clan)
        bonus_candidates = [
            trait_name for trait_name in CULTURE_TRAIT_NAMES
            if trait_name != signature
            and OPPOSING_CULTURE_TRAITS.get(trait_name) != signature
        ]
        bonus_trait = str(rng.choice(bonus_candidates))
        values = {trait_name: 0.0 for trait_name in CULTURE_TRAIT_NAMES}
        values[signature] = float(min(0.20, max(0.07, getattr(self.culture(clan), signature)
                            + rng.uniform(-0.02, 0.03))))
        values[bonus_trait] = float(rng.uniform(0.03, 0.10))
        return CultureTraits(**values)

    def __post_init__(self) -> None:
        self.clans = {
            clan: traits if isinstance(traits, CultureTraits) else CultureTraits(**traits)
            for clan, traits in self.clans.items()
        }


@dataclass
class VolcanoConfig:
    eruption_interval_turns: int = 4
    immediate_mortality_min: float = 0.00
    immediate_mortality_max: float = 0.03
    store_destruction_min: float = 0.05
    store_destruction_max: float = 0.20
    land_damage_min: float = 0.10
    land_damage_max: float = 0.25
    forced_displacement_fraction: float = 0.12
    land_recovery_fraction: float = 0.35
    ash_bonus: float = 0.15
    ash_bonus_turns: int = 4


@dataclass
class ConflictConfig:
    enabled: bool = True
    shortage_threshold: float = 0.72
    base_chance: float = 0.015
    shortage_pressure: float = 0.30
    winner_casualty_rate: float = 0.035
    loser_casualty_rate: float = 0.10
    displacement_fraction: float = 0.30
    food_destruction_min: float = 0.08
    food_destruction_max: float = 0.25
    land_damage_min: float = 0.02
    land_damage_max: float = 0.10
    secession_chance_severe: float = 0.06
    secession_min_population: int = 60


@dataclass
class CivilWarConfig:
    enabled: bool = False
    trigger_turn: int = 250
    max_factions: int = 5
    branch_fraction_min: float = 0.18
    branch_fraction_max: float = 0.38
    shock_bonus: float = 0.55


@dataclass
class SimulationConfig:
    turns: int = 500
    years_per_turn: int = 5
    seed: int = 20260923
    demography: DemographyConfig = field(default_factory=DemographyConfig)
    food: FoodConfig = field(default_factory=FoodConfig)
    migration: MigrationConfig = field(default_factory=MigrationConfig)
    tribes: TribeConfig = field(default_factory=TribeConfig)
    traits: TraitConfig = field(default_factory=TraitConfig)
    volcano: VolcanoConfig = field(default_factory=VolcanoConfig)
    conflict: ConflictConfig = field(default_factory=ConflictConfig)
    civil_war: CivilWarConfig = field(default_factory=CivilWarConfig)
    arrivals: tuple[ArrivalEventConfig, ...] = field(default_factory=tuple)

    def validate(self) -> None:
        if self.turns < 0 or self.years_per_turn <= 0:
            raise ValueError("turns must be nonnegative and years_per_turn positive")
        probabilities = {
            "child_survival": self.demography.child_survival,
            "adult_survival": self.demography.adult_survival,
            "elder_survival": self.demography.elder_survival,
            "child_aging_fraction": self.demography.child_aging_fraction,
            "adult_aging_fraction": self.demography.adult_aging_fraction,
            "shortage_mortality_onset": self.demography.shortage_mortality_onset,
            "spoilage_fraction": self.food.spoilage_fraction,
            "shortage_wave_chance": self.migration.shortage_wave_chance,
            "storm_chance": self.migration.storm_chance,
            "clan_destination_pull": self.migration.clan_destination_pull,
            "clan_endogamy": self.tribes.clan_endogamy,
            "clan_cohesion": self.tribes.clan_cohesion,
            "shock_memory_decay": self.tribes.shock_memory_decay,
            "mixed_clan_chance": self.tribes.mixed_clan_chance,
            "political_absorption_chance": self.tribes.political_absorption_chance,
            "political_split_chance": self.tribes.political_split_chance,
        }
        for name, value in probabilities.items():
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must be between 0 and 1")
        if self.migration.family_size_min <= 0 or self.migration.family_size_max < self.migration.family_size_min:
            raise ValueError("invalid family-size range")
        if self.migration.households_per_1000_people < 0 or self.migration.shortage_wave_households < 0:
            raise ValueError("migration rates must be nonnegative")
        if self.food.labor_saturation <= 0 or self.food.comfortable_security <= 0:
            raise ValueError("food scales must be positive")
        if self.demography.fertility_full_security <= 0:
            raise ValueError("fertility_full_security must be positive")
        if self.tribes.culture_mutation_band < 0:
            raise ValueError("culture_mutation_band must be nonnegative")
        if self.tribes.diaspora_collapse_population <= 0:
            raise ValueError("diaspora_collapse_population must be positive")
        if self.tribes.political_maturity_turns < 0:
            raise ValueError("political_maturity_turns must be nonnegative")
        if self.civil_war.trigger_turn < 0:
            raise ValueError("civil_war trigger_turn must be nonnegative")
        if self.civil_war.max_factions < 2:
            raise ValueError("civil_war max_factions must be at least 2")
        if not 0.0 <= self.civil_war.branch_fraction_min <= 1.0:
            raise ValueError("civil_war branch_fraction_min must be between 0 and 1")
        if not 0.0 <= self.civil_war.branch_fraction_max <= 1.0:
            raise ValueError("civil_war branch_fraction_max must be between 0 and 1")
        if self.civil_war.branch_fraction_max < self.civil_war.branch_fraction_min:
            raise ValueError("civil_war branch_fraction_max must be at least branch_fraction_min")
        if not 0.0 <= self.civil_war.shock_bonus <= 1.0:
            raise ValueError("civil_war shock_bonus must be between 0 and 1")
        for arrival in self.arrivals:
            if arrival.turn < 0:
                raise ValueError("arrival turn must be nonnegative")
            if arrival.population <= 0:
                raise ValueError("arrival population must be positive")
            if arrival.dwell_min_turns <= 0 or arrival.dwell_max_turns < arrival.dwell_min_turns:
                raise ValueError("arrival dwell range is invalid")
            if not 0.0 <= arrival.land_drain_per_turn <= 1.0:
                raise ValueError("arrival land_drain_per_turn must be between 0 and 1")
            if not 0.0 <= arrival.food_security_floor <= 1.5:
                raise ValueError("arrival food_security_floor must be between 0 and 1.5")
            if not 0.0 <= arrival.departure_fraction <= 1.0:
                raise ValueError("arrival departure_fraction must be between 0 and 1")
            if not 0.0 <= arrival.mass_move_chance <= 1.0:
                raise ValueError("arrival mass_move_chance must be between 0 and 1")
            if not 0.0 <= arrival.mass_move_land_threshold <= 1.0:
                raise ValueError("arrival mass_move_land_threshold must be between 0 and 1")
            if arrival.cohesion_bonus < 0:
                raise ValueError("arrival cohesion_bonus must be nonnegative")


def default_valleys() -> List[ValleySpec]:
    """Four founding tribes; Landfall carries the recessive blue-eye genotype."""
    return [
        ValleySpec("Landfall", "Tideborn", 760.0, 900.0, 700.0, 0, 0, 500,
                   spoilage_rate=0.95,
                   initial_land_health=0.78, initial_ash_bonus_turns=16),
        ValleySpec("Highreach", "Cloudfolk", 650.0, 820.0, 650.0, 500, 0, 0,
                   spoilage_rate=0.72),
        ValleySpec("Saltmarsh", "Reedkin", 590.0, 680.0, 590.0, 500, 0, 0,
                   spoilage_rate=1.30),
        ValleySpec("Emberfield", "Ashclan", 620.0, 760.0, 620.0, 500, 0, 0,
                   spoilage_rate=1.05),
    ]


def main_volcanic_scenario() -> tuple[SimulationConfig, List[ValleySpec]]:
    """Recovering Landfall and an eruption every 20 years around the ring."""
    return SimulationConfig(), default_valleys()


def tideborn_defensive_scenario() -> tuple[SimulationConfig, List[ValleySpec]]:
    """Test whether a stationary, cohesive, xenophobic Tideborn can hold Landfall."""
    config = SimulationConfig()
    config.arrivals = ()
    config.tribes.clan_cohesion = 0.38
    config.tribes.political_maturity_turns = 6
    config.migration.households_per_1000_people = 3.1
    config.traits.locked_founders = frozenset({"Tideborn"})
    config.traits.clans["Tideborn"] = CultureTraits(
        strength_in_numbers=0.25,
        xenophobic=1,
        stationary=1,
    )
    valleys = default_valleys()
    valleys[0] = ValleySpec(
        "Landfall", "Tideborn", 820.0, 1180.0, 880.0, 0, 0, 720,
        spoilage_rate=0.92,
        initial_land_health=0.92,
        initial_ash_bonus_turns=20,
    )
    return config, valleys


def slow_volcanic_scenario() -> tuple[SimulationConfig, List[ValleySpec]]:
    """Same founding geography, but each valley erupts once every 160 years."""
    config = SimulationConfig()
    config.volcano.eruption_interval_turns = 8
    return config, default_valleys()


def neutral_control_scenario() -> tuple[SimulationConfig, List[ValleySpec]]:
    """Equal environments and no eruptions for checking neutral genetic behavior."""
    config = SimulationConfig()
    config.volcano.eruption_interval_turns = 0
    valleys = [
        ValleySpec(spec.name, spec.tribe, 620.0, 1050.0, 620.0,
                   spec.init_BB, spec.init_Bb, spec.init_bb)
        for spec in default_valleys()
    ]
    return config, valleys


def civil_war_scenario() -> tuple[SimulationConfig, List[ValleySpec]]:
    """Late succession crisis that fractures surviving founding clans into rival factions."""
    config = SimulationConfig(turns=650)
    config.arrivals = ()
    config.civil_war = CivilWarConfig(
        enabled=True,
        trigger_turn=500,
        max_factions=5,
        branch_fraction_min=0.20,
        branch_fraction_max=0.40,
        shock_bonus=0.65,
    )
    config.conflict.base_chance = 0.012
    config.conflict.shortage_pressure = 0.26
    config.conflict.secession_chance_severe = 0.10
    return config, default_valleys()