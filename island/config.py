"""Configuration for the five-year, food-mediated island simulation."""
from dataclasses import dataclass, field
from typing import List


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


@dataclass
class TribeConfig:
    clan_endogamy: float = 0.20
    clan_cohesion: float = 0.30
    local_child_influence: float = 0.20
    minimum_independent_population: int = 20
    secession_suffixes: tuple[str, ...] = (
        "West", "East", "Upper", "Lower", "River", "Ash",
    )


@dataclass
class VolcanoConfig:
    eruption_interval_turns: int = 4
    immediate_mortality_min: float = 0.05
    immediate_mortality_max: float = 0.12
    store_destruction_min: float = 0.35
    store_destruction_max: float = 0.70
    land_damage_min: float = 0.45
    land_damage_max: float = 0.75
    forced_displacement_fraction: float = 0.12
    land_recovery_fraction: float = 0.18
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
class SimulationConfig:
    turns: int = 800
    years_per_turn: int = 5
    seed: int = 20260923
    demography: DemographyConfig = field(default_factory=DemographyConfig)
    food: FoodConfig = field(default_factory=FoodConfig)
    migration: MigrationConfig = field(default_factory=MigrationConfig)
    tribes: TribeConfig = field(default_factory=TribeConfig)
    volcano: VolcanoConfig = field(default_factory=VolcanoConfig)
    conflict: ConflictConfig = field(default_factory=ConflictConfig)

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


def default_valleys() -> List[ValleySpec]:
    """Four founding tribes; Landfall carries the recessive blue-eye genotype."""
    return [
        ValleySpec("Landfall", "Tideborn", 760.0, 1200.0, 700.0, 0, 0, 200,
                   initial_land_health=0.78, initial_ash_bonus_turns=16),
        ValleySpec("Highreach", "Cloudfolk", 650.0, 1100.0, 650.0, 500, 0, 0),
        ValleySpec("Saltmarsh", "Reedkin", 590.0, 1000.0, 590.0, 500, 0, 0),
        ValleySpec("Emberfield", "Ashclan", 620.0, 1050.0, 620.0, 500, 0, 0),
    ]


def main_volcanic_scenario() -> tuple[SimulationConfig, List[ValleySpec]]:
    """Recovering Landfall and an eruption every 20 years around the ring."""
    return SimulationConfig(), default_valleys()


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