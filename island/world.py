"""Five-year island simulation with overlapping cohorts, food stocks, and tribes."""
from dataclasses import replace
from dataclasses import asdict, dataclass, field
from typing import Dict, List, Optional
import numpy as np

from . import conflict, demography, genetics, resources, sampling, tribes
from .config import ArrivalEventConfig, SimulationConfig, ValleySpec

NEWCOMER, ESTABLISHED = range(2)
SHAPE = (2, 3, 3)  # residency, life stage, genotype


@dataclass
class Valley:
    name: str
    founding_tribe: str
    land_yield: float
    storage_limit: float
    food_stock: float
    cohorts: Dict[str, np.ndarray] = field(default_factory=dict)
    land_health: float = 1.0
    food_security: float = 1.0
    ash_bonus_turns: int = 0
    erupted_this_turn: bool = False
    shock_memory: float = 0.0
    surplus_memory: float = 0.0

    @property
    def population(self) -> int:
        return sum(int(counts.sum()) for counts in self.cohorts.values())

    @property
    def stage_counts(self) -> np.ndarray:
        total = np.zeros(3, dtype=np.int64)
        for counts in self.cohorts.values():
            total += counts.sum(axis=(0, 2))
        return total

    @property
    def genotype_counts(self) -> np.ndarray:
        total = np.zeros(3, dtype=np.int64)
        for counts in self.cohorts.values():
            total += counts.sum(axis=(0, 1))
        return total

    @property
    def q(self) -> Optional[float]:
        n_BB, n_Bb, n_bb = self.genotype_counts
        total = int(n_BB + n_Bb + n_bb)
        if total == 0:
            return None
        return float((n_Bb + 2 * n_bb) / (2 * total))

    def ensure_tribe(self, tribe: str) -> np.ndarray:
        if tribe not in self.cohorts:
            self.cohorts[tribe] = np.zeros(SHAPE, dtype=np.int64)
        return self.cohorts[tribe]

    def clean_empty_tribes(self) -> None:
        self.cohorts = {name: counts for name, counts in self.cohorts.items() if counts.sum() > 0}


class Island:
    def __init__(self, valley_specs: List[ValleySpec], config: SimulationConfig,
                 rng: np.random.Generator):
        config.validate()
        self.cfg = config
        self.rng = rng
        self.valleys: List[Valley] = []
        for spec in valley_specs:
            age_counts = demography.initial_age_counts(
                (spec.init_BB, spec.init_Bb, spec.init_bb), config.demography, rng)
            cohorts = np.zeros(SHAPE, dtype=np.int64)
            cohorts[ESTABLISHED] = age_counts
            self.valleys.append(Valley(
                spec.name, spec.tribe, spec.land_yield, spec.storage_limit,
                spec.initial_food, {spec.tribe: cohorts},
                land_health=spec.initial_land_health,
                ash_bonus_turns=spec.initial_ash_bonus_turns,
            ))
        self.n = len(self.valleys)
        self.founding_clans = tuple(dict.fromkeys(spec.tribe for spec in valley_specs))
        self.path_neighbors = {i: {(i - 1) % self.n, (i + 1) % self.n} for i in range(self.n)}
        self.mixed_clans: Dict[frozenset[str], str] = {}
        self.arrival_schedule: Dict[int, List[ArrivalEventConfig]] = {}
        self.arrival_by_clan: Dict[str, ArrivalEventConfig] = {
            arrival.clan: arrival for arrival in config.arrivals
        }
        for arrival in config.arrivals:
            self.arrival_schedule.setdefault(arrival.turn, []).append(arrival)
        self.history: List[dict] = []
        self.event_log: List[dict] = []
        self._reset_tracking()

    def _reset_tracking(self) -> None:
        self.history = []
        self.event_log = []
        self.mixed_clans = {}
        self.cumulative_deaths_by_source = {
            "eruption": 0,
            "baseline": 0,
            "shortage": 0,
            "storm": 0,
            "war": 0,
        }
        self.cumulative_baseline_deaths_by_stage = {
            stage: 0 for stage in demography.LIFE_STAGES
        }
        self.cumulative_deaths_by_genotype = {
            genotype: 0 for genotype in genetics.GENOTYPES
        }
        self.cumulative_deaths_by_clan: Dict[str, int] = {
            clan: 0 for clan in self.founding_clans
        }
        self.clan_lifecycles: Dict[str, dict] = {}
        self.active_arrivals: Dict[str, dict] = {}
        for clan in self.founding_clans:
            self._register_clan(clan, 0, "founding")

    def _find_valley_index(self, name: str) -> int:
        for index, valley in enumerate(self.valleys):
            if valley.name == name:
                return index
        raise ValueError(f"unknown valley {name!r}")

    def _register_clan(self, clan: str, turn: int, origin: str,
                       parents: Optional[List[str]] = None,
                       valley: Optional[str] = None) -> None:
        if clan in self.clan_lifecycles:
            return
        self.cumulative_deaths_by_clan.setdefault(clan, 0)
        self.clan_lifecycles[clan] = {
            "birth_turn": turn,
            "birth_year": turn * self.cfg.years_per_turn,
            "origin": origin,
            "parents": list(parents or []),
            "birth_valley": valley,
            "extinct_turn": None,
            "extinct_year": None,
        }

    def _genotype_totals(self, counts: np.ndarray) -> np.ndarray:
        values = np.asarray(counts, dtype=np.int64)
        if values.ndim == 3:
            return values.sum(axis=(0, 1))
        if values.ndim == 2:
            return values.sum(axis=0)
        raise ValueError("unsupported cohort shape")

    def _record_deaths(self, source: str, deaths_by_clan: Dict[str, np.ndarray]) -> int:
        total_deaths = 0
        for clan, dead_counts in deaths_by_clan.items():
            dead = np.asarray(dead_counts, dtype=np.int64)
            clan_deaths = int(dead.sum())
            if clan_deaths <= 0:
                continue
            total_deaths += clan_deaths
            self.cumulative_deaths_by_clan[clan] = (
                self.cumulative_deaths_by_clan.get(clan, 0) + clan_deaths
            )
            for genotype, count in zip(genetics.GENOTYPES, self._genotype_totals(dead)):
                self.cumulative_deaths_by_genotype[genotype] += int(count)
        self.cumulative_deaths_by_source[source] += total_deaths
        return total_deaths

    def _diff_counts(self, before: Dict[str, np.ndarray],
                     after: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        delta = {}
        for clan, before_counts in before.items():
            after_counts = after.get(clan)
            if after_counts is None:
                diff = before_counts.copy()
            else:
                diff = before_counts - after_counts
            if int(diff.sum()) > 0:
                delta[clan] = diff
        return delta

    def _island_clan_populations(self) -> Dict[str, int]:
        populations: Dict[str, int] = {}
        for valley in self.valleys:
            for clan, counts in valley.cohorts.items():
                populations[clan] = populations.get(clan, 0) + int(counts.sum())
        return populations

    def _record_clan_extinctions(self, previous_populations: Dict[str, int],
                                 turn: int) -> List[dict]:
        current_populations = self._island_clan_populations()
        events_out = []
        for clan, previous in previous_populations.items():
            if previous <= 0 or current_populations.get(clan, 0) > 0:
                continue
            lifecycle = self.clan_lifecycles.setdefault(clan, {
                "birth_turn": 0,
                "birth_year": 0,
                "origin": "unknown",
                "parents": [],
                "birth_valley": None,
                "extinct_turn": None,
                "extinct_year": None,
            })
            if lifecycle["extinct_turn"] is not None:
                continue
            lifecycle["extinct_turn"] = turn
            lifecycle["extinct_year"] = turn * self.cfg.years_per_turn
            events_out.append({
                "turn": turn,
                "year": turn * self.cfg.years_per_turn,
                "type": "clan_extinct",
                "tribe": clan,
                "origin": lifecycle["origin"],
                "total_deaths": self.cumulative_deaths_by_clan.get(clan, 0),
            })
        return events_out

    def travel_cost(self, origin: int, destination: int) -> float:
        if destination in self.path_neighbors[origin]:
            return self.cfg.migration.path_cost
        return self.cfg.migration.water_cost

    def _thin_all(self, valley: Valley, survival: float) -> int:
        before = valley.population
        for tribe, counts in valley.cohorts.items():
            valley.cohorts[tribe] = self.rng.binomial(counts, survival).astype(np.int64)
        valley.clean_empty_tribes()
        return before - valley.population

    def _weighted_gene_effect(self, valley: Valley, effect: tuple[float, float, float]) -> float:
        adults = sum(
            (counts[:, demography.ADULT, :].sum(axis=0)
             for counts in valley.cohorts.values()),
            np.zeros(3, dtype=np.int64),
        )
        total = int(adults.sum())
        if total == 0:
            return 0.0
        return float(np.dot(adults, np.asarray(effect, dtype=float)) / total)

    def _genotype_migration_drive(self, counts: np.ndarray) -> float:
        adults = counts[:, demography.ADULT, :].sum(axis=0)
        total = int(adults.sum())
        if total == 0:
            return 0.0
        return float(np.dot(
            adults, np.asarray(self.cfg.traits.genes.migration_drive, dtype=float)) / total)

    def _blue_share_from_stage_counts(self, counts: np.ndarray) -> float:
        total = int(np.asarray(counts, dtype=np.int64).sum())
        if total == 0:
            return 0.0
        return float(np.asarray(counts, dtype=np.int64)[..., 2].sum() / total)

    def _valley_blue_share(self, valley: Valley) -> float:
        total = valley.population
        if total <= 0:
            return 0.0
        return float(sum(int(counts[:, :, 2].sum()) for counts in valley.cohorts.values()) / total)

    def _weighted_culture_signal(self, valley: Valley, attribute: str) -> float:
        if valley.population <= 0:
            return 0.0
        return sum(
            int(counts.sum()) * getattr(self.cfg.traits.culture(clan), attribute)
            for clan, counts in valley.cohorts.items()
        ) / valley.population

    def _arrival_cohesion_bonus(self, valley: Valley) -> float:
        if valley.population <= 0:
            return 0.0
        bonus = 0.0
        for clan, counts in valley.cohorts.items():
            arrival = self.arrival_by_clan.get(clan)
            if arrival is None:
                continue
            bonus += arrival.cohesion_bonus * int(counts.sum()) / valley.population
        return bonus

    def _cross_clan_openness(self, first_clan: str, first_counts: np.ndarray,
                             second_clan: str, second_counts: np.ndarray) -> float:
        first_culture = self.cfg.traits.culture(first_clan)
        second_culture = self.cfg.traits.culture(second_clan)
        phenotype_gap = abs(
            self._blue_share_from_stage_counts(first_counts)
            - self._blue_share_from_stage_counts(second_counts)
        )
        xenophile = 0.5 * (first_culture.xenophile + second_culture.xenophile)
        xenophobic = 0.5 * (first_culture.xenophobic + second_culture.xenophobic)
        return max(0.15, 1.0 + 1.2 * xenophile * phenotype_gap - 1.4 * xenophobic * phenotype_gap)

    def _refugee_destination(self, origin: int,
                             displaced_by_clan: Dict[str, np.ndarray]) -> Optional[int]:
        options = [index for index in range(self.n) if index != origin]
        weights = []
        for destination in options:
            valley = self.valleys[destination]
            outlook = 0.2 + valley.food_security + valley.food_stock / max(valley.storage_limit, 1.0)
            receiving_xenophile = self._weighted_culture_signal(valley, "xenophile")
            receiving_xenophobic = self._weighted_culture_signal(valley, "xenophobic")
            receiving_nurturing = self._weighted_culture_signal(valley, "nurturing")
            receiving_harsh = self._weighted_culture_signal(valley, "harsh_discipline")
            valley_blue_share = self._valley_blue_share(valley)
            refugee_total = 0
            same_clan_anchor = 0.0
            phenotype_gap = 0.0
            for clan, displaced in displaced_by_clan.items():
                clan_total = int(displaced.sum())
                if clan_total == 0:
                    continue
                refugee_total += clan_total
                same_clan_anchor += clan_total * (
                    int(valley.cohorts.get(clan, np.zeros(SHAPE, dtype=np.int64)).sum())
                    / max(valley.population, 1)
                )
                phenotype_gap += clan_total * abs(
                    self._blue_share_from_stage_counts(displaced) - valley_blue_share
                )
            if refugee_total == 0:
                weights.append(0.0)
                continue
            phenotype_gap /= refugee_total
            same_clan_anchor /= refugee_total
            acceptance = (
                1.0
                + 0.75 * same_clan_anchor
                + 1.10 * receiving_xenophile * phenotype_gap
                - 1.30 * receiving_xenophobic * phenotype_gap
                + 0.15 * receiving_nurturing
                - 0.12 * receiving_harsh
            )
            weights.append(max(0.05, outlook * acceptance / self.travel_cost(origin, destination)))
        total = sum(weights)
        if total <= 0:
            return None
        return int(self.rng.choice(options, p=np.asarray(weights) / total))

    def _arrivals(self, turn: int) -> List[dict]:
        events_out = []
        for arrival in self.arrival_schedule.get(turn, []):
            valley_index = self._find_valley_index(arrival.valley)
            valley = self.valleys[valley_index]
            age_counts = demography.initial_age_counts(
                (arrival.init_BB, arrival.init_Bb, arrival.init_bb),
                self.cfg.demography,
                self.rng,
            )
            cohorts = np.zeros(SHAPE, dtype=np.int64)
            cohorts[NEWCOMER] = age_counts
            valley.ensure_tribe(arrival.clan)[:] += cohorts
            self.cfg.traits.clans[arrival.clan] = arrival.traits
            departure_turn = turn + int(self.rng.integers(
                arrival.dwell_min_turns, arrival.dwell_max_turns + 1
            ))
            self.active_arrivals[arrival.clan] = {
                "departure_turn": departure_turn,
                "arrival_turn": turn,
                "arrival_valley": arrival.valley,
            }
            self._register_clan(arrival.clan, turn, "arrival", valley=arrival.valley)
            events_out.append({
                "turn": turn,
                "year": turn * self.cfg.years_per_turn,
                "type": "clan_arrival",
                "tribe": arrival.clan,
                "valley": arrival.valley,
                "count": arrival.population,
                "departure_turn": departure_turn,
                "departure_year": departure_turn * self.cfg.years_per_turn,
            })
        return events_out

    def _move_or_depart_arrivals(self, turn: int) -> List[dict]:
        events_out = []
        for clan, state in list(self.active_arrivals.items()):
            valley_counts = []
            total_counts = np.zeros(SHAPE, dtype=np.int64)
            for index, valley in enumerate(self.valleys):
                counts = valley.cohorts.get(clan)
                if counts is None or int(counts.sum()) == 0:
                    continue
                valley_counts.append((index, counts.copy()))
                total_counts += counts
            total_population = int(total_counts.sum())
            if total_population == 0:
                self.active_arrivals.pop(clan, None)
                continue
            if turn >= state["departure_turn"]:
                removed_total = 0
                for index, counts in valley_counts:
                    departing = self.rng.binomial(
                        counts,
                        self.arrival_by_clan[clan].departure_fraction,
                    ).astype(np.int64)
                    self.valleys[index].cohorts[clan] -= departing
                    removed_total += int(departing.sum())
                    self.valleys[index].clean_empty_tribes()
                self.active_arrivals.pop(clan, None)
                events_out.append({
                    "turn": turn,
                    "year": turn * self.cfg.years_per_turn,
                    "type": "clan_departure",
                    "tribe": clan,
                    "count": removed_total,
                })
                continue

            dominant_index, dominant_counts = max(
                valley_counts, key=lambda item: int(item[1].sum())
            )
            origin_valley = self.valleys[dominant_index]
            arrival = self.arrival_by_clan[clan]
            should_move = len(valley_counts) > 1
            should_move = should_move or origin_valley.land_health <= arrival.mass_move_land_threshold
            should_move = should_move or self.rng.random() < arrival.mass_move_chance
            if not should_move:
                continue
            destination = self._destination(dominant_index, clan, total_counts)
            if destination is None:
                continue
            moved_counts = np.zeros(SHAPE, dtype=np.int64)
            for index, counts in valley_counts:
                moved_counts += counts
                self.valleys[index].cohorts.pop(clan, None)
                self.valleys[index].clean_empty_tribes()
            arriving = np.zeros(SHAPE, dtype=np.int64)
            arriving[NEWCOMER] = moved_counts.sum(axis=0)
            self.valleys[destination].ensure_tribe(clan)[:] += arriving
            events_out.append({
                "turn": turn,
                "year": turn * self.cfg.years_per_turn,
                "type": "clan_expedition",
                "tribe": clan,
                "from": origin_valley.name,
                "to": self.valleys[destination].name,
                "count": int(moved_counts.sum()),
                "route": "path" if destination in self.path_neighbors[dominant_index] else "canoe",
            })
        return events_out

    def _erupt(self, turn: int) -> List[dict]:
        for valley in self.valleys:
            valley.erupted_this_turn = False
        interval = self.cfg.volcano.eruption_interval_turns
        if interval <= 0 or turn <= 0 or turn % interval != 0:
            return []
        index = (turn // interval) % self.n
        valley = self.valleys[index]
        before_by_clan = {clan: counts.copy() for clan, counts in valley.cohorts.items()}
        cfg = self.cfg.volcano
        mortality = self.rng.uniform(cfg.immediate_mortality_min, cfg.immediate_mortality_max)
        before = valley.population
        fire_resistance = np.asarray(self.cfg.traits.genes.fire_resistance, dtype=float)
        for clan, counts in valley.cohorts.items():
            culture = self.cfg.traits.culture(clan)
            vulnerability = 1.0 + 0.10 * culture.strength_in_numbers
            survival = 1.0 - mortality * vulnerability * (1.0 - fire_resistance)
            valley.cohorts[clan] = self.rng.binomial(
                counts, np.clip(survival, 0.0, 1.0)).astype(np.int64)
        valley.clean_empty_tribes()
        deaths = before - valley.population
        store_fraction = self.rng.uniform(cfg.store_destruction_min, cfg.store_destruction_max)
        food_destroyed = valley.food_stock * store_fraction
        valley.food_stock -= food_destroyed
        land_damage = self.rng.uniform(cfg.land_damage_min, cfg.land_damage_max)
        valley.land_health *= 1.0 - land_damage
        valley.erupted_this_turn = True
        valley.ash_bonus_turns = cfg.ash_bonus_turns
        valley.shock_memory = min(1.0, valley.shock_memory + 0.70)
        self._record_deaths("eruption", self._diff_counts(before_by_clan, valley.cohorts))
        return [{
            "turn": turn,
            "year": turn * self.cfg.years_per_turn,
            "type": "eruption",
            "valley": valley.name,
            "deaths": deaths,
            "food_destroyed": food_destroyed,
            "land_damage": land_damage,
        }]

    def _food_and_demography(self) -> List[dict]:
        ledgers = []
        for valley in self.valleys:
            stages = valley.stage_counts
            effective_land = valley.land_health * (
                1.0 + self._weighted_gene_effect(
                    valley, self.cfg.traits.genes.ash_farming))
            stationary_share = 0.0
            migratory_share = 0.0
            harsh_share = 0.0
            nurturing_share = 0.0
            total_people = max(valley.population, 1)
            for clan, counts in valley.cohorts.items():
                share = counts.sum() / total_people
                culture = self.cfg.traits.culture(clan)
                stationary_share += share * culture.stationary
                migratory_share += share * culture.migratory
                harsh_share += share * culture.harsh_discipline
                nurturing_share += share * culture.nurturing
            effective_land *= 1.0 + 0.15 * stationary_share - 0.08 * migratory_share
            effective_land *= 1.0 + 0.06 * harsh_share - 0.03 * nurturing_share
            if valley.ash_bonus_turns > 0:
                effective_land *= 1.0 + self.cfg.volcano.ash_bonus
            food = resources.update_food(
                valley.food_stock, valley.storage_limit, valley.land_yield, effective_land,
                int(stages[demography.ADULT]), int(stages[demography.CHILD]),
                int(stages[demography.ELDER]), self.cfg.food,
            )
            valley.food_stock = food.ending_stock
            valley.food_security = resources.smoothed_security(
                valley.food_security, food.security, self.cfg.food)
            stock_ratio = valley.food_stock / max(valley.storage_limit, 1.0)
            good_year = valley.food_security >= 0.92 and stock_ratio >= 0.35
            valley.surplus_memory = min(
                1.0,
                0.82 * valley.surplus_memory + (0.18 if good_year else 0.0),
            )
            baseline_deaths = 0
            shortage_deaths = 0
            extractive_land_loss = 0.0
            baseline_dead_by_clan: Dict[str, np.ndarray] = {}
            shortage_dead_by_clan: Dict[str, np.ndarray] = {}
            for tribe, counts in valley.cohorts.items():
                transitioned = np.zeros_like(counts)
                baseline_dead = np.zeros((3, 3), dtype=np.int64)
                shortage_dead = np.zeros((3, 3), dtype=np.int64)
                culture = self.cfg.traits.culture(tribe)
                arrival = self.arrival_by_clan.get(tribe)
                protected_security = valley.food_security
                if arrival is not None and valley.food_stock > 0:
                    protected_security = max(protected_security, arrival.food_security_floor)
                adjusted_demography = replace(
                    self.cfg.demography,
                    child_survival=float(np.clip(
                        self.cfg.demography.child_survival
                        + 0.03 * culture.nurturing
                        - 0.04 * culture.harsh_discipline,
                        0.0,
                        1.0,
                    )),
                    adult_survival=float(np.clip(
                        self.cfg.demography.adult_survival
                        + 0.01 * culture.harsh_discipline,
                        0.0,
                        1.0,
                    )),
                )
                for residency in (NEWCOMER, ESTABLISHED):
                    result = demography.transition_cohorts(
                        counts[residency], protected_security, adjusted_demography, self.rng)
                    transitioned[residency] = result.counts
                    baseline_deaths += result.baseline_deaths
                    shortage_deaths += result.shortage_deaths
                    baseline_dead += result.baseline_death_counts
                    shortage_dead += result.shortage_death_counts
                valley.cohorts[tribe] = transitioned
                if arrival is not None and valley.population > 0:
                    extractive_land_loss += arrival.land_drain_per_turn * min(
                        1.0,
                        int(transitioned.sum()) / max(valley.population, 1),
                    )
                if int(baseline_dead.sum()) > 0:
                    baseline_dead_by_clan[tribe] = baseline_dead
                if int(shortage_dead.sum()) > 0:
                    shortage_dead_by_clan[tribe] = shortage_dead
            valley.clean_empty_tribes()
            if extractive_land_loss > 0:
                valley.land_health *= max(0.0, 1.0 - extractive_land_loss)
            baseline_stage_totals = np.zeros(3, dtype=np.int64)
            self._record_deaths("baseline", baseline_dead_by_clan)
            self._record_deaths("shortage", shortage_dead_by_clan)
            for dead_counts in baseline_dead_by_clan.values():
                baseline_stage_totals += np.asarray(dead_counts, dtype=np.int64).sum(axis=1)
            for stage, count in zip(demography.LIFE_STAGES, baseline_stage_totals):
                self.cumulative_baseline_deaths_by_stage[stage] += int(count)
            ledgers.append({
                "valley": valley.name,
                "food_produced": food.produced,
                "food_spoiled": food.spoiled,
                "food_consumed": food.consumed,
                "food_unmet": food.unmet,
                "baseline_deaths": baseline_deaths,
                "baseline_child_deaths": int(baseline_stage_totals[demography.CHILD]),
                "baseline_adult_deaths": int(baseline_stage_totals[demography.ADULT]),
                "baseline_elder_deaths": int(baseline_stage_totals[demography.ELDER]),
                "shortage_deaths": shortage_deaths,
                "extractive_land_loss": extractive_land_loss,
            })
        return ledgers

    def _destination(self, origin: int, clan: Optional[str] = None,
                     clan_counts: Optional[np.ndarray] = None) -> Optional[int]:
        options = [index for index in range(self.n) if index != origin]
        weights = []
        for destination in options:
            valley = self.valleys[destination]
            outlook = 0.2 + valley.food_security + valley.food_stock / max(valley.storage_limit, 1.0)
            if clan is not None:
                culture = self.cfg.traits.culture(clan)
                if clan in valley.cohorts:
                    clan_share = int(valley.cohorts[clan].sum()) / max(valley.population, 1)
                    outlook *= 1.0 + self.cfg.migration.clan_destination_pull * clan_share * (
                        1.0 + culture.xenophobic - 0.4 * culture.xenophile
                    )
                if clan_counts is not None:
                    phenotype_gap = abs(
                        self._blue_share_from_stage_counts(clan_counts)
                        - self._valley_blue_share(valley)
                    )
                    outlook *= max(
                        0.2,
                        1.0 + 0.9 * culture.xenophile * phenotype_gap
                        - 1.1 * culture.xenophobic * phenotype_gap,
                    )
            weights.append(outlook / self.travel_cost(origin, destination))
        total = sum(weights)
        if total <= 0:
            return None
        return int(self.rng.choice(options, p=np.asarray(weights) / total))

    def _migrate(self, turn: int) -> tuple[List[dict], List[dict]]:
        cfg = self.cfg.migration
        deltas: List[Dict[str, np.ndarray]] = [dict() for _ in self.valleys]
        events_out: List[dict] = []
        ledger = [{"migration_attempted": 0, "migration_moved": 0,
                   "storm_deaths": 0} for _ in self.valleys]
        snapshots = [{tribe: counts.copy() for tribe, counts in valley.cohorts.items()}
                     for valley in self.valleys]
        for origin, tribe_counts in enumerate(snapshots):
            valley = self.valleys[origin]
            available_by_tribe = {tribe: counts.copy() for tribe, counts in tribe_counts.items()}
            expected_households = valley.population * cfg.households_per_1000_people / 1000.0
            stock_ratio = valley.food_stock / max(valley.storage_limit, 1.0)
            surplus = max(0.0, stock_ratio - cfg.surplus_stock_threshold)
            expected_households *= 1.0 + cfg.surplus_emigration_bonus * (
                valley.surplus_memory + surplus)
            shortage = max(0.0, 1.0 - valley.food_security)
            if shortage > 0 and self.rng.random() < cfg.shortage_wave_chance * shortage:
                expected_households += cfg.shortage_wave_households * shortage
            household_count = int(self.rng.poisson(expected_households))
            for _ in range(household_count):
                tribe_names = [name for name, counts in available_by_tribe.items()
                               if counts.sum() >= cfg.family_size_min
                               and name not in self.active_arrivals]
                if not tribe_names:
                    break
                tribe_weights = np.asarray(
                    [available_by_tribe[name].sum() * (
                        1.0 + self.cfg.traits.culture(name).migratory
                        - 0.6 * self.cfg.traits.culture(name).stationary
                        + self._genotype_migration_drive(available_by_tribe[name]))
                     for name in tribe_names], dtype=float)
                tribe_weights = np.maximum(tribe_weights, 0.01)
                tribe = str(self.rng.choice(tribe_names, p=tribe_weights / tribe_weights.sum()))
                counts = available_by_tribe[tribe]
                available = int(counts.sum())
                family_size = int(self.rng.integers(cfg.family_size_min, cfg.family_size_max + 1))
                family_size = min(family_size, available)
                selected = np.zeros(SHAPE, dtype=np.int64)
                adult_counts = counts[:, demography.ADULT, :]
                if adult_counts.sum() > 0:
                    chosen_adult = sampling.weighted_sample_without_replacement_counts(
                        adult_counts.reshape(-1), np.ones(adult_counts.size), 1, self.rng)
                    selected[:, demography.ADULT, :] = chosen_adult.reshape(2, 3)
                remaining = counts - selected
                if family_size > int(selected.sum()):
                    selected += sampling.weighted_sample_without_replacement_counts(
                        remaining.reshape(-1), np.ones(remaining.size),
                        family_size - int(selected.sum()), self.rng).reshape(SHAPE)
                available_by_tribe[tribe] -= selected
                destination = self._destination(origin, tribe, selected)
                if destination is None:
                    available_by_tribe[tribe] += selected
                    continue
                ledger[origin]["migration_attempted"] += family_size
                source_delta = deltas[origin].setdefault(tribe, np.zeros(SHAPE, dtype=np.int64))
                source_delta -= selected
                water = destination not in self.path_neighbors[origin]
                route = "canoe" if water else "path"
                storm_chance = cfg.storm_chance * (
                    1.0 + 0.35 * self.cfg.traits.culture(tribe).migratory)
                if water and self.rng.random() < min(1.0, storm_chance):
                    ledger[origin]["storm_deaths"] += family_size
                    self._record_deaths("storm", {tribe: selected})
                    events_out.append({
                        "turn": turn, "year": turn * self.cfg.years_per_turn,
                        "type": "storm", "tribe": tribe,
                        "from": self.valleys[origin].name,
                        "to": self.valleys[destination].name, "lost": family_size,
                    })
                    continue
                arriving = np.zeros(SHAPE, dtype=np.int64)
                arriving[NEWCOMER] = selected.sum(axis=0)
                destination_delta = deltas[destination].setdefault(
                    tribe, np.zeros(SHAPE, dtype=np.int64))
                destination_delta += arriving
                ledger[origin]["migration_moved"] += family_size
                events_out.append({
                    "turn": turn, "year": turn * self.cfg.years_per_turn,
                    "type": "migration", "tribe": tribe,
                    "from": self.valleys[origin].name,
                    "to": self.valleys[destination].name,
                    "count": family_size, "route": route,
                })
        for index, valley in enumerate(self.valleys):
            for tribe, delta in deltas[index].items():
                updated = valley.ensure_tribe(tribe) + delta
                if np.any(updated < 0):
                    raise RuntimeError("migration produced negative cohort counts")
                valley.cohorts[tribe] = updated
            valley.clean_empty_tribes()
        return events_out, ledger

    def _conflicts(self, turn: int) -> tuple[List[dict], List[dict]]:
        cfg = self.cfg.conflict
        ledger = [{"war_deaths": 0, "war_refugees": 0,
                   "war_food_destroyed": 0.0, "war_land_damage": 0.0,
                   "seceded": 0} for _ in self.valleys]
        if not cfg.enabled:
            return [], ledger

        planned = []
        snapshots = [{tribe: counts.copy() for tribe, counts in valley.cohorts.items()}
                     for valley in self.valleys]
        for index, tribe_counts in enumerate(snapshots):
            if sum(int(counts.sum()) for counts in tribe_counts.values()) < 2:
                continue
            valley = self.valleys[index]
            shortage = max(0.0, cfg.shortage_threshold - valley.food_security) / max(
                cfg.shortage_threshold, 1e-9)
            damage_pressure = max(0.0, 1.0 - valley.land_health)
            total_people = sum(int(counts.sum()) for counts in tribe_counts.values())
            cultural_pressure = 0.0
            if total_people:
                cultural_pressure = sum(
                    counts.sum() * max(
                        0.0,
                        1.0 + 1.2 * self.cfg.traits.culture(clan).warlike
                        - self.cfg.traits.culture(clan).peaceful)
                    for clan, counts in tribe_counts.items()) / total_people
            probability = min(1.0, cultural_pressure * (
                cfg.base_chance + cfg.shortage_pressure * shortage
                + 0.10 * damage_pressure))
            if self.rng.random() >= probability:
                continue
            xenophobic_share = self._weighted_culture_signal(valley, "xenophobic")
            xenophile_share = self._weighted_culture_signal(valley, "xenophile")
            harsh_share = self._weighted_culture_signal(valley, "harsh_discipline")
            nurturing_share = self._weighted_culture_signal(valley, "nurturing")
            split = conflict.partition_factions(
                tribe_counts,
                min(1.0, self.cfg.tribes.clan_cohesion
                    + self.cfg.tribes.shock_cohesion_bonus * valley.shock_memory
                    + self._arrival_cohesion_bonus(valley)
                    + 0.18 * xenophobic_share + 0.12 * harsh_share
                    - 0.16 * xenophile_share - 0.08 * nurturing_share),
                self.rng)
            planned.append((index, split, probability))

        events_out = []
        refugee_flows = []
        for index, split, probability in planned:
            valley = self.valleys[index]
            first_adults = sum(int(counts[:, demography.ADULT, :].sum())
                               for counts in split.first.values()) + 1
            second_adults = sum(int(counts[:, demography.ADULT, :].sum())
                                for counts in split.second.values()) + 1
            first_wins = bool(self.rng.random() < first_adults / (first_adults + second_adults))
            winning_side = split.first if first_wins else split.second
            losing_side = split.second if first_wins else split.first
            survivors_by_clan = {}
            displaced_by_clan = {}
            deaths_by_clan = {}
            deaths = 0
            for clan in valley.cohorts:
                culture = self.cfg.traits.culture(clan)
                war_resistance = np.asarray(self.cfg.traits.genes.war_resistance)
                resistance = war_resistance + 0.12 * culture.warlike
                resistance += 0.10 * culture.strong_individuals
                vulnerability = (1.0 + 0.12 * culture.strength_in_numbers
                                 + 0.18 * culture.peaceful
                                 - 0.10 * culture.strong_individuals)
                winners = self.rng.binomial(
                    winning_side[clan], np.clip(
                        1.0 - cfg.winner_casualty_rate * vulnerability * (1.0 - resistance),
                        0.0, 1.0)).astype(np.int64)
                losers = self.rng.binomial(
                    losing_side[clan], np.clip(
                        1.0 - cfg.loser_casualty_rate * vulnerability * (1.0 - resistance),
                        0.0, 1.0)).astype(np.int64)
                deaths_by_clan[clan] = winning_side[clan] + losing_side[clan] - winners - losers
                deaths += int(winning_side[clan].sum() + losing_side[clan].sum()
                              - winners.sum() - losers.sum())
                displaced_count = int(round(losers.sum() * cfg.displacement_fraction))
                displaced = sampling.weighted_sample_without_replacement_counts(
                    losers.reshape(-1), np.ones(losers.size), displaced_count,
                    self.rng).reshape(SHAPE)
                displaced_by_clan[clan] = displaced
                survivors_by_clan[clan] = winners + losers - displaced

            existing_names = {name for v in self.valleys for name in v.cohorts}
            seceded_name = None
            largest_displaced_clan = max(
                displaced_by_clan, key=lambda clan: int(displaced_by_clan[clan].sum()))
            largest_displaced = int(displaced_by_clan[largest_displaced_clan].sum())
            if (largest_displaced >= cfg.secession_min_population
                    and self.rng.random() < cfg.secession_chance_severe):
                seceded_name = tribes.unique_branch_name(
                    largest_displaced_clan, existing_names, self.cfg.tribes)
                displaced_by_clan[seceded_name] = displaced_by_clan.pop(largest_displaced_clan)
                self.cfg.traits.clans[seceded_name] = tribes.mixed_culture(
                    self.cfg.traits.culture(largest_displaced_clan),
                    self.cfg.traits.culture(largest_displaced_clan),
                    self.cfg.tribes.culture_mutation_band, self.rng)
                self._register_clan(
                    seceded_name,
                    turn,
                    "split",
                    parents=[largest_displaced_clan],
                    valley=valley.name,
                )
                events_out.append({
                    "turn": turn,
                    "year": turn * self.cfg.years_per_turn,
                    "type": "clan_split",
                    "valley": valley.name,
                    "from_clan": largest_displaced_clan,
                    "new_clan": seceded_name,
                    "count": largest_displaced,
                })

            valley.cohorts = survivors_by_clan
            food_fraction = self.rng.uniform(cfg.food_destruction_min, cfg.food_destruction_max)
            food_destroyed = valley.food_stock * food_fraction
            valley.food_stock -= food_destroyed
            land_damage = self.rng.uniform(cfg.land_damage_min, cfg.land_damage_max)
            valley.land_health *= 1.0 - land_damage
            valley.shock_memory = min(1.0, valley.shock_memory + 0.35)
            self._record_deaths("war", deaths_by_clan)

            refugee_count = sum(int(counts.sum()) for counts in displaced_by_clan.values())
            destination = self._refugee_destination(index, displaced_by_clan)
            if destination is not None and refugee_count > 0:
                refugee_flows.append((index, destination, displaced_by_clan))
            ledger[index]["war_deaths"] += deaths
            ledger[index]["war_refugees"] += refugee_count
            ledger[index]["war_food_destroyed"] += food_destroyed
            ledger[index]["war_land_damage"] += land_damage
            ledger[index]["seceded"] += largest_displaced if seceded_name else 0
            events_out.append({
                "turn": turn, "year": turn * self.cfg.years_per_turn,
                "type": "war", "valley": valley.name,
                "winner": "first" if first_wins else "second", "deaths": deaths,
                "refugees": refugee_count, "food_destroyed": food_destroyed,
                "land_damage": land_damage,
                "clan_splits": split.first_probabilities,
                "outcome": "secession" if seceded_name else "displaced",
                "new_clan": seceded_name,
                "probability": probability,
            })

        for origin, destination, displaced_by_clan in refugee_flows:
            for clan, displaced in displaced_by_clan.items():
                if displaced.sum() == 0:
                    continue
                arriving = np.zeros(SHAPE, dtype=np.int64)
                arriving[NEWCOMER] = displaced.sum(axis=0)
                self.valleys[destination].ensure_tribe(clan)[:] += arriving
                events_out.append({
                    "turn": turn, "year": turn * self.cfg.years_per_turn,
                    "type": "refugees", "tribe": clan,
                    "from": self.valleys[origin].name,
                    "to": self.valleys[destination].name,
                    "count": int(displaced.sum()),
                    "route": "path" if destination in self.path_neighbors[origin] else "canoe",
                })
        for valley in self.valleys:
            valley.clean_empty_tribes()
        return events_out, ledger

    def _births(self, turn: int) -> tuple[List[dict], List[dict]]:
        events_out = []
        ledgers = []
        for valley in self.valleys:
            adult_genotypes_by_tribe = {
                tribe: counts[:, demography.ADULT, :].sum(axis=0)
                for tribe, counts in valley.cohorts.items()
            }
            adult_by_tribe = {
                tribe: int(counts.sum()) for tribe, counts in adult_genotypes_by_tribe.items()
            }
            adults = sum(adult_by_tribe.values())
            births = demography.expected_births(
                adults, valley.food_security, self.cfg.demography, self.rng)
            nurturing_bonus = sum(
                adult_by_tribe[tribe] * self.cfg.traits.culture(tribe).nurturing
                for tribe in adult_by_tribe
            ) / adults
            harsh_penalty = sum(
                adult_by_tribe[tribe] * self.cfg.traits.culture(tribe).harsh_discipline
                for tribe in adult_by_tribe
            ) / adults
            births = int(round(births * (
                1.0 + self.cfg.demography.surplus_fertility_bonus
                * valley.surplus_memory)))
            births = int(round(births * (1.0 + 0.10 * nurturing_bonus - 0.06 * harsh_penalty)))
            if births > 0 and adults > 0:
                tribe_names = list(adult_by_tribe)
                strength = sum(
                    adult_by_tribe[tribe] * (
                        self.cfg.traits.culture(tribe).strength_in_numbers
                        - 0.5 * self.cfg.traits.culture(tribe).strong_individuals
                    )
                    for tribe in tribe_names
                ) / adults
                births = int(round(births * (1.0 + 0.15 * strength)))
                shares = np.asarray([adult_by_tribe[name] for name in tribe_names], dtype=float)
                shares /= shares.sum()
                pair_weights = np.outer(shares, shares)
                endogamy = self.cfg.tribes.clan_endogamy
                pair_weights *= 1.0 - endogamy
                diagonal = np.diag_indices_from(pair_weights)
                pair_weights[diagonal] += endogamy * shares
                for first_index, first_tribe in enumerate(tribe_names):
                    for second_index, second_tribe in enumerate(tribe_names):
                        if first_index == second_index:
                            continue
                        pair_weights[first_index, second_index] *= self._cross_clan_openness(
                            first_tribe,
                            adult_genotypes_by_tribe[first_tribe],
                            second_tribe,
                            adult_genotypes_by_tribe[second_tribe],
                        )
                pair_probabilities = (pair_weights / pair_weights.sum()).reshape(-1)
                pair_birth_totals = self.rng.multinomial(births, pair_probabilities).reshape(
                    len(tribe_names), len(tribe_names))
                cross_tribe_births = 0
                pair_details = []
                for first_index, first_tribe in enumerate(tribe_names):
                    for second_index, second_tribe in enumerate(tribe_names):
                        pair_births = int(pair_birth_totals[first_index, second_index])
                        if pair_births == 0:
                            continue
                        genotype_counts = np.asarray(genetics.birth_counts_from_parent_pools(
                            adult_genotypes_by_tribe[first_tribe],
                            adult_genotypes_by_tribe[second_tribe], pair_births, self.rng),
                            dtype=np.int64)
                        if first_index == second_index:
                            valley.ensure_tribe(first_tribe)[
                                ESTABLISHED, demography.CHILD, :] += genotype_counts
                            continue
                        cross_tribe_births += pair_births
                        parent_key = frozenset((first_tribe, second_tribe))
                        mixed_name = self.mixed_clans.get(parent_key)
                        openness = self._cross_clan_openness(
                            first_tribe,
                            adult_genotypes_by_tribe[first_tribe],
                            second_tribe,
                            adult_genotypes_by_tribe[second_tribe],
                        )
                        mixed_clan_chance = min(
                            1.0,
                            self.cfg.tribes.mixed_clan_chance * openness,
                        )
                        if (mixed_name is None
                            and "-" not in first_tribe
                            and "-" not in second_tribe
                            and self.rng.random() < mixed_clan_chance):
                            existing_names = {
                                name for current_valley in self.valleys
                                for name in current_valley.cohorts
                            }
                            mixed_name = tribes.unique_mixed_name(
                                first_tribe, second_tribe, existing_names)
                            self.mixed_clans[parent_key] = mixed_name
                            self.cfg.traits.clans[mixed_name] = tribes.mixed_culture(
                                self.cfg.traits.culture(first_tribe),
                                self.cfg.traits.culture(second_tribe),
                                self.cfg.tribes.culture_mutation_band, self.rng)
                            self._register_clan(
                                mixed_name,
                                turn,
                                "merge",
                                parents=[first_tribe, second_tribe],
                                valley=valley.name,
                            )
                            events_out.append({
                                "turn": turn,
                                "year": turn * self.cfg.years_per_turn,
                                "type": "clan_merge",
                                "valley": valley.name,
                                "first": first_tribe,
                                "second": second_tribe,
                                "new_clan": mixed_name,
                            })
                        first, second = tribes.allocate_mixed_children(
                            genotype_counts, 0.5, shares[first_index], self.cfg.tribes, self.rng)
                        if mixed_name is not None:
                            valley.ensure_tribe(mixed_name)[
                                ESTABLISHED, demography.CHILD, :] += genotype_counts
                        else:
                            valley.ensure_tribe(first_tribe)[
                                ESTABLISHED, demography.CHILD, :] += first
                            valley.ensure_tribe(second_tribe)[
                                ESTABLISHED, demography.CHILD, :] += second
                        pair_details.append({
                            "first": first_tribe, "second": second_tribe,
                            "children": pair_births, "clan": mixed_name,
                        })
            else:
                cross_tribe_births = 0
            ledgers.append({"valley": valley.name, "births": births,
                            "cross_tribe_births": cross_tribe_births})
        return events_out, ledgers

    def _advance_residency_and_land(self) -> None:
        recovery = self.cfg.volcano.land_recovery_fraction
        for valley in self.valleys:
            for counts in valley.cohorts.values():
                counts[ESTABLISHED] += counts[NEWCOMER]
                counts[NEWCOMER] = 0
            valley.land_health += (1.0 - valley.land_health) * recovery
            valley.land_health = min(max(valley.land_health, 0.0), 1.0)
            valley.shock_memory *= self.cfg.tribes.shock_memory_decay
            if valley.ash_bonus_turns > 0:
                valley.ash_bonus_turns -= 1

    def step(self, turn: int) -> None:
        starts = [valley.population for valley in self.valleys]
        clans_before = self._island_clan_populations()
        arrival_events = self._arrivals(turn)
        event_batch = self._erupt(turn)
        demographic_ledgers = self._food_and_demography()
        migration_events, migration_ledgers = self._migrate(turn)
        conflict_events, conflict_ledgers = self._conflicts(turn)
        birth_events, birth_ledgers = self._births(turn)
        expedition_events = self._move_or_depart_arrivals(turn)
        self._advance_residency_and_land()
        lifecycle_events = self._record_clan_extinctions(clans_before, turn)
        self.event_log.extend(
            arrival_events + event_batch + migration_events + conflict_events + birth_events
            + expedition_events + lifecycle_events
        )
        ledgers = []
        for index, valley in enumerate(self.valleys):
            ledgers.append({
                "valley": valley.name,
                "starting_population": starts[index],
                **demographic_ledgers[index],
                **migration_ledgers[index],
                **conflict_ledgers[index],
                "assimilated": 0,
                **birth_ledgers[index],
                "ending_population": valley.population,
                "ending_food": valley.food_stock,
            })
        self.history.append(self._snapshot(turn, ledgers))

    def _snapshot(self, turn: int, ledgers: Optional[List[dict]] = None) -> dict:
        island_population = sum(valley.population for valley in self.valleys)
        valleys = []
        population_by_clan: Dict[str, int] = {}
        bb_population_by_clan: Dict[str, int] = {}
        bb_count_by_clan: Dict[str, int] = {}
        for valley in self.valleys:
            q = valley.q
            clan_populations = {name: int(counts.sum())
                                for name, counts in valley.cohorts.items()}
            clan_bb_counts = {name: int(counts[:, :, 2].sum())
                              for name, counts in valley.cohorts.items()}
            for name, population in clan_populations.items():
                population_by_clan[name] = population_by_clan.get(name, 0) + population
                bb_count = clan_bb_counts.get(name, 0)
                bb_population_by_clan[name] = bb_population_by_clan.get(name, 0) + bb_count
                bb_count_by_clan[name] = bb_count_by_clan.get(name, 0) + bb_count
            trait_populations = {
                trait: {
                    "positive": sum(
                        clan_populations.get(clan, 0)
                        for clan in clan_populations
                        if getattr(self.cfg.traits.culture(clan), trait) > 0
                    ),
                    "negative": sum(
                        clan_populations.get(clan, 0)
                        for clan in clan_populations
                        if getattr(self.cfg.traits.culture(clan), trait) < 0
                    ),
                    "neutral": sum(
                        clan_populations.get(clan, 0)
                        for clan in clan_populations
                        if getattr(self.cfg.traits.culture(clan), trait) == 0
                    ),
                }
                for trait in ("warlike", "peaceful", "strength_in_numbers",
                              "strong_individuals", "xenophile", "xenophobic",
                              "harsh_discipline", "nurturing", "migratory", "stationary")
            }
            valleys.append({
                "name": valley.name,
                "population": valley.population,
                "newcomers": sum(int(counts[NEWCOMER].sum()) for counts in valley.cohorts.values()),
                "children": int(valley.stage_counts[demography.CHILD]),
                "adults": int(valley.stage_counts[demography.ADULT]),
                "elders": int(valley.stage_counts[demography.ELDER]),
                "n_BB": int(valley.genotype_counts[0]),
                "n_Bb": int(valley.genotype_counts[1]),
                "n_bb": int(valley.genotype_counts[2]),
                "q": q,
                "food_stock": valley.food_stock,
                "storage_limit": valley.storage_limit,
                "food_security": valley.food_security,
                "land_health": valley.land_health,
                "ash_bonus_turns": valley.ash_bonus_turns,
                "shock_memory": valley.shock_memory,
                "surplus_memory": valley.surplus_memory,
                "eruption": valley.erupted_this_turn,
                "tribes": clan_populations,
                "clan_bb_counts": clan_bb_counts,
                "clan_traits": {
                    clan: asdict(self.cfg.traits.culture(clan))
                    for clan in clan_populations
                },
                "trait_populations": trait_populations,
            })
        return {
            "turn": turn,
            "year": turn * self.cfg.years_per_turn,
            "valleys": valleys,
            "ledger": ledgers or [],
            "totals": {
                "population": island_population,
                "deaths_by_source": dict(self.cumulative_deaths_by_source),
                "baseline_deaths_by_stage": dict(self.cumulative_baseline_deaths_by_stage),
                "deaths_by_genotype": dict(self.cumulative_deaths_by_genotype),
                "deaths_by_clan": dict(self.cumulative_deaths_by_clan),
                "population_by_clan": population_by_clan,
                "bb_population_by_clan": bb_population_by_clan,
                "bb_share_by_clan": {
                    clan: (bb_count_by_clan[clan] / population_by_clan[clan])
                    for clan in population_by_clan
                    if population_by_clan[clan] > 0
                },
                "clan_lifecycles": {
                    clan: dict(details)
                    for clan, details in self.clan_lifecycles.items()
                },
            },
        }

    def run(self, turns: Optional[int] = None) -> List[dict]:
        total_turns = self.cfg.turns if turns is None else turns
        self._reset_tracking()
        self.history = [self._snapshot(0)]
        for turn in range(1, total_turns + 1):
            self.step(turn)
        return self.history

    def provenance(self) -> dict:
        return {
            "model": "island-food-cohort-v1",
            "seed": self.cfg.seed,
            "config": asdict(self.cfg),
        }