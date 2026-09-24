"""Five-year island simulation with overlapping cohorts, food stocks, and tribes."""
from dataclasses import asdict, dataclass, field
from typing import Dict, List, Optional
import numpy as np

from . import conflict, demography, genetics, resources, sampling, tribes
from .config import SimulationConfig, ValleySpec

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
        self.path_neighbors = {i: {(i - 1) % self.n, (i + 1) % self.n} for i in range(self.n)}
        self.history: List[dict] = []
        self.event_log: List[dict] = []

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

    def _erupt(self, turn: int) -> List[dict]:
        for valley in self.valleys:
            valley.erupted_this_turn = False
        interval = self.cfg.volcano.eruption_interval_turns
        if interval <= 0 or turn <= 0 or turn % interval != 0:
            return []
        index = (turn // interval) % self.n
        valley = self.valleys[index]
        cfg = self.cfg.volcano
        mortality = self.rng.uniform(cfg.immediate_mortality_min, cfg.immediate_mortality_max)
        deaths = self._thin_all(valley, 1.0 - mortality)
        store_fraction = self.rng.uniform(cfg.store_destruction_min, cfg.store_destruction_max)
        food_destroyed = valley.food_stock * store_fraction
        valley.food_stock -= food_destroyed
        land_damage = self.rng.uniform(cfg.land_damage_min, cfg.land_damage_max)
        valley.land_health *= 1.0 - land_damage
        valley.erupted_this_turn = True
        valley.ash_bonus_turns = cfg.ash_bonus_turns
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
            effective_land = valley.land_health
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
            baseline_deaths = 0
            shortage_deaths = 0
            for tribe, counts in valley.cohorts.items():
                transitioned = np.zeros_like(counts)
                for residency in (NEWCOMER, ESTABLISHED):
                    result = demography.transition_cohorts(
                        counts[residency], valley.food_security, self.cfg.demography, self.rng)
                    transitioned[residency] = result.counts
                    baseline_deaths += result.baseline_deaths
                    shortage_deaths += result.shortage_deaths
                valley.cohorts[tribe] = transitioned
            valley.clean_empty_tribes()
            ledgers.append({
                "valley": valley.name,
                "food_produced": food.produced,
                "food_spoiled": food.spoiled,
                "food_consumed": food.consumed,
                "food_unmet": food.unmet,
                "baseline_deaths": baseline_deaths,
                "shortage_deaths": shortage_deaths,
            })
        return ledgers

    def _destination(self, origin: int, clan: Optional[str] = None) -> Optional[int]:
        options = [index for index in range(self.n) if index != origin]
        weights = []
        for destination in options:
            valley = self.valleys[destination]
            outlook = 0.2 + valley.food_security + valley.food_stock / max(valley.storage_limit, 1.0)
            if clan is not None and clan in valley.cohorts:
                clan_share = int(valley.cohorts[clan].sum()) / max(valley.population, 1)
                outlook *= 1.0 + self.cfg.migration.clan_destination_pull * clan_share
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
            shortage = max(0.0, 1.0 - valley.food_security)
            if shortage > 0 and self.rng.random() < cfg.shortage_wave_chance * shortage:
                expected_households += cfg.shortage_wave_households * shortage
            household_count = int(self.rng.poisson(expected_households))
            for _ in range(household_count):
                tribe_names = [name for name, counts in available_by_tribe.items()
                               if counts.sum() >= cfg.family_size_min]
                if not tribe_names:
                    break
                tribe_weights = np.asarray(
                    [available_by_tribe[name].sum() for name in tribe_names], dtype=float)
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
                destination = self._destination(origin, tribe)
                if destination is None:
                    available_by_tribe[tribe] += selected
                    continue
                ledger[origin]["migration_attempted"] += family_size
                source_delta = deltas[origin].setdefault(tribe, np.zeros(SHAPE, dtype=np.int64))
                source_delta -= selected
                water = destination not in self.path_neighbors[origin]
                route = "canoe" if water else "path"
                if water and self.rng.random() < cfg.storm_chance:
                    ledger[origin]["storm_deaths"] += family_size
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
            probability = min(1.0, cfg.base_chance + cfg.shortage_pressure * shortage
                              + 0.10 * damage_pressure)
            if self.rng.random() >= probability:
                continue
            split = conflict.partition_factions(
                tribe_counts, self.cfg.tribes.clan_cohesion, self.rng)
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
            deaths = 0
            for clan in valley.cohorts:
                winners = self.rng.binomial(
                    winning_side[clan], 1.0 - cfg.winner_casualty_rate).astype(np.int64)
                losers = self.rng.binomial(
                    losing_side[clan], 1.0 - cfg.loser_casualty_rate).astype(np.int64)
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

            valley.cohorts = survivors_by_clan
            food_fraction = self.rng.uniform(cfg.food_destruction_min, cfg.food_destruction_max)
            food_destroyed = valley.food_stock * food_fraction
            valley.food_stock -= food_destroyed
            land_damage = self.rng.uniform(cfg.land_damage_min, cfg.land_damage_max)
            valley.land_health *= 1.0 - land_damage

            refugee_count = sum(int(counts.sum()) for counts in displaced_by_clan.values())
            destination = self._destination(index)
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
            if births > 0 and adults > 0:
                tribe_names = list(adult_by_tribe)
                shares = np.asarray([adult_by_tribe[name] for name in tribe_names], dtype=float)
                shares /= shares.sum()
                pair_weights = np.outer(shares, shares)
                endogamy = self.cfg.tribes.clan_endogamy
                pair_weights *= 1.0 - endogamy
                diagonal = np.diag_indices_from(pair_weights)
                pair_weights[diagonal] += endogamy * shares
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
                        first, second = tribes.allocate_mixed_children(
                            genotype_counts, 0.5, shares[first_index], self.cfg.tribes, self.rng)
                        valley.ensure_tribe(first_tribe)[
                            ESTABLISHED, demography.CHILD, :] += first
                        valley.ensure_tribe(second_tribe)[
                            ESTABLISHED, demography.CHILD, :] += second
                        pair_details.append({
                            "first": first_tribe, "second": second_tribe,
                            "children": pair_births,
                        })
                if pair_details:
                    events_out.append({
                        "turn": turn, "year": turn * self.cfg.years_per_turn,
                        "type": "mixed_families", "valley": valley.name,
                        "children": cross_tribe_births, "pairs": pair_details,
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
            if valley.ash_bonus_turns > 0:
                valley.ash_bonus_turns -= 1

    def step(self, turn: int) -> None:
        starts = [valley.population for valley in self.valleys]
        event_batch = self._erupt(turn)
        demographic_ledgers = self._food_and_demography()
        migration_events, migration_ledgers = self._migrate(turn)
        conflict_events, conflict_ledgers = self._conflicts(turn)
        birth_events, birth_ledgers = self._births(turn)
        self._advance_residency_and_land()
        self.event_log.extend(event_batch + migration_events + conflict_events + birth_events)
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
        valleys = []
        for valley in self.valleys:
            q = valley.q
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
                "eruption": valley.erupted_this_turn,
                "tribes": {name: int(counts.sum()) for name, counts in valley.cohorts.items()},
            })
        return {
            "turn": turn,
            "year": turn * self.cfg.years_per_turn,
            "valleys": valleys,
            "ledger": ledgers or [],
        }

    def run(self, turns: Optional[int] = None) -> List[dict]:
        total_turns = self.cfg.turns if turns is None else turns
        self.history = [self._snapshot(0)]
        self.event_log = []
        for turn in range(1, total_turns + 1):
            self.step(turn)
        return self.history

    def provenance(self) -> dict:
        return {
            "model": "island-food-cohort-v1",
            "seed": self.cfg.seed,
            "config": asdict(self.cfg),
        }