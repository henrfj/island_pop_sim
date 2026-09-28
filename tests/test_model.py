import unittest
import numpy as np

from island import conflict, demography, genetics, resources, tribes
from island.config import (ArrivalEventConfig, CultureTraits, SimulationConfig,
                           default_valleys, tideborn_defensive_scenario)
from island.run_sim import build_run_config
from island.world import Island


class PrimitiveTests(unittest.TestCase):
    def test_opposing_culture_traits_cancel_to_one_side(self):
        traits = CultureTraits(
            warlike=0.10,
            peaceful=0.10,
            strength_in_numbers=0.15,
            strong_individuals=0.05,
            xenophile=0.20,
            xenophobic=0.05,
            harsh_discipline=0.08,
            nurturing=0.03,
            migratory=0.20,
            stationary=0.05,
        )
        self.assertEqual(traits.warlike, 0.0)
        self.assertEqual(traits.peaceful, 0.0)
        self.assertAlmostEqual(traits.strength_in_numbers, 0.10)
        self.assertAlmostEqual(traits.strong_individuals, 0.0)
        self.assertAlmostEqual(traits.xenophile, 0.15)
        self.assertAlmostEqual(traits.xenophobic, 0.0)
        self.assertAlmostEqual(traits.harsh_discipline, 0.05)
        self.assertAlmostEqual(traits.nurturing, 0.0)
        self.assertAlmostEqual(traits.migratory, 0.15)
        self.assertAlmostEqual(traits.stationary, 0.0)

    def test_cohort_transition_accounts_for_deaths(self):
        rng = np.random.default_rng(1)
        counts = np.full((3, 3), 20, dtype=np.int64)
        result = demography.transition_cohorts(counts, 0.4, SimulationConfig().demography, rng)
        self.assertEqual(
            int(counts.sum()),
            int(result.counts.sum()) + result.baseline_deaths + result.shortage_deaths,
        )
        self.assertTrue(np.all(result.counts >= 0))

    def test_food_balance(self):
        initial_stock = 450.0
        result = resources.update_food(
            initial_stock, 500.0, 400.0, 0.8, 200, 100, 40, SimulationConfig().food)
        self.assertAlmostEqual(
            initial_stock + result.produced - result.spoiled,
            result.consumed + result.ending_stock,
        )
        self.assertGreaterEqual(result.unmet, 0.0)

    def test_parent_pools_control_offspring_genotype(self):
        rng = np.random.default_rng(2)
        children = genetics.birth_counts_from_parent_pools(
            np.array([100, 0, 0]), np.array([0, 0, 100]), 100, rng)
        self.assertEqual(children, (0, 100, 0))

    def test_single_clan_can_split_across_factions(self):
        rng = np.random.default_rng(3)
        counts = np.full((2, 3, 3), 20, dtype=np.int64)
        split = conflict.partition_factions({"Tideborn": counts}, 0.3, rng)
        self.assertGreater(split.first["Tideborn"].sum(), 0)
        self.assertGreater(split.second["Tideborn"].sum(), 0)
        np.testing.assert_array_equal(
            counts, split.first["Tideborn"] + split.second["Tideborn"])

    def test_branch_names_are_unique_and_lineage_readable(self):
        cfg = SimulationConfig().tribes
        existing = {"Tideborn", "Tideborn West"}
        name = tribes.unique_branch_name("Tideborn", existing, cfg)
        self.assertTrue(name.startswith("Tideborn "))
        self.assertNotIn(name, existing)

    def test_mixed_culture_is_mutated_but_bounded(self):
        first = CultureTraits(warlike=0.10)
        second = CultureTraits(stationary=0.10)
        mixed = tribes.mixed_culture(first, second, 1.0, np.random.default_rng(4))
        self.assertTrue(0.0 <= mixed.warlike <= 0.20)
        self.assertTrue(0.0 <= mixed.stationary <= 0.20)
        self.assertNotEqual(mixed, first)


class IslandTests(unittest.TestCase):
    def make_island(self, turns=12, eruptions=True):
        cfg = SimulationConfig(turns=turns)
        if not eruptions:
            cfg.volcano.eruption_interval_turns = 0
        return Island(default_valleys(), cfg, np.random.default_rng(cfg.seed))

    def test_seeded_run_is_deterministic(self):
        first = self.make_island()
        second = self.make_island()
        self.assertEqual(first.run(), second.run())
        self.assertEqual(first.event_log, second.event_log)

    def test_snapshot_invariants(self):
        island = self.make_island()
        for snapshot in island.run():
            self.assertIn("totals", snapshot)
            self.assertIn("deaths_by_source", snapshot["totals"])
            self.assertIn("baseline_deaths_by_stage", snapshot["totals"])
            self.assertIn("deaths_by_genotype", snapshot["totals"])
            self.assertIn("deaths_by_clan", snapshot["totals"])
            self.assertIn("population_by_clan", snapshot["totals"])
            self.assertIn("bb_population_by_clan", snapshot["totals"])
            self.assertIn("bb_share_by_clan", snapshot["totals"])
            for valley in snapshot["valleys"]:
                self.assertEqual(valley["population"], valley["n_BB"] + valley["n_Bb"] + valley["n_bb"])
                self.assertEqual(valley["population"], valley["children"] + valley["adults"] + valley["elders"])
                self.assertGreaterEqual(valley["food_stock"], 0.0)
                self.assertGreaterEqual(valley["land_health"], 0.0)
                self.assertLessEqual(valley["land_health"], 1.0)
                self.assertIn("trait_populations", valley)
                self.assertIn("clan_traits", valley)
                self.assertIn("clan_bb_counts", valley)

    def test_arrival_clan_lands_and_leaves_on_schedule(self):
        cfg = SimulationConfig(turns=6)
        cfg.arrivals = (
            ArrivalEventConfig(
                turn=2,
                valley="Landfall",
                clan="Blackwake",
                init_bb=80,
                dwell_min_turns=2,
                dwell_max_turns=2,
                departure_fraction=1.0,
                mass_move_chance=0.0,
            ),
        )
        island = Island(default_valleys(), cfg, np.random.default_rng(cfg.seed))
        island.run()
        arrival_events = [event for event in island.event_log if event["type"] == "clan_arrival"]
        departure_events = [event for event in island.event_log if event["type"] == "clan_departure"]
        self.assertEqual([event["turn"] for event in arrival_events], [2])
        self.assertEqual([event["turn"] for event in departure_events], [4])
        self.assertEqual(island.history[-1]["totals"]["population_by_clan"].get("Blackwake", 0), 0)

    def test_tideborn_start_xenophile_and_migratory(self):
        tideborn = SimulationConfig().traits.culture("Tideborn")
        self.assertAlmostEqual(tideborn.xenophile, 0.10)
        self.assertAlmostEqual(tideborn.migratory, 0.10)
        self.assertEqual(tideborn.xenophobic, 0.0)

    def test_tideborn_defensive_scenario_reconfigures_founders(self):
        cfg, _ = tideborn_defensive_scenario()
        tideborn = cfg.traits.culture("Tideborn")
        self.assertAlmostEqual(tideborn.stationary, 0.10)
        self.assertAlmostEqual(tideborn.strength_in_numbers, 0.10)
        self.assertAlmostEqual(tideborn.xenophobic, 0.10)
        self.assertEqual(tideborn.migratory, 0.0)

    def test_blackwake_cli_override_can_disable_arrival(self):
        cfg, _ = build_run_config(turns=20, scenario_name="main", blackwake="off")
        self.assertEqual(cfg.arrivals, ())

    def test_refugees_prefer_more_xenophilic_destinations(self):
        island = self.make_island(turns=1)
        displaced = np.zeros((2, 3, 3), dtype=np.int64)
        displaced[1, demography.ADULT, 2] = 100

        island.valleys[1].cohorts = {"Cloudfolk": np.zeros((2, 3, 3), dtype=np.int64)}
        island.valleys[2].cohorts = {"Reedkin": np.zeros((2, 3, 3), dtype=np.int64)}
        island.valleys[3].cohorts = {"Ashclan": np.zeros((2, 3, 3), dtype=np.int64)}

        island.valleys[1].cohorts["Cloudfolk"][1, demography.ADULT, 0] = 200
        island.valleys[2].cohorts["Reedkin"][1, demography.ADULT, 0] = 200
        island.valleys[3].cohorts["Ashclan"][1, demography.ADULT, 0] = 200

        island.cfg.traits.clans["Cloudfolk"] = CultureTraits(xenophile=0.35)
        island.cfg.traits.clans["Reedkin"] = CultureTraits(xenophobic=0.35)
        island.cfg.traits.clans["Ashclan"] = CultureTraits()

        picks = [island._refugee_destination(0, {"Tideborn": displaced}) for _ in range(120)]
        self.assertGreater(picks.count(1), picks.count(2))

    def test_cross_clan_births_can_create_mixed_clan(self):
        island = self.make_island(turns=1, eruptions=False)
        island.cfg.tribes.mixed_clan_chance = 1.0
        island.valleys[0].cohorts["Cloudfolk"] = island.valleys[0].cohorts["Tideborn"].copy()
        events, _ = island._births(1)
        self.assertTrue(any("-" in name for name in island.cfg.traits.clans))
        self.assertTrue(any(event["type"] == "clan_merge" for event in events))
        self.assertFalse(any(event["type"] == "mixed_families" for event in events))

    def test_calm_baseline_does_not_use_shortage_mortality(self):
        island = self.make_island(turns=40, eruptions=False)
        history = island.run()
        shortage_deaths = sum(
            row["shortage_deaths"] for snapshot in history[1:] for row in snapshot["ledger"])
        self.assertEqual(shortage_deaths, 0)
        final_population = sum(v["population"] for v in history[-1]["valleys"])
        self.assertGreater(final_population, 1500)
        self.assertLess(final_population, 2600)

    def test_first_eruption_is_recorded(self):
        island = self.make_island(turns=8)
        island.run()
        eruptions = [event for event in island.event_log if event["type"] == "eruption"]
        self.assertEqual(len(eruptions), 2)
        self.assertEqual([event["turn"] for event in eruptions], [4, 8])
        self.assertGreater(eruptions[0]["food_destroyed"], 0.0)

    def test_conflict_conserves_genotypes_without_casualties(self):
        island = self.make_island(turns=1)
        cfg = island.cfg.conflict
        cfg.base_chance = 1.0
        cfg.shortage_pressure = 0.0
        cfg.winner_casualty_rate = 0.0
        cfg.loser_casualty_rate = 0.0
        cfg.displacement_fraction = 0.25
        cfg.secession_chance_severe = 0.0
        before = sum((v.genotype_counts for v in island.valleys), np.zeros(3, dtype=np.int64))

        events, ledger = island._conflicts(1)

        after = sum((v.genotype_counts for v in island.valleys), np.zeros(3, dtype=np.int64))
        np.testing.assert_array_equal(before, after)
        self.assertEqual(sum(row["war_deaths"] for row in ledger), 0)
        self.assertTrue(any(event["type"] == "war" for event in events))
        self.assertTrue(any(event["type"] == "refugees" for event in events))


if __name__ == "__main__":
    unittest.main()