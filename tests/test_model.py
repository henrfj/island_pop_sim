import unittest
import numpy as np

from island import conflict, demography, genetics, resources, tribes
from island.config import SimulationConfig, default_valleys
from island.world import Island


class PrimitiveTests(unittest.TestCase):
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
            for valley in snapshot["valleys"]:
                self.assertEqual(valley["population"], valley["n_BB"] + valley["n_Bb"] + valley["n_bb"])
                self.assertEqual(valley["population"], valley["children"] + valley["adults"] + valley["elders"])
                self.assertGreaterEqual(valley["food_stock"], 0.0)
                self.assertGreaterEqual(valley["land_health"], 0.0)
                self.assertLessEqual(valley["land_health"], 1.0)

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