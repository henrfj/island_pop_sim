import unittest
import tempfile
from pathlib import Path
import numpy as np

from island import conflict, demography, genetics, resources, tribes
from island.config import (ArrivalEventConfig, CultureTraits, SimulationConfig,
                           civil_war_scenario, default_valleys,
                           tideborn_defensive_scenario)
from island.run_sim import build_run_config
from island.viz.export_html import export_history_html
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
            syncretic=0.12,
            insular=0.04,
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
        self.assertAlmostEqual(traits.syncretic, 0.08)
        self.assertAlmostEqual(traits.insular, 0.0)
        self.assertAlmostEqual(traits.harsh_discipline, 0.05)
        self.assertAlmostEqual(traits.nurturing, 0.0)
        self.assertAlmostEqual(traits.migratory, 0.15)
        self.assertAlmostEqual(traits.stationary, 0.0)

    def test_ancestry_effects_mark_bb_as_island_rooted(self):
        genes = SimulationConfig().traits.genes
        self.assertGreater(genes.island_immunity[0], genes.island_immunity[2])
        self.assertLess(genes.frontier_drive[0], genes.frontier_drive[2])

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

    def test_mixed_names_are_canonical_and_compact(self):
        culture = CultureTraits(agrarian=0.30, nurturing=0.20)
        name = tribes.unique_mixed_name(
            "Tideborn-Ashclan",
            "Cloudfolk",
            "Cloudfolk",
            culture,
            set(),
            np.random.default_rng(7),
        )
        self.assertTrue(name.endswith(" Cloud"))
        self.assertTrue(name.split(" ", 1)[0] in {"Field", "Harvest", "Loam"})


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
                self.assertIn("habitability", valley)
                self.assertIn("dominant_clan", valley)
                self.assertIn("dominant_traits", valley)

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

    def test_default_landfall_now_starts_at_500_people(self):
        landfall = default_valleys()[0]
        self.assertEqual(landfall.init_bb, 500)

    def test_founding_clans_get_signature_plus_random_bonus_trait(self):
        island = self.make_island(turns=1, eruptions=False)
        expected_signatures = {
            "Tideborn": "migratory",
            "Cloudfolk": "agrarian",
            "Reedkin": "nurturing",
            "Ashclan": "warlike",
        }
        for clan, signature in expected_signatures.items():
            traits = island.cfg.traits.culture(clan)
            active_traits = [
                name for name, value in traits.__dict__.items()
                if value > 0
            ]
            self.assertIn(signature, active_traits)
            self.assertEqual(len(active_traits), 2)

    def test_tideborn_defensive_scenario_reconfigures_founders(self):
        cfg, _ = tideborn_defensive_scenario()
        tideborn = cfg.traits.culture("Tideborn")
        self.assertAlmostEqual(tideborn.stationary, 1.0)
        self.assertAlmostEqual(tideborn.strength_in_numbers, 0.25)
        self.assertAlmostEqual(tideborn.xenophobic, 1.0)
        self.assertEqual(tideborn.migratory, 0.0)
        self.assertEqual(cfg.arrivals, ())

    def test_main_scenario_has_no_blackwake_by_default(self):
        cfg, _ = build_run_config(turns=20, scenario_name="main")
        self.assertEqual(cfg.arrivals, ())

    def test_blackwake_cli_override_can_enable_arrival(self):
        cfg, _ = build_run_config(turns=20, scenario_name="main", blackwake="on")
        self.assertEqual(len(cfg.arrivals), 1)
        self.assertEqual(cfg.arrivals[0].clan, "Blackwake")

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
        created_names = [event["new_clan"] for event in events if event["type"] == "clan_merge"]
        self.assertTrue(created_names)
        self.assertTrue(all(name in island.cfg.traits.clans for name in created_names))
        self.assertTrue(any(event["type"] == "clan_merge" for event in events))
        self.assertFalse(any(event["type"] == "mixed_families" for event in events))

    def test_mixed_clans_can_merge_again_in_later_generations(self):
        island = self.make_island(turns=1, eruptions=False)
        island.cfg.tribes.mixed_clan_chance = 1.0
        template = island.valleys[0].cohorts["Tideborn"].copy()
        island.valleys[0].cohorts["Cloudfolk"] = template.copy()
        events, _ = island._births(1)
        first_mixed = next(event["new_clan"] for event in events if event["type"] == "clan_merge")

        island.valleys[0].cohorts[first_mixed] = template.copy()
        island.valleys[0].cohorts["Ashclan"] = template.copy()
        events, _ = island._births(2)

        later_merges = [event for event in events if event["type"] == "clan_merge"]
        self.assertTrue(any(first_mixed in (event["first"], event["second"]) for event in later_merges))
        self.assertTrue(any(event["new_clan"] != first_mixed for event in later_merges))

    def test_calm_baseline_does_not_use_shortage_mortality(self):
        island = self.make_island(turns=40, eruptions=False)
        history = island.run()
        shortage_deaths = sum(
            row["shortage_deaths"] for snapshot in history[1:] for row in snapshot["ledger"])
        self.assertEqual(shortage_deaths, 0)
        final_population = sum(v["population"] for v in history[-1]["valleys"])
        self.assertGreater(final_population, 1500)
        self.assertLess(final_population, 5000)

    def test_first_eruption_is_recorded(self):
        island = self.make_island(turns=8)
        island.run()
        eruptions = [event for event in island.event_log if event["type"] == "eruption"]
        self.assertEqual(len(eruptions), 2)
        self.assertEqual([event["turn"] for event in eruptions], [4, 8])
        self.assertGreater(eruptions[0]["food_destroyed"], 0.0)

    def test_tideborn_defensive_scenario_keeps_tideborn_alive_at_landfall(self):
        cfg, valleys = tideborn_defensive_scenario()
        cfg.turns = 120
        island = Island(valleys, cfg, np.random.default_rng(cfg.seed))
        island.run()
        landfall = next(valley for valley in island.history[-1]["valleys"] if valley["name"] == "Landfall")
        self.assertGreater(landfall["tribes"].get("Tideborn", 0), 0)
        self.assertEqual(landfall["dominant_clan"], "Tideborn")

    def test_civil_war_scenario_triggers_succession_crisis_and_wars(self):
        cfg, valleys = civil_war_scenario()
        cfg.civil_war.trigger_turn = 2
        cfg.turns = 3
        island = Island(valleys, cfg, np.random.default_rng(cfg.seed))
        island.run()
        crisis_events = [event for event in island.event_log if event["type"] == "succession_crisis"]
        civil_war_splits = [
            event for event in island.event_log
            if event["type"] == "clan_split" and event.get("reason") == "civil_war"
        ]
        post_trigger_wars = [
            event for event in island.event_log
            if event["type"] == "war" and event["turn"] >= cfg.civil_war.trigger_turn
        ]
        self.assertTrue(crisis_events)
        self.assertTrue(civil_war_splits)
        self.assertTrue(post_trigger_wars)

    def test_exported_replay_contains_all_history_snapshots(self):
        island = self.make_island(turns=4, eruptions=False)
        history = island.run()
        with tempfile.TemporaryDirectory() as tmpdir:
            output = export_history_html(
                history,
                island.event_log,
                [valley.name for valley in island.valleys],
                island.provenance(),
                path=str(Path(tmpdir) / "replay.html"),
            )
            html = Path(output).read_text(encoding="utf-8")
        self.assertIn(f'"turn": {history[-1]["turn"]}', html)
        self.assertIn('"clan_distribution"', html)

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
        war_events = [event for event in events if event["type"] == "war"]
        self.assertTrue(war_events)
        self.assertIn("winner_clans", war_events[0])
        self.assertIn("loser_clans", war_events[0])
        self.assertTrue(war_events[0]["winner_clans"])
        self.assertTrue(any(event["type"] == "refugees" for event in events))

    def test_small_diaspora_clan_can_be_absorbed_without_war(self):
        island = self.make_island(turns=1, eruptions=False)
        island.cfg.tribes.political_absorption_chance = 1.0
        island.cfg.tribes.political_split_chance = 0.0
        valley = island.valleys[0]
        valley.cohorts = {
            "Cloudfolk": np.zeros((2, 3, 3), dtype=np.int64),
            "Tideborn West": np.zeros((2, 3, 3), dtype=np.int64),
        }
        valley.cohorts["Cloudfolk"][1, demography.ADULT, 0] = 120
        valley.cohorts["Tideborn West"][1, demography.ADULT, 1] = 10
        island.cfg.traits.clans["Cloudfolk"] = CultureTraits(
            strength_in_numbers=0.8,
            xenophobic=0.4,
            stationary=0.6,
            harsh_discipline=0.3,
        )
        island.cfg.traits.clans["Tideborn West"] = CultureTraits(
            xenophile=0.8,
            migratory=0.8,
            peaceful=0.4,
        )
        island._register_clan("Tideborn West", 1, "split", parents=["Tideborn"], valley="Landfall")

        events, _ = island._clan_dynamics(6)

        self.assertTrue(any(event["type"] == "clan_absorbed" for event in events))
        self.assertNotIn("Tideborn West", island.valleys[0].cohorts)

    def test_dispersed_low_cohesion_clan_can_split_without_war(self):
        island = self.make_island(turns=1, eruptions=False)
        island.cfg.tribes.political_absorption_chance = 0.0
        island.cfg.tribes.political_split_chance = 1.0
        island.cfg.tribes.minimum_independent_population = 20

        home = np.zeros((2, 3, 3), dtype=np.int64)
        enclave = np.zeros((2, 3, 3), dtype=np.int64)
        host = np.zeros((2, 3, 3), dtype=np.int64)
        home[1, demography.ADULT, 1] = 90
        enclave[1, demography.ADULT, 1] = 60
        host[1, demography.ADULT, 0] = 120

        island.valleys[0].cohorts = {"Tideborn": home.copy()}
        island.valleys[1].cohorts = {"Tideborn": enclave.copy(), "Cloudfolk": host.copy()}
        island.cfg.traits.clans["Tideborn"] = CultureTraits(
            strong_individuals=0.9,
            migratory=0.9,
            xenophile=0.4,
        )
        island.cfg.traits.clans["Cloudfolk"] = CultureTraits(
            strength_in_numbers=0.8,
            xenophobic=0.5,
            stationary=0.7,
        )

        events, _ = island._clan_dynamics(5)

        self.assertTrue(any(event["type"] == "clan_split" and event.get("reason") == "political" for event in events))
        self.assertTrue(any(name.startswith("Tideborn ") for name in island.valleys[1].cohorts if name != "Tideborn"))


if __name__ == "__main__":
    unittest.main()