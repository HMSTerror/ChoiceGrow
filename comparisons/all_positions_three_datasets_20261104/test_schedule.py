"""Exercise phase isolation, dependency reuse, seed aggregation, and fixed method semantics."""
from copy import deepcopy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import run_suite as suite


class SchedulingTests(unittest.TestCase):
    def jobs(self):
        return [{"id": f"{dataset}{seed}", "phase": "main", "state": "complete"}
                for seed in suite.SEEDS for dataset in suite.DATASETS]

    def test_factorial_waits_for_all_nine_complete_results(self):
        jobs = self.jobs()
        self.assertTrue(suite.ready_for_factorial(jobs))
        for state in ("training", "evaluating", "running", "failed", "queued"):
            changed = deepcopy(jobs)
            changed[-1]["state"] = state
            self.assertFalse(suite.ready_for_factorial(changed))
        self.assertFalse(suite.ready_for_factorial(jobs[:-1]))

    def test_only_current_phase_can_launch_and_hold_prevents_launches(self):
        jobs = [{"id": "main", "phase": "main", "state": "queued"},
                {"id": "factorial", "phase": "factorial", "state": "queued"},
                {"id": "reuse", "phase": "factorial", "state": "waiting_dependency"}]
        manifest = {"phase": "main", "jobs": jobs}
        self.assertEqual([job["id"] for job in suite.launchable_jobs(manifest, {})], ["main"])
        self.assertEqual(suite.launchable_jobs(manifest, {"hold_new_jobs": True}), [])
        manifest["phase"] = "factorial"
        self.assertEqual([job["id"] for job in suite.launchable_jobs(manifest, {})], ["factorial"])

    def test_transition_reuses_three_main_runs_without_any_worker_launch(self):
        manifest = {"phase": "main", "jobs": self.jobs() + [
            {"id": f"reuse{seed}", "phase": "factorial", "state": "waiting_dependency"} for seed in suite.SEEDS]}
        with patch.object(suite, "complete_dependency_reuse") as reuse, patch.object(suite, "save_manifest"), \
                patch.object(suite, "summarize"), patch.object(suite.subprocess, "Popen") as launch:
            suite.advance_phase(manifest)
        self.assertEqual(reuse.call_count, 3)
        self.assertEqual(manifest["phase"], "factorial")
        launch.assert_not_called()

    def test_all_configs_use_the_fixed_statistics_and_current_training_protocol(self):
        for dataset in suite.DATASETS:
            for seed in suite.SEEDS:
                for arm in suite.ARMS:
                    cfg = suite.p.make_config(dataset, seed, arm, suite.NEIGHBORS, suite.BACKGROUND)
                    self.assertEqual(cfg["neighbor_statistics"], "all_positions")
                    self.assertEqual(cfg["background_statistics"], "prefix")
                    self.assertEqual(cfg["training"]["learning_rate"], .0006)
                    self.assertEqual(cfg["training"]["patience"], 150)
                    self.assertEqual(cfg["diffusion"]["partition_seed"], 100)
                    self.assertFalse(cfg["tf32"])
                    if arm.endswith("community"):
                        self.assertEqual(cfg["diffusion"]["initial_shares"], [.2165, .7835])
                        self.assertTrue(cfg["diffusion"]["dynamic_rate_gate"])
                    else:
                        self.assertEqual(cfg["diffusion"]["level_rates"], [cfg["diffusion"]["total_budget"]])
                        self.assertFalse(cfg["diffusion"]["dynamic_rate_gate"])

    def test_seed_statistics_are_not_sampling_stream_statistics(self):
        result = suite.p.moments([.01, .02, .03])
        self.assertAlmostEqual(result["mean"], .02)
        self.assertAlmostEqual(result["std"], .01)
        self.assertEqual(result["n"], 3)


if __name__ == "__main__":
    unittest.main(verbosity=2)
