"""Check the predecessor barrier and the intended single-factor comparison."""
from copy import deepcopy
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import run_suite as suite


class FollowupTests(unittest.TestCase):
    def make_previous(self, root):
        root = Path(root)
        suite.p.write_json(root / "execution_control.json", {"automatic_factorial_after_main_complete": False})
        suite.p.write_json(root / "status.json", {"coordinator_pid": 123})
        for dataset in suite.DATASETS:
            for seed in suite.SEEDS:
                folder = root / "main" / dataset / f"seed_{seed}"
                folder.mkdir(parents=True)
                suite.p.write_json(folder / "worker_status.json", {"state": "complete"})
                for filename in suite.REQUIRED:
                    (folder / filename).touch()
        return root

    def test_barrier_waits_for_every_test_and_long_tail_artifact(self):
        with tempfile.TemporaryDirectory() as temporary, patch.object(suite.p, "pid_alive", return_value=False):
            root = self.make_previous(temporary)
            self.assertTrue(suite.dependency_state(root)["ready"])
            for filename in suite.REQUIRED:
                path = root / "main/Beauty/seed_102" / filename
                path.unlink()
                state = suite.dependency_state(root)
                self.assertFalse(state["ready"])
                self.assertEqual(state["completed"], 8)
                path.touch()

    def test_barrier_waits_for_coordinator_even_after_last_worker_completes(self):
        with tempfile.TemporaryDirectory() as temporary, patch.object(suite.p, "pid_alive", return_value=True):
            self.assertFalse(suite.dependency_state(self.make_previous(temporary))["ready"])

    def test_failed_predecessor_does_not_release_followup_training(self):
        with tempfile.TemporaryDirectory() as temporary, patch.object(suite.p, "pid_alive", return_value=False):
            root = self.make_previous(temporary)
            suite.p.write_json(root / "main/Sports/seed_102/worker_status.json", {"state": "failed"})
            with self.assertRaisesRegex(RuntimeError, "Predecessor jobs failed"):
                suite.dependency_state(root)

    def test_training_or_evaluating_predecessor_is_not_complete(self):
        with tempfile.TemporaryDirectory() as temporary, patch.object(suite.p, "pid_alive", return_value=False):
            root = self.make_previous(temporary)
            for state in ("training", "evaluating"):
                suite.p.write_json(root / "main/Toys/seed_102/worker_status.json", {"state": state})
                self.assertFalse(suite.dependency_state(root)["ready"])

    def test_cancelled_factorial_cannot_be_reenabled_silently(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = self.make_previous(temporary)
            suite.p.write_json(root / "execution_control.json", {"automatic_factorial_after_main_complete": True})
            with self.assertRaisesRegex(RuntimeError, "cancelled factorial"):
                suite.dependency_state(root)

    def test_only_background_statistics_differ_from_current_suite(self):
        for dataset in suite.DATASETS:
            for seed in suite.SEEDS:
                old_b = suite.p.make_config(dataset, seed, "neighbor_community", "all_positions", "target")
                new_b = suite.p.make_config(dataset, seed, "neighbor_community", "all_positions", "prefix")
                normalized = deepcopy(old_b)
                normalized["background_statistics"] = "prefix"
                self.assertEqual(normalized, new_b)
                self.assertEqual(old_b["diffusion"]["initial_shares"], [.2165, .7835])
                self.assertTrue(old_b["diffusion"]["dynamic_rate_gate"])

    def test_seed_standard_deviation_does_not_count_streams_as_training_runs(self):
        result = suite.p.moments([.01, .02, .03])
        self.assertEqual(result["n"], 3)
        self.assertAlmostEqual(result["mean"], .02)
        self.assertAlmostEqual(result["std"], .01)


if __name__ == "__main__":
    unittest.main(verbosity=2)
