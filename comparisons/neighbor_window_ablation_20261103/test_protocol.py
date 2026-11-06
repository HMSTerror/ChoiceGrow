"""Check statistical definitions, numerical ablations, and validation-only selection before training."""
from copy import deepcopy
import json
from pathlib import Path
import tempfile
import unittest

import run_suite as suite
import torch
from checkpoint import make_denoiser
from diffusion import ChoiceDiffusion
from neighbors import build_neighbors, neighbor_counts
from train import EMA, seed_all


class NeighborProtocolTests(unittest.TestCase):
    def test_each_position_pair_once_with_distance_weights(self):
        history = torch.tensor([[6, 0, 1, 2, 3]])
        target = torch.tensor([4])
        rows = neighbor_counts(history, target, 6, statistics="all_positions")
        expected = {0: {1: 1., 2: .5, 3: 1/3},
                    1: {2: 1., 3: .5, 4: 1/3}, 2: {3: 1., 4: .5}, 3: {4: 1.}}
        self.assertEqual(dict(rows), expected)
        self.assertEqual(rows[0][1], 1.)  # Earlier edges are not counted once per overlapping prefix.
        self.assertNotIn(4, rows[0])  # Distance four is outside the window.
        self.assertNotIn(6, rows)  # Padding is never an item.
        self.assertNotIn(4, rows)  # A target with no later position has no observed outgoing edge.

    def test_original_endpoint_counts_and_repeated_occurrences(self):
        sequence = torch.tensor([[5, 0, 1, 2], [5, 0, 1, 2]])
        target = torch.tensor([3, 3])
        old = neighbor_counts(sequence, target, 5)
        self.assertEqual(dict(old), {2: {3: 2.}, 1: {3: 1.}, 0: {3: 2/3}})
        repeated = neighbor_counts(torch.tensor([[5, 0, 1, 0]]), torch.tensor([1]), 5,
                                   statistics="all_positions")
        self.assertEqual(repeated[0][1], 2 + 1/3)
        self.assertEqual(repeated[1][0], 1.)

    def test_self_prior_topk_and_empty_row_fallback(self):
        history, target = torch.tensor([[6, 0, 1, 2, 3]]), torch.tensor([4])
        ids, weights = build_neighbors(history, target, 6, count=2, statistics="all_positions")
        self.assertEqual(ids[0].tolist(), [0, 1])  # Tie is broken by item ID.
        torch.testing.assert_close(weights[0], torch.tensor([.5, .5]))
        self.assertEqual(ids[5, 0].item(), 5)
        self.assertEqual(weights[5, 0].item(), 1.)
        torch.testing.assert_close(weights.sum(1), torch.ones(6))

    def test_background_prefix_labels_do_not_include_first_or_repeat_histories(self):
        counts = suite.prefix_counts({"seq": torch.tensor([[6, 0, 1, 2, 3]]),
                                      "target": torch.tensor([4])}, 6)
        self.assertEqual(counts.tolist(), [0, 1, 1, 1, 1, 0])


class SelectionTests(unittest.TestCase):
    def test_validation_only_ranking_is_normalized_across_datasets(self):
        with tempfile.TemporaryDirectory() as temp:
            jobs = []
            for index, (neighbors, background) in enumerate(suite.VARIANTS):
                for dataset, baseline in (("Toys", .04), ("Beauty", .05)):
                    output = Path(temp) / f"{index}_{dataset}"
                    output.mkdir()
                    # Variant 2 improves 10% on both validation sets and must win.
                    value = baseline * (1.1 if index == 2 else 1.)
                    (output / "training.json").write_text(json.dumps({"best_epoch": 10,
                        "best_validation": {"ndcg@10": value}}))
                    # Deliberately give a different variant an arbitrarily larger test score.
                    (output / "report.json").write_text(json.dumps({"test": {"ndcg@10": 999 if index == 3 else 0}}))
                    jobs.append({"phase": "screening", "dataset": dataset,
                                 "variant": suite.variant_name(neighbors, background), "output": str(output)})
            selection = suite.choose_variant({"jobs": jobs, "selection_metric": "mean_relative_best_validation_ndcg10"})
            self.assertEqual(selection["selected_variant"], "all_positions__target")
            self.assertFalse(selection["uses_test_metrics"])
            self.assertAlmostEqual(selection["scores"]["all_positions__target"], .1)

    def test_root_config_has_one_budget_and_no_gate(self):
        for arm in suite.ARMS:
            config = suite.make_config("Toys", 101, arm, "all_positions", "target")
            self.assertEqual(config["training"]["learning_rate"], .0006)
            self.assertEqual(config["training"]["patience"], 150)
            if arm.endswith("root"):
                self.assertNotIn("initial_shares", config["diffusion"])
                self.assertEqual(config["diffusion"]["level_rates"], [config["diffusion"]["total_budget"]])
                self.assertFalse(config["diffusion"]["dynamic_rate_gate"])
            else:
                self.assertEqual(config["diffusion"]["initial_shares"], [.2165, .7835])
                self.assertTrue(config["diffusion"]["dynamic_rate_gate"])

    def test_reuse_signature_preserves_data_protocol_parameters(self):
        config = suite.make_config("Toys", 100, "neighbor_community", "all_positions", "target")
        changed = deepcopy(config)
        changed["training"]["patience"] = 100
        self.assertNotEqual(suite.signature(config), suite.signature(changed))
        changed = deepcopy(config)
        changed["tf32"] = True
        self.assertNotEqual(suite.signature(config), suite.signature(changed))


class NumericalAblationTests(unittest.TestCase):
    def test_each_factorial_arm_trains_and_root_matches_dense_generator(self):
        torch.set_num_threads(2)
        for arm in suite.ARMS:
            with self.subTest(arm=arm), tempfile.TemporaryDirectory() as temp:
                output = Path(temp)
                config = suite.make_config("Toys", 100, arm, "all_positions", "target")
                # A small catalog tests mathematics and gradients without starting experimental training.
                config.update(history_length=4, hidden_size=16)
                n = 6
                history, target = torch.tensor([[6, 0, 1, 2], [6, 6, 3, 4]]), torch.tensor([3, 5])
                ids, weights = build_neighbors(history, target, n, count=3, statistics="all_positions")
                base = torch.arange(1, n+1, dtype=torch.float32)
                base = base / base.sum()
                blocks = torch.tensor([[0, 0, 0, 1, 1, 1], [0, 0, 0, 0, 0, 0]])
                if arm.endswith("root"):
                    blocks = blocks[-1:]
                artifact = {"base": base, "neighbor_ids": ids, "neighbor_weights": weights, "blocks": blocks}
                process = ChoiceDiffusion(artifact, config)
                (output / "prepared").mkdir()
                (output / "prepared/metadata.json").write_text(json.dumps({"item_count": n}))
                seed_all(100)
                model = make_denoiser(output, config, process)
                loss = process.loss(model, history, target)
                self.assertTrue(torch.isfinite(loss))
                loss.backward()
                params = list(model.parameters()) + list(process.parameters())
                gradients = [p.grad for p in params if p.grad is not None]
                self.assertTrue(gradients)
                self.assertTrue(all(torch.isfinite(g).all() for g in gradients))
                if arm.endswith("community"):
                    self.assertTrue(any(p.grad is not None for p in process.generator.rate_policy.gate.parameters()))
                else:
                    self.assertFalse(process.generator.rate_policy.is_learned)
                    context = process.generator.context(history)
                    rate = process.generator.rate_policy.initial_rates[0]
                    dense = rate * context.rho[:, None, :].expand(-1, n, -1)
                    dense = dense + config["diffusion"]["reset_rate"] * base[None, None, :]
                    dense = dense - (rate + config["diffusion"]["reset_rate"]) * torch.eye(n)[None]
                    for current in range(n):
                        x = torch.full((2,), current, dtype=torch.long)
                        expected = dense[:, :, current].clone()
                        expected[:, current] = 0.
                        torch.testing.assert_close(process.generator.incoming_rates(x, context), expected)
                torch.optim.AdamW(params, lr=.0006).step()
                EMA(model, .9999).update(model)


if __name__ == "__main__":
    unittest.main(verbosity=2)
