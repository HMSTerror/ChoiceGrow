"""Check three-level exact probabilities, gate gradients and preservation of reused two-level controls."""
import unittest

import run_suite as suite
from configuration import resolve_config
from diffusion import ChoiceDiffusion
from evaluate import Evaluator

torch = suite.p.torch


class RandomLevelTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def process(self):
        n = 7
        base = torch.arange(1, n + 1, dtype=torch.float32)
        base = base / base.sum()
        artifact = {"base": base, "neighbor_ids": torch.arange(n)[:, None],
                    "neighbor_weights": torch.ones(n, 1),
                    "blocks": suite.random_partition(n, 2, 4, 100)}
        config = resolve_config(suite.make_config(100, "random_three"))
        return ChoiceDiffusion(artifact, config), config

    def test_real_catalog_partition_is_nested_seeded_and_has_expected_sizes(self):
        blocks = suite.random_partition(11924)
        self.assertEqual([len(level.unique()) for level in blocks], [187, 12, 1])
        torch.testing.assert_close(blocks[0] // 16, blocks[1], rtol=0, atol=0)
        torch.testing.assert_close(blocks, suite.random_partition(11924), rtol=0, atol=0)
        self.assertFalse(torch.equal(blocks, suite.random_partition(11924, seed=101)))
        self.assertLessEqual(int(torch.bincount(blocks[0]).max()), 64)
        self.assertLessEqual(int(torch.bincount(blocks[1]).max()), 1024)

    def test_initial_shares_preserve_nonroot_budget_and_reject_invalid_sums(self):
        native = resolve_config(suite.make_config(100, "native_two"))
        random = resolve_config(suite.make_config(100, "random_three"))
        self.assertAlmostEqual(sum(suite.INITIAL_SHARES[:2]), native["diffusion"]["initial_shares"][0])
        self.assertAlmostEqual(suite.INITIAL_SHARES[0] / sum(suite.INITIAL_SHARES[:2]), .887)
        self.assertAlmostEqual(sum(random["diffusion"]["level_rates"]), random["diffusion"]["total_budget"])
        random["diffusion"]["initial_shares"] = [.2, .2, .7]
        with self.assertRaisesRegex(ValueError, "sum to 1"):
            resolve_config(random)

    def test_two_level_control_configuration_matches_completed_source(self):
        for seed in suite.SEEDS:
            existing = suite.p.read_json(suite.SOURCE / f"seed_{seed}/resolved_config.json")
            self.assertEqual(resolve_config(suite.make_config(seed, "native_two")), existing)

    def test_three_level_exact_kernel_matches_dense_matrix_exponential(self):
        process, _ = self.process()
        sequence = torch.tensor([[0, 1], [2, 3]])
        source = torch.tensor([4, 6])
        sigma = torch.tensor([.2, .7])
        context = process.generator.context(sequence)
        probabilities = process.kernel.probability(source, sigma, context)
        generator = process.generator
        identity = torch.eye(process.item_count, dtype=torch.float64)
        expected = []
        for batch in range(len(source)):
            rho = context.rho[batch].double()
            q = generator.reset_rate * (generator.base.double().expand(process.item_count, -1) - identity)
            for level in range(3):
                blocks = generator.partition.blocks[level]
                mask = blocks[:, None] == blocks[None, :]
                projection = rho[None, :] * mask / context.block_mass[level][batch].double()[:, None]
                q = q + context.level_rates[batch, level].double() * (projection - identity)
            expected.append(torch.matrix_exp(sigma[batch].double() * q)[source[batch]])
        torch.testing.assert_close(probabilities.double(), torch.stack(expected), rtol=2e-6, atol=2e-7)
        torch.testing.assert_close(probabilities.sum(-1), torch.ones(2), rtol=2e-6, atol=2e-7)

    def test_three_level_probabilities_retain_gradient_to_rate_gate(self):
        process, _ = self.process()
        sequence = torch.tensor([[0, 1], [2, 3]])
        context = process.generator.context(sequence)
        probabilities = process.kernel.probability(torch.tensor([4, 6]), torch.tensor([.2, .7]), context)
        weighted = (probabilities * torch.arange(process.item_count)).sum()
        weighted.backward()
        gradient = process.generator.rate_policy.gate.network[-1].weight.grad
        self.assertIsNotNone(gradient)
        self.assertTrue(torch.isfinite(gradient).all())
        self.assertGreater(float(gradient.abs().max()), 1e-8)

    def test_three_level_reporting_labels_fine_coarse_and_catalog(self):
        process, config = self.process()
        values = {"seq": torch.tensor([[0, 1], [2, 3]]), "length": torch.tensor([2, 2]),
                  "target": torch.tensor([4, 6])}
        stats = Evaluator(config, torch.device("cpu")).rate_statistics(process, values)
        for label, share in zip(("fine", "coarse", "catalog"), suite.INITIAL_SHARES):
            self.assertAlmostEqual(stats[label]["mean"], share, places=6)
        self.assertAlmostEqual(stats["root_mean"], stats["catalog"]["mean"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
