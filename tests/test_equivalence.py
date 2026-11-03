"""Check the structured kernel against a dense exponential and verify parameter ownership."""
from pathlib import Path
import unittest

import torch

from configuration import load_config
from diffusion import ChoiceDiffusion
from transition import UserContext

ROOT = Path(__file__).resolve().parents[1]


def settings(learned=True, levels=2):
    config = load_config(ROOT / "config.json")
    config.update(dataset="Toys", smoke=False, hidden_size=8, history_length=4)
    config["diffusion"].update(dynamic_rate_gate=learned, sigma_max=.2, reverse_intervals=2)
    if levels != 2:
        config["diffusion"]["level_rates"] = [2.302585092994046] if levels == 1 else [.4, .1, 1.802585092994046]
    config["training"].update(heads=2, batch_size=2, eval_batch_size=2,
        epochs=4, validation_interval=2, dropout=.1)
    return config


def artifact(levels=2):
    base = torch.arange(1, 13, dtype=torch.float32)
    base = base / base.sum()
    ids = torch.arange(12)
    community, coarse, root = ids // 2, ids // 4, torch.zeros(12, dtype=torch.long)
    blocks = torch.stack({1: [root], 2: [community, root], 3: [community, coarse, root]}[levels])
    return {"base": base, "blocks": blocks, "neighbor_ids": torch.stack((ids, (ids + 1) % 12), -1),
            "neighbor_weights": torch.tensor([.7, .3]).expand(12, -1).clone()}


class EquivalenceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def assert_exact(self, a, b):
        torch.testing.assert_close(a, b, rtol=0, atol=0)

    def test_dense_exponential_and_generator_conservation(self):
        for learned, levels in ((False, 1), (False, 2), (False, 3), (True, 2)):
            with self.subTest(learned=learned, levels=levels):
                stored = artifact(levels)
                stored["base"] = torch.arange(1, 13, dtype=torch.float64)
                stored["base"] /= stored["base"].sum()
                process = ChoiceDiffusion(stored, settings(learned, levels)).double()
                gen = process.generator
                one = gen.context(torch.tensor([[12, 12, 0, 1]]))
                rates = one.level_rates.expand(12, -1) if learned else one.level_rates
                ctx = UserContext(one.rho.expand(12, -1), tuple(x.expand(12, -1) for x in one.block_mass), rates)
                eye = torch.eye(12, dtype=torch.float64)
                q = gen.reset_rate * (gen.base[None] - eye)
                for level in range(levels):
                    rate = one.level_rates[0, level] if learned else one.level_rates[level]
                    projection = gen.partition.projection(torch.arange(12), level, ctx.rho, ctx.block_mass)
                    q = q + rate * (projection - eye)
                torch.testing.assert_close(q.sum(-1), torch.zeros(12, dtype=torch.float64), atol=2e-12, rtol=0)
                for sigma in (.001, .2, 10.):
                    expected = torch.matrix_exp(sigma * q)
                    actual = process.kernel.probability(torch.arange(12), torch.full((12,), sigma, dtype=torch.float64), ctx)
                    torch.testing.assert_close(actual, expected, atol=2e-12, rtol=2e-11)
                incoming = q.T.clone(); incoming.fill_diagonal_(0.)
                # Dense and structured formulas use different arithmetic orders; allow float64 rounding error here.
                torch.testing.assert_close(gen.incoming_rates(torch.arange(12), ctx), incoming,
                                           atol=2e-12, rtol=2e-11)

    def test_gate_keeps_rng_and_parameter_ownership(self):
        torch.manual_seed(81); before = torch.random.get_rng_state()
        process = ChoiceDiffusion(artifact(), settings())
        self.assert_exact(before, torch.random.get_rng_state())
        gate = process.generator.rate_policy.gate
        self.assertEqual(len(list(process.parameters())), len(list(gate.parameters())))
        self.assertEqual(len(list(process.named_buffers())), len(list(process.named_buffers(remove_duplicate=False))))
        process.double()
        context = process.generator.context(torch.tensor([[12, 12, 0, 1]]))
        self.assertEqual(context.rho.dtype, torch.float64)
        self.assertEqual(context.level_rates.dtype, torch.float64)


if __name__ == "__main__":
    unittest.main()
