"""Test file defaults, explicit CLI precedence, and invalid-share rejection to prevent incorrect experiments."""
from contextlib import redirect_stderr, redirect_stdout
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import torch

from configuration import load_config, resolve_config, validate_initial_shares
from transition import DynamicRateGate
import run


ROOT = Path(__file__).resolve().parents[1]


class ConfigurationTests(unittest.TestCase):
    def test_gate_starts_from_configured_shares_and_preserves_budget(self):
        """Verify the initial allocation and fixed budget across distinct destination distributions."""
        current = load_config(ROOT / "config.json")
        settings = current["diffusion"]
        gate = DynamicRateGate(settings["level_rates"])
        rho = torch.stack((torch.ones(12), torch.arange(1, 13, dtype=torch.float32)))
        rho = rho / rho.sum(-1, keepdim=True)
        rates = gate(rho)
        shares = rates / rates.sum(-1, keepdim=True)
        expected = torch.tensor(settings["initial_shares"]).expand_as(shares)
        torch.testing.assert_close(shares, expected, rtol=1e-6, atol=1e-7)
        torch.testing.assert_close(rates.sum(-1), torch.full((2,), settings["total_budget"]), rtol=1e-6, atol=1e-7)
        self.assertTrue(current["diffusion"]["dynamic_rate_gate"])

    def test_explicit_overrides_and_omission_are_distinct(self):
        config = load_config(ROOT / "config.json")
        config["diffusion"]["dynamic_rate_gate"] = False
        self.assertFalse(resolve_config(config)["diffusion"]["dynamic_rate_gate"])
        changed = resolve_config(config, initial_shares=[.3, .7], learnable_rates=True)
        budget = config["diffusion"]["total_budget"]
        self.assertEqual(changed["diffusion"]["level_rates"], [budget * .3, budget * .7])
        self.assertTrue(changed["diffusion"]["dynamic_rate_gate"])
        self.assertEqual(config["diffusion"]["initial_shares"], [.2165, .7835])
        self.assertFalse(resolve_config(changed, learnable_rates=False)["diffusion"]["dynamic_rate_gate"])

    def test_reject_invalid_weights_and_budget(self):
        for values in ([.2, .7], [.2, .9], [0., 1.], [-.1, 1.1], [float("nan"), .5],
                       [float("inf"), .5], [.5], [.2, .3, .5]):
            with self.subTest(values=values), self.assertRaises(ValueError):
                validate_initial_shares(values)
        for budget in (0., -1., float("nan"), float("inf")):
            config = load_config(ROOT / "config.json")
            config["diffusion"]["total_budget"] = budget
            with self.subTest(budget=budget), self.assertRaises(ValueError):
                resolve_config(config)

    def test_old_checkpoint_configuration_is_preserved(self):
        config = load_config(ROOT / "config.json")
        config["diffusion"].pop("initial_shares")
        config["diffusion"].pop("total_budget")
        config["diffusion"]["level_rates"] = [.5, 1.8]
        self.assertEqual(resolve_config(config), config)
        changed = resolve_config(config, initial_shares=[.4, .6])
        budget = sum(config["diffusion"]["level_rates"])
        self.assertEqual(changed["diffusion"]["level_rates"], [budget * .4, budget * .6])

    def test_cli_reads_file_defaults_and_applies_explicit_overrides(self):
        """Test precedence through the actual CLI parser, substituting only preprocessing to avoid rereading datasets."""
        with tempfile.TemporaryDirectory() as temp:
            config = load_config(ROOT / "config.json")
            config.update(dataset="Beauty", seed=102, device="cpu", smoke=True)
            config["diffusion"].update(initial_shares=[.3, .7], dynamic_rate_gate=False)
            config_path = Path(temp) / "config.json"
            config_path.write_text(json.dumps(config), encoding="utf-8")
            for suffix, extra, expected in (
                ("defaults", [], ("Beauty", 102, [.3, .7], False)),
                ("overrides", ["--dataset", "Sports", "--seed", "101", "--initial-shares", ".2", ".8",
                               "--learnable-rates"], ("Sports", 101, [.2, .8], True)),
            ):
                output = Path(temp) / suffix
                argv = ["run.py", "--config", str(config_path), "--output", str(output), "--stage", "prepare", *extra]
                with patch("sys.argv", argv), patch.object(run, "prepare") as prepare, redirect_stdout(io.StringIO()):
                    run.main()
                stored = json.loads((output / "resolved_config.json").read_text(encoding="utf-8"))
                self.assertEqual((stored["dataset"], stored["seed"], stored["diffusion"]["initial_shares"],
                                  stored["diffusion"]["dynamic_rate_gate"]), expected)
                self.assertEqual(prepare.call_args.args[-1], stored)

    def test_cli_bad_sum_fails_before_creating_output(self):
        with tempfile.TemporaryDirectory() as temp:
            output = Path(temp) / "must_not_start"
            argv = ["run.py", "--output", str(output), "--initial-shares", ".4", ".7"]
            error = io.StringIO()
            with patch("sys.argv", argv), redirect_stderr(error), self.assertRaises(SystemExit) as caught:
                run.main()
            self.assertEqual(caught.exception.code, 2)
            self.assertIn("must sum to 1", error.getvalue())
            self.assertFalse(output.exists())


if __name__ == "__main__":
    unittest.main()
