"""Old/new differential checks across matrices, training, sampling and checkpoints."""
from contextlib import redirect_stdout
from copy import deepcopy
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch

from artifacts import build_artifacts
from checkpoint import CheckpointStore
from configuration import load_config
from data import load_split, prepare, write_json
from diffusion import ChoiceDiffusion
from evaluate import Evaluator
from models import Denoiser
from train import Trainer
from transition import UserContext
from tests.reference_loader import load_reference

ROOT = Path(__file__).resolve().parents[1]
OLD = load_reference()


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


def pairs(config, device="cpu"):
    stored = artifact(len(config["diffusion"]["level_rates"]))
    old_class = OLD.diffusion.DynamicChoiceProcess if config["diffusion"]["dynamic_rate_gate"] else OLD.diffusion.ChoiceProcess
    return old_class(deepcopy(stored), config).to(device), ChoiceDiffusion(deepcopy(stored), config).to(device)


def prepare_fixture(directory, config):
    directory.mkdir(parents=True, exist_ok=True)
    prepared = directory / "prepared"
    prepared.mkdir()
    values = {"seq": np.array([[12, 12, 0, 1], [12, 2, 3, 4], [12, 12, 6, 7], [8, 9, 10, 11]]),
              "length": np.array([2, 3, 2, 4]), "target": np.array([2, 5, 8, 0])}
    for split in ("train", "val", "test"):
        np.savez(prepared / f"{split}.npz", **values)
    np.save(prepared / "train_target_counts.npy", np.bincount(values["target"], minlength=12))
    write_json(prepared / "metadata.json", {"item_count": 12, "dataset": "Toys", "history_length": 4})
    torch.save(artifact(), directory / "structure.pt")


class EquivalenceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def assert_exact(self, a, b):
        torch.testing.assert_close(a, b, rtol=0, atol=0)

    def compare_math(self, device):
        sequence = torch.tensor([[12, 12, 0, 1], [12, 2, 3, 4]], device=device)
        source = torch.tensor([2, 5], device=device)
        current = torch.tensor([1, 3], device=device)
        sigma = torch.tensor([.08, .15], device=device)
        for learned, levels in ((False, 1), (False, 2), (False, 3), (True, 2)):
            for neighbor_mix in (0., .02):
                with self.subTest(device=device, learned=learned, levels=levels, neighbor_mix=neighbor_mix):
                    config = settings(learned, levels)
                    config["destination"]["neighbor_mix"] = neighbor_mix
                    old, new = pairs(config, device)
                    if learned:
                        with torch.no_grad():
                            old.rate_gate.network[-1].bias.copy_(torch.tensor([.7, -.3], device=device))
                        new.generator.rate_policy.gate.load_state_dict(old.rate_gate.state_dict(), strict=True)
                    a, b = old.context(sequence), new.generator.context(sequence)
                    self.assert_exact(a.rho, b.rho)
                    for x, y in zip(a.block_mass, b.block_mass):
                        self.assert_exact(x, y)
                    if learned:
                        self.assert_exact(a.level_rates, b.level_rates)
                    self.assert_exact(old.coefficients(sigma, a), new.kernel.coefficients(sigma, b))
                    self.assert_exact(old.forward_probability(source, sigma, a), new.kernel.probability(source, sigma, b))
                    self.assert_exact(old.incoming_rates(current, a), new.generator.incoming_rates(current, b))
                    self.assert_exact(old.terminal_reference(.2, a), new.kernel.terminal_reference(.2, b))

    def test_probability_and_rates_exact_cpu(self):
        self.compare_math("cpu")

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA not available")
    def test_probability_and_rates_exact_cuda(self):
        self.compare_math("cuda:0")

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
                # Legacy-versus-current differential comparisons still require atol=rtol=0.
                torch.testing.assert_close(gen.incoming_rates(torch.arange(12), ctx), incoming,
                                           atol=2e-12, rtol=2e-11)

    def compare_training_step(self, device):
        config = settings()
        sequence = torch.tensor([[12, 12, 0, 1], [12, 2, 3, 4]], device=device)
        target = torch.tensor([2, 5], device=device)
        old, new = pairs(config, device)
        torch.manual_seed(701)
        a = OLD.models.Denoiser(12, config).to(device)
        torch.manual_seed(701)
        b = Denoiser(12, config).to(device)
        for key in a.state_dict():
            self.assert_exact(a.state_dict()[key], b.state_dict()[key])
        pa = list(a.parameters()) + list(old.rate_gate.parameters())
        pb = list(b.parameters()) + list(new.generator.rate_policy.gate.parameters())
        oa = torch.optim.AdamW(pa, lr=config["training"]["learning_rate"], weight_decay=0.)
        ob = torch.optim.AdamW(pb, lr=config["training"]["learning_rate"], weight_decay=0.)
        torch.manual_seed(901); la = old.loss(a, sequence, target); la.backward()
        torch.manual_seed(901); lb = new.loss(b, sequence, target); lb.backward()
        self.assert_exact(la, lb)
        for x, y in zip(pa, pb):
            self.assert_exact(x.grad, y.grad)
        torch.nn.utils.clip_grad_norm_(pa, 1.); torch.nn.utils.clip_grad_norm_(pb, 1.)
        oa.step(); ob.step()
        for x, y in zip(pa, pb):
            self.assert_exact(x, y)
        a.eval(); b.eval()
        torch.manual_seed(902); sa, ta = old.sample_scores(a, sequence, 2.)
        torch.manual_seed(902); sb, tb = new.sample_scores(b, sequence, 2.)
        self.assert_exact(sa, sb)
        self.assertEqual(ta, tb)

    def test_loss_gradient_adamw_and_reverse_sampler_exact_cpu(self):
        self.compare_training_step("cpu")

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA not available")
    def test_loss_gradient_adamw_and_reverse_sampler_exact_cuda(self):
        self.compare_training_step("cuda:0")

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

    def compare_trainers(self, config, fake_validation=False):
        with tempfile.TemporaryDirectory() as temp:
            left, right = Path(temp) / "old", Path(temp) / "new"
            prepare_fixture(left, config); prepare_fixture(right, config)
            validation = [{"ndcg@10": x} for x in (.1, .2, .2, .2)]
            with redirect_stdout(io.StringIO()):
                if fake_validation:
                    with patch.object(OLD.train, "evaluate_diffusion", side_effect=deepcopy(validation)):
                        OLD.train.train_diffusion(left, config, torch.device("cpu"))
                    with patch.object(Evaluator, "evaluate", side_effect=deepcopy(validation)):
                        Trainer(right, config, torch.device("cpu")).fit()
                else:
                    OLD.train.train_diffusion(left, config, torch.device("cpu"))
                    Trainer(right, config, torch.device("cpu")).fit()
            self.assertEqual((left / "choice_log.jsonl").read_text(), (right / "choice_log.jsonl").read_text())
            a = json.loads((left / "training.json").read_text())
            b = json.loads((right / "training.json").read_text())
            for key in ("elapsed_seconds", "selected_checkpoint_seconds"):
                a.pop(key); b.pop(key)
            self.assertEqual(a, b)
            if fake_validation:
                self.assertEqual((b["best_epoch"], b["epochs_completed"], b["stop_reason"]), (4, 8, "early_stopping"))
            for filename in ("choice_best.pt", "choice_newst.pt"):
                x, y = (torch.load(p / filename, weights_only=True) for p in (left, right))
                for field in ("model", "rate_gate"):
                    self.assertEqual(list(x[field]), list(y[field]))
                    for key in x[field]:self.assert_exact(x[field][key], y[field][key])
                self.assertEqual(x["epoch"], y["epoch"])
            # Both readers consume the other implementation's saved checkpoint.
            old_process = OLD.train.load_process(left, config, torch.device("cpu"))
            new_process = ChoiceDiffusion.from_directory(right, config, torch.device("cpu"))
            old_state = CheckpointStore(left, config).load_best(new_process, torch.device("cpu"))
            new_state = torch.load(right / "choice_best.pt", weights_only=True)
            OLD.train.restore_process(old_process, new_state)
            self.assert_exact(old_process.rate_gate.network[-1].bias, new_process.generator.rate_policy.gate.network[-1].bias)
            if not fake_validation:
                with redirect_stdout(io.StringIO()):
                    x = OLD.train.final_evaluation(left, config, torch.device("cpu"))
                    y = Evaluator(config, torch.device("cpu")).test_best(right)
                self.assertEqual(x, y)
                with np.load(left / "recommendations.npz") as x, np.load(right / "recommendations.npz") as y:
                    np.testing.assert_array_equal(x["top10"], y["top10"])

    def test_training_logs_checkpoints_and_three_stream_evaluation_exact(self):
        self.compare_trainers(settings())

    def test_uncapped_early_stopping_matches_original(self):
        config = settings(); config["training"].update(epochs=0, patience=4)
        self.compare_trainers(config, fake_validation=True)

    def test_raw_preparation_and_training_only_artifacts_match(self):
        config = settings(); config.update(history_length=10, smoke=True)
        with tempfile.TemporaryDirectory() as temp:
            old, new = Path(temp) / "old", Path(temp) / "new"
            # Compare both loaders on the same DF from the new, traceable dataset directory.
            # Adapt only legacy directory naming, leaving the frozen reference code unchanged.
            with patch.dict(OLD.data.DATASETS, {"Toys": "Toys"}):
                OLD.data.prepare(ROOT / "dataset/processed", old, "Toys", config)
            prepare(ROOT / "dataset/processed", new, "Toys", config)
            for split in ("train", "val", "test"):
                for key, value in load_split(old, split).items():
                    self.assert_exact(value, load_split(new, split)[key])
            OLD.train.build_artifacts(old, config, torch.device("cpu"))
            build_artifacts(new, config, torch.device("cpu"))
            a, b = (torch.load(p / "structure.pt", weights_only=True) for p in (old, new))
            for key in a:self.assert_exact(a[key], b[key])


if __name__ == "__main__":
    unittest.main()
