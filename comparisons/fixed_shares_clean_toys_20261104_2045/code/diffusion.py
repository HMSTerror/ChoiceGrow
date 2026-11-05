"""Diffusion training and reverse recommendation using Q_u, the exact forward kernel, and ChoiceGrow scores.

Training: sample noise, compute exact forward probabilities, corrupt the target, and optimize score entropy.
Inference: initialize from the terminal reference, reverse-jump with score-adjusted rates, and return ranking scores.
"""
import math
from pathlib import Path

import torch
from torch import nn

from kernel import ExactForwardKernel
from transition import UserGenerator


def sample_training_noise(batch_size, settings, device):
    """Sample [B] noise scales and loss weights with the original importance-sampling scheme.

    The grid contains training_intervals+1 points; sigma_importance samples using the noise derivative.
    Inverse-probability correction preserves the expectation of the original uniformly discretized objective.
    """
    grid = torch.linspace(1., .001, settings["training_intervals"] + 1, device=device)
    log_range = math.log(settings["sigma_max"] / settings["sigma_min"])
    sigma = settings["sigma_min"] * torch.exp(grid * log_range)
    derivative = sigma * log_range
    mode = settings.get("time_sampling", "uniform")
    if mode == "uniform":
        indices = torch.randint(len(grid), (batch_size,), device=device)
        correction = 1.
    elif mode == "sigma_importance":
        probability = .1 / len(grid) + .9 * derivative.detach() / derivative.detach().sum()
        indices = torch.multinomial(probability, batch_size, replacement=True)
        correction = (1. / (len(grid) * probability[indices])).detach()
    else:
        raise ValueError(f"Unsupported time_sampling: {mode}")
    return sigma[indices], derivative[indices] * correction

class ReverseSampler:
    """Adaptive reverse jumps; the generator supplies incoming rates separately from neural score prediction."""
    def __init__(self, settings):
        """Store the reverse intervals, CFL constraint, exit-rate cap, and substep limit."""
        self.settings = settings

    @torch.no_grad()
    def sample(self, model, sequence, guidance, kernel):
        """Map histories [B,H] to catalog scores [B,N] and substep statistics, without gradients.

        Construct each history's context once. Encode history once, then predict score ratios from the
        current item and noise scale at each substep, preserving the original random-number call order.
        """
        settings = self.settings
        context = kernel.generator.context(sequence)
        history = model.encode_history(sequence)
        initial = kernel.terminal_reference(settings["sigma_max"], context)
        current = torch.multinomial(initial, 1).squeeze(1)
        grid = torch.linspace(settings["sigma_max"], settings["sigma_min"],
                              settings["reverse_intervals"] + 1, device=sequence.device)
        step_count = torch.zeros(len(sequence), device=sequence.device)
        cap_count = 0
        for high, low in zip(grid[:-1], grid[1:]):
            remaining = torch.full((len(sequence),), float(high - low), device=sequence.device)
            for _ in range(settings["max_substeps"]):
                active = remaining > 1e-7
                if not active.any():
                    break
                sigma = low + remaining
                scores = model.guided_scores(history, current, sigma, guidance)
                incoming = kernel.generator.incoming_rates(current, context)
                # Reverse rate = Q_u[y,x] times the predicted probability ratio; combine in log space for stability.
                log_rates = incoming.log() + scores
                log_exit = torch.logsumexp(log_rates, -1)
                if not torch.isfinite(log_exit).all():
                    raise FloatingPointError("Nonfinite reverse exit rate")
                # Cap extreme exit rates and constrain the substep with CFL to bound the single-step event probability.
                effective = log_exit.clamp_max(settings["log_exit_cap"])
                cap_count += int(((effective < log_exit) & active).sum())
                dt = torch.minimum(remaining, settings["reverse_cfl"] * torch.exp(-effective))
                if torch.any(active & (dt <= 0)):
                    raise FloatingPointError("Reverse step underflow")
                # Sample whether to jump and the destination separately; inactive rows retain their current state.
                event = (torch.rand_like(remaining) < dt * effective.exp()) & active
                destination = torch.multinomial((log_rates - log_exit[:, None]).exp(), 1).squeeze(1)
                current = torch.where(event, destination, current)
                remaining = (remaining - dt).clamp_min(0)
                step_count += active
            if torch.any(remaining > 1e-7):
                raise RuntimeError("Reverse substep budget exceeded")
        endpoint = torch.full((len(sequence),), settings["sigma_min"], device=sequence.device)
        return model.guided_scores(history, current, endpoint, guidance), {
            "mean_reverse_substeps": float(step_count.mean()), "exit_cap_activations": cap_count}


class ChoiceDiffusion(nn.Module):
    """Training-facing interface for loss computation and recommendation score generation.

    Ownership follows diffusion -> exact kernel -> Q_u -> destination/partition/rate policy.
    Register the gate only once to avoid duplicate optimization or inconsistent device transfers.
    """
    def __init__(self, artifact, config):
        """Compose the mathematical modules; only the rate gate is trainable here, while Trainer owns the denoiser."""
        super().__init__()
        self.settings = config["diffusion"]
        self.kernel = ExactForwardKernel(UserGenerator.from_artifact(artifact, config))
        self.sampler = ReverseSampler(self.settings)
        self.item_count = self.generator.item_count

    @classmethod
    def from_directory(cls, output, config, device):
        """Load structure.pt and move the complete module tree to the requested device."""
        artifact = torch.load(Path(output) / "structure.pt", map_location="cpu", weights_only=True)
        return cls(artifact, config).to(device)

    @property
    def generator(self):
        """Access the same Q_u object owned by the exact kernel."""
        return self.kernel.generator

    def entropy(self, log_score, probability, current, context):
        """Compute per-sample score entropy with gradient paths through probabilities, incoming rates, and the gate.

        reference is the exact forward log-probability ratio; delta is the prediction error.
        expm1(delta)-delta is stable near zero, weighted by the reference ratio and incoming rates.
        """
        reference = probability.log() - probability.gather(1, current[:, None]).log()
        delta = log_score - reference
        return (self.generator.incoming_rates(current, context) * reference.exp()
                * (torch.expm1(delta) - delta)).sum(-1)

    def loss(self, model, sequence, target):
        """Return a scalar loss; discrete noisy-item sampling is not differentiated, while other paths retain gradients."""
        sigma, entropy_weight = sample_training_noise(len(target), self.settings, target.device)
        context = self.generator.context(sequence)
        probability = self.kernel.probability(target, sigma, context)
        current = torch.multinomial(probability, 1).squeeze(1)
        score = model(sequence, current, sigma)
        values = entropy_weight * self.entropy(score, probability, current, context)
        return values.mean()

    def sample_scores(self, model, sequence, guidance):
        """Delegate inference with the current denoiser and exact kernel to the reverse sampler."""
        return self.sampler.sample(model, sequence, guidance, self.kernel)
