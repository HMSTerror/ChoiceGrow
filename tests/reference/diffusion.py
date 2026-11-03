"""History-fixed CTMC, exact nested-partition forward kernel, and reverse sampler."""
# Note: docs/decisions/implemented/architecture/2026-11-02-neighbor-real-main.md
from dataclasses import dataclass, replace
import math

import torch
from torch import nn
import torch.nn.functional as F


def sample_training_noise(batch_size, settings, device):
    """NewChoice losses.py sigma-derivative proposal, corrected to uniform-grid expectation."""
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


@dataclass
class Context:
    rho: torch.Tensor
    block_mass: tuple
    level_rates: torch.Tensor | None = None


class ChoiceProcess(nn.Module):
    def __init__(self, artifact, config):
        super().__init__()
        self.settings = config["diffusion"]
        self.destination = config["destination"]
        for name in ("base", "blocks", "neighbor_ids", "neighbor_weights"):
            self.register_buffer(name, artifact[name])
        self.level_count = len(self.blocks)
        if not self.level_count or len(self.settings["level_rates"]) != self.level_count:
            raise ValueError("Each partition level must have exactly one diffusion rate")
        self.item_count = len(self.base)
        alpha = self.settings["reset_rate"]
        scale = self.settings.get("rate_scale", 1.)
        self.rates = tuple(scale * (1 - alpha) * value for value in self.settings["level_rates"])
        self.reset_rate = scale * alpha
        # Column-source convention: bridge P_l G represents the paper's G P_l.
        # P_root G = P_root: L nested levels need 2L+1 coordinates (5 for two, 7 for three).
        # Note: docs/decisions/implemented/feature/2026-11-02-native-community-two-level.md
        root = self.level_count - 1
        self.basis = (("I", -1), *(('P', level) for level in range(self.level_count)),
                      ("G", -1), *(('B', level) for level in range(root)))

        def product(left, right):
            a, i = left
            b, j = right
            if a == "I":
                result = right
            elif b == "I":
                result = left
            elif a == "P" and b == "P":
                result = ("P", max(i, j))
            elif a == "G":
                result = ("G", -1)
            elif a == "P" and b == "G":
                result = ("B", i)
            elif a == "P" and b == "B":
                result = ("B", max(i, j))
            else:  # B_i P_j = B_i, B_i G = B_i, B_i B_j = B_i.
                result = left
            return ("P", root) if result == ("B", root) else result

        table = torch.tensor([[self.basis.index(product(a, b)) for b in self.basis]
                              for a in self.basis])
        structure = F.one_hot(table, len(self.basis)).permute(0, 2, 1).double()
        self.register_buffer("algebra_structure", structure, persistent=False)
        generator = torch.tensor([-sum(self.rates) - self.reset_rate, *self.rates,
                                  self.reset_rate, *([0.] * root)],
                                 dtype=torch.float64)
        self.register_buffer("algebra", torch.einsum("l,ldr->dr", generator, structure))

    def context(self, sequence):
        base = self.base.expand(len(sequence), -1)
        if self.destination["neighbor_mix"] == 0:
            # Preserve the exact shared-background ablation, including historical checkpoints.
            return Context(base, tuple(self.block_sum(base, level) for level in range(self.level_count)))
        valid = sequence < self.item_count
        safe = sequence.clamp_max(self.item_count - 1)
        age = sequence.shape[1] - 1 - torch.arange(sequence.shape[1], device=sequence.device)
        weights = valid * torch.exp(-self.destination["history_decay"] * age)
        ids = self.neighbor_ids[safe]
        neighbor_weights = self.neighbor_weights[safe] * weights[..., None]
        recent = torch.zeros_like(base).scatter_add(1, ids.flatten(1), neighbor_weights.flatten(1))
        recent = recent / weights.sum(-1, keepdim=True)
        mix = self.destination["neighbor_mix"]
        rho = (1 - mix) * base + mix * recent
        rho = rho / rho.sum(-1, keepdim=True)
        block_mass = tuple(self.block_sum(rho, level) for level in range(self.level_count))
        return Context(rho, block_mass)

    def block_sum(self, values, level):
        ids = self.blocks[level].expand(len(values), -1)
        totals = torch.zeros_like(values).scatter_add(1, ids, values)
        return totals.gather(1, ids)

    def projection(self, source, level, context):
        ids = self.blocks[level]
        mask = ids[None] == ids[source, None]
        denominator = context.block_mass[level].gather(1, source[:, None])
        return context.rho * mask / denominator

    def bridge(self, level, context):
        return context.rho * self.block_sum(self.base.expand_as(context.rho), level) / context.block_mass[level]

    def coefficients(self, sigma, context=None):
        unique, inverse = torch.unique(sigma.double().reshape(-1), return_inverse=True)
        values = torch.matrix_exp(unique[:, None, None] * self.algebra)[..., 0]
        return values[inverse].to(self.base.dtype)

    def forward_probability(self, source, sigma, context):
        weights = self.coefficients(sigma, context)
        result = weights[:, :1] * F.one_hot(source, self.item_count)
        for level in range(self.level_count):
            result = result + weights[:, level + 1:level + 2] * self.projection(source, level, context)
        reset = self.level_count + 1
        result = result + weights[:, reset:reset + 1] * self.base
        for level in range(self.level_count - 1):
            index = reset + 1 + level
            result = result + weights[:, index:index + 1] * self.bridge(level, context)
        if not torch.isfinite(result).all() or torch.any(result <= 0):
            raise FloatingPointError("Invalid forward probability")
        return result

    def terminal_reference(self, sigma, context):
        sigma = torch.full((len(context.rho),), float(sigma), device=self.base.device,
                           dtype=self.base.dtype)
        weights = self.coefficients(sigma, context)
        reset = self.level_count + 1
        result = (weights[:, :1] + weights[:, reset:reset + 1]) * self.base
        for level in range(self.level_count):
            coefficient = weights[:, level + 1:level + 2]
            if level < self.level_count - 1:
                index = reset + 1 + level
                coefficient = coefficient + weights[:, index:index + 1]
            result = result + coefficient * self.bridge(level, context)
        return result / result.sum(-1, keepdim=True)

    def incoming_rates(self, current, context):
        """Return Q_row[y, current], with the self edge removed."""
        rate = self.reset_rate * self.base[current, None].expand_as(context.rho)
        destination = context.rho.gather(1, current[:, None])
        for level, intensity in enumerate(self.rates):
            ids = self.blocks[level]
            members = ids[None] == ids[current, None]
            denominator = context.block_mass[level].gather(1, current[:, None])
            rate = rate + intensity * destination * members / denominator
        return rate.scatter(1, current[:, None], 0.0)

    def entropy(self, log_score, probability, current, context):
        reference = probability.log() - probability.gather(1, current[:, None]).log()
        delta = log_score - reference
        return (self.incoming_rates(current, context) * reference.exp()
                * (torch.expm1(delta) - delta)).sum(-1)

    def loss(self, model, sequence, target):
        settings = self.settings
        sigma, entropy_weight = sample_training_noise(len(target), settings, target.device)
        context = self.context(sequence)
        probability = self.forward_probability(target, sigma, context)
        current = torch.multinomial(probability, 1).squeeze(1)
        score = model(sequence, current, sigma)
        # The sole training objective: derivative-weighted denoising score entropy.
        values = entropy_weight * self.entropy(score, probability, current, context)
        return values.mean()

    @torch.no_grad()
    def sample_scores(self, model, sequence, guidance):
        settings = self.settings
        context = self.context(sequence)
        history = model.encode_history(sequence)
        initial = self.terminal_reference(settings["sigma_max"], context)
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
                incoming = self.incoming_rates(current, context)
                log_rates = incoming.log() + scores
                log_exit = torch.logsumexp(log_rates, -1)
                if not torch.isfinite(log_exit).all():
                    raise FloatingPointError("Nonfinite reverse exit rate")
                effective = log_exit.clamp_max(settings["log_exit_cap"])
                cap_count += int(((effective < log_exit) & active).sum())
                dt = torch.minimum(remaining, settings["reverse_cfl"] * torch.exp(-effective))
                if torch.any(active & (dt <= 0)):
                    raise FloatingPointError("Reverse step underflow")
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


class DynamicRateGate(nn.Module):
    """Port of E:/NewChoice/choice_graph.py::DynamicRateGate, unchanged features/init."""
    def __init__(self, base_rates, hidden_dim=16, temperature=1.0):
        super().__init__()
        if temperature <= 0.0 or not math.isfinite(float(temperature)):
            raise ValueError("dynamic rate temperature must be finite and positive")
        rates = torch.as_tensor(base_rates, dtype=torch.float32)
        if rates.ndim != 1 or torch.any(rates <= 0) or not torch.isfinite(rates).all():
            raise ValueError("dynamic rate gate requires positive finite base rates")
        self.total_rate = float(rates.sum().item())
        self.temperature = float(temperature)
        self.register_buffer("base_logits", torch.log(rates / rates.sum()))
        self.network = nn.Sequential(nn.Linear(3, int(hidden_dim)), nn.SiLU(),
                                     nn.Linear(int(hidden_dim), int(rates.numel())))
        nn.init.zeros_(self.network[-1].weight)
        nn.init.zeros_(self.network[-1].bias)

    def forward(self, rho):
        if rho.ndim != 2:
            raise ValueError("dynamic rate gate expects rho with shape [batch, items]")
        eps = torch.finfo(rho.dtype).eps
        safe_rho = rho.clamp_min(eps)
        entropy = -(safe_rho * safe_rho.log()).sum(dim=-1)
        entropy = entropy / math.log(max(2, rho.shape[-1]))
        topk = min(100, rho.shape[-1])
        top_mass = safe_rho.topk(topk, dim=-1).values.sum(dim=-1)
        concentration = 1.0 - entropy
        features = torch.stack((entropy, top_mass, concentration), dim=-1)
        logits = self.base_logits.to(rho.dtype) + self.network(features)
        weights = torch.softmax(logits / self.temperature, dim=-1)
        return self.total_rate * weights


class DynamicChoiceProcess(ChoiceProcess):
    # Note: docs/decisions/implemented/feature/2026-11-02-dynamic-community-rates.md
    def __init__(self, artifact, config):
        super().__init__(artifact, config)
        settings = config["diffusion"]
        if self.level_count != 2:
            raise ValueError("This experiment requires community/catalog two levels")
        # Keep gate initialization out of the denoiser's RNG stream.
        with torch.random.fork_rng(devices=[]):
            torch.random.default_generator.manual_seed(config["seed"] + 1741)
            self.rate_gate = DynamicRateGate(settings["level_rates"],
                settings.get("dynamic_rate_hidden_dim", 16), settings.get("dynamic_rate_temperature", 1.))

    def context(self, sequence):
        context = super().context(sequence)
        multiplier = self.settings.get("rate_scale", 1.) * (1 - self.settings["reset_rate"])
        return replace(context, level_rates=multiplier * self.rate_gate(context.rho))

    def coefficients(self, sigma, context=None):
        if context is None or context.level_rates is None:
            raise ValueError("Dynamic coefficients require history-fixed context rates")
        rates = context.level_rates.double()
        reset = torch.full_like(rates[:, :1], self.reset_rate)
        generator = torch.cat((-rates.sum(-1, keepdim=True) - reset, rates, reset,
                               torch.zeros_like(rates[:, :self.level_count - 1])), -1)
        algebra = torch.einsum("bl,ldr->bdr", generator, self.algebra_structure)
        return torch.matrix_exp(sigma.double().reshape(-1, 1, 1) * algebra)[..., 0].to(self.base.dtype)

    def incoming_rates(self, current, context):
        rate = self.reset_rate * self.base[current, None].expand_as(context.rho)
        destination = context.rho.gather(1, current[:, None])
        for level in range(self.level_count):
            ids = self.blocks[level]
            members = ids[None] == ids[current, None]
            denominator = context.block_mass[level].gather(1, current[:, None])
            rate = rate + context.level_rates[:, level:level + 1] * destination * members / denominator
        return rate.scatter(1, current[:, None], 0.0)
