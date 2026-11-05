"""User-conditioned generator Q_u and the paper's matrix assembly formula.

Q_u = sum_l r_{u,l} (P_{u,l} - I) + reset_rate * (G - I)
r_{u,l} = (1-alpha) * lambda_{u,l}; G[i,j] = base[j].

Public Q_u[i,j] denotes the rate from item i to j; every row sums to zero.
rho_u defines within-block destinations, lambda defines level budgets, and G resets to shared background.
The main method learns rates: level_rates derived from initial_shares and total_budget
define the gate's initialization and fixed total budget.
After training, user-distribution features determine the shares; disabling learning gives fixed-rate ablations.
This module assembles rates; kernel.py computes transition probabilities exp(sigma * Q_u).
Represent the catalog matrix implicitly, materializing only requested incoming rates and small algebra coefficients.
"""
# Note: docs/decisions/implemented/architecture/2026-11-03-paper-modules.md
# Note: docs/decisions/implemented/feature/2026-11-04-fixed-shares.md
from abc import ABC, abstractmethod
from dataclasses import dataclass
import math

import torch
from torch import nn

from neighbors import NeighborDestination
from scope import NestedPartition


@dataclass
class UserContext:
    """History-specific context for a batch, reused throughout each diffusion path.

    rho: [B,N] destination probabilities over N items for each user.
    block_mass: L tensors [B,N], with each position holding the probability mass of its containing block.
    level_rates: [B,L] for learned rates, or L Python floats when learning is disabled.
    Fixed mode retains scalar floats to preserve legacy multiplication order and rounding.
    """
    rho: torch.Tensor
    block_mass: tuple
    level_rates: tuple | torch.Tensor


class RatePolicy(nn.Module, ABC):
    """Rate-policy interface supplying effective level rates to one shared Q_u constructor.

    Encapsulate the common (1-alpha) scaling; subclasses decide whether lambda changes during learning.
    LearnedRates is the main method; FixedRates supports the disabled-learning ablation.
    """
    is_learned = False

    def __init__(self, settings):
        """Store initial budgets and effective scaling; UserGenerator handles the separate reset term alpha."""
        super().__init__()
        alpha = settings["reset_rate"]
        self.multiplier = 1 - alpha
        self.initial_rates = tuple(self.multiplier * value for value in settings["level_rates"])

    @abstractmethod
    def forward(self, rho):
        """Map destination probabilities [B,N] to fixed scalar rates or learned effective rates [B,L]."""


class FixedRates(RatePolicy):
    """Disabled-learning ablation using the configured rates for every user and epoch."""
    def forward(self, rho):
        """Return the original scalar tuple; rho is accepted only to share the policy interface."""
        return self.initial_rates


class FixedShares(FixedRates):
    """Use explicit constant shares of Lambda, with no gate or trainable parameters.

    fixed_shares are ordered like the partition levels. A single [1] is Root;
    [s_C, 1-s_C] is community/catalog. Zero shares are valid structural controls.
    Validation and early stopping do not depend on this policy implementation.
    """
    def __init__(self, settings):
        shares = settings["fixed_shares"]
        if not isinstance(shares, (list, tuple)) or not shares:
            raise ValueError("fixed_shares must be a nonempty list of level shares")
        shares = tuple(float(value) for value in shares)
        if any(not math.isfinite(value) or value < 0 or value > 1 for value in shares):
            raise ValueError("fixed_shares must be finite and between zero and one")
        if not math.isclose(sum(shares), 1., rel_tol=0., abs_tol=1e-9):
            raise ValueError("fixed_shares must sum to 1; values are not silently normalized")
        budget = float(settings["total_budget"])
        if not math.isfinite(budget) or budget <= 0:
            raise ValueError("total_budget must be finite and positive")
        self.shares, self.total_budget = shares, budget
        super().__init__({**settings, "level_rates": [budget * share for share in shares]})


class DynamicRateGate(nn.Module):
    """Original three-feature gate: normalized entropy, Top100 mass, and concentration.

    Map rho [B,N] to raw level budgets lambda [B,L].
    Softmax normalizes shares to one; multiplying by the initial total Lambda preserves the total budget.
    Keep parameter names and initialization order compatible with existing rate_gate weights.
    """
    def __init__(self, base_rates, hidden_dim=16, temperature=1.0):
        """Store base_logits and zero the final layer so temperature 1 reproduces the configured initial shares."""
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
        """Summarize each destination distribution and adjust initial log shares through a 3-to-16-to-L network."""
        if rho.ndim != 2:
            raise ValueError("dynamic rate gate expects rho with shape [batch, items]")
        eps = torch.finfo(rho.dtype).eps
        safe_rho = rho.clamp_min(eps)  # Avoid log(0) using the original dtype-dependent lower bound.
        entropy = -(safe_rho * safe_rho.log()).sum(dim=-1)
        entropy = entropy / math.log(max(2, rho.shape[-1]))
        topk = min(100, rho.shape[-1])
        top_mass = safe_rho.topk(topk, dim=-1).values.sum(dim=-1)
        concentration = 1.0 - entropy
        features = torch.stack((entropy, top_mass, concentration), dim=-1)
        # Initial shares are the logit baseline; the network learns adjustments rather than permanently fixing them.
        logits = self.base_logits.to(rho.dtype) + self.network(features)
        weights = torch.softmax(logits / self.temperature, dim=-1)
        return self.total_rate * weights


class LearnedRates(RatePolicy):
    """Main rate policy: initialize from configured shares and optimize the gate jointly with the denoiser."""
    is_learned = True

    def __init__(self, settings, seed):
        """Create the two-level gate with isolated RNG initialization so denoiser initialization remains unchanged."""
        super().__init__(settings)
        if len(settings["level_rates"]) != 2:
            raise ValueError("This experiment requires community/catalog two levels")
        # Restore the preceding CPU RNG state; seed+1741 follows the original implementation.
        with torch.random.fork_rng(devices=[]):
            torch.random.default_generator.manual_seed(seed + 1741)
            self.gate = DynamicRateGate(settings["level_rates"],
                settings.get("dynamic_rate_hidden_dim", 16), settings.get("dynamic_rate_temperature", 1.))

    def forward(self, rho):
        """Scale the gate's raw lambda by (1-alpha) to obtain the effective rates in Q_u."""
        return self.multiplier * self.gate(rho)


class UserGenerator(nn.Module):
    """Combine destinations, nested scopes, and a rate policy into Q_u for a given history.

    Own all three submodules so .to(device) and parameter management recurse consistently.
    incoming_rates(x)[y] returns Q_u[y,x] with a zero diagonal.
    Reverse diffusion multiplies these incoming rates by the denoiser's predicted score ratios.
    """

    def __init__(self, destination, partition, rate_policy, settings):
        """Inject the three mathematical modules, check level counts, and store the separate shared-reset rate."""
        super().__init__()
        self.destination = destination
        self.partition = partition
        self.rate_policy = rate_policy
        self.level_count = partition.level_count
        if not self.level_count or len(rate_policy.initial_rates) != self.level_count:
            raise ValueError("Each partition level must have exactly one diffusion rate")
        self.item_count = destination.item_count
        self.reset_rate = settings["reset_rate"]

    @classmethod
    def from_artifact(cls, artifact, config):
        """Construct from structure.pt; select the rate-policy implementation only at this boundary."""
        settings = config["diffusion"]
        if "fixed_shares" in settings:
            if settings.get("dynamic_rate_gate", False):
                raise ValueError("fixed_shares requires dynamic_rate_gate=false")
            policy = FixedShares(settings)
        else:
            policy = (LearnedRates(settings, config["seed"]) if settings.get("dynamic_rate_gate", False)
                      else FixedRates(settings))
        return cls(NeighborDestination(artifact, config["destination"]),
                   NestedPartition(artifact["blocks"]), policy, settings)

    @property
    def base(self):
        """Expose shared background b through its existing buffer without duplicate registration or copying."""
        return self.destination.base

    def context(self, sequence):
        """Build one path's context: history [B,H] -> rho_u -> block masses -> effective rates."""
        rho = self.destination(sequence)
        return UserContext(rho, self.partition.masses(rho), self.rate_policy(rho))

    def algebra_coordinates(self, context=None):
        """Express Q_u as coefficients on the basis [I, P_l, G, bridge_l].

        The identity coefficient is -sum(r_l)-reset, ensuring all generator rows sum to zero.
        Bridge coefficients are zero in the generator but appear in its matrix exponential.
        Return [B,2L+1] for learned rates or [2L+1] for fixed rates; exponentiation uses float64.
        """
        root = self.level_count - 1
        if self.rate_policy.is_learned:
            if context is None or not isinstance(context.level_rates, torch.Tensor):
                raise ValueError("Dynamic coefficients require history-fixed context rates")
            rates = context.level_rates.double()
            reset = torch.full_like(rates[:, :1], self.reset_rate)
            return torch.cat((-rates.sum(-1, keepdim=True) - reset, rates, reset,
                              torch.zeros_like(rates[:, :root])), -1)
        rates = self.rate_policy.initial_rates
        return torch.tensor([-sum(rates) - self.reset_rate, *rates, self.reset_rate,
                             *([0.] * root)], dtype=torch.float64, device=self.base.device)

    def incoming_rates(self, current, context):
        """Return incoming rates Q_u[:,current] as [B,N] for loss computation and reverse sampling.

        Accumulate shared reset, then within-block incoming rates, and finally remove self-transitions.
        Gather destination rho and block mass directly instead of constructing a [B,N,N] matrix.
        """
        rate = self.reset_rate * self.base[current, None].expand_as(context.rho)
        destination = context.rho.gather(1, current[:, None])
        for level in range(self.level_count):
            intensity = (context.level_rates[:, level:level + 1] if self.rate_policy.is_learned
                         else context.level_rates[level])
            ids = self.partition.blocks[level]
            members = ids[None] == ids[current, None]
            denominator = context.block_mass[level].gather(1, current[:, None])
            rate = rate + intensity * destination * members / denominator
        return rate.scatter(1, current[:, None], 0.0)
