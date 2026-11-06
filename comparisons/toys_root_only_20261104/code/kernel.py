"""Exact forward kernel exp(sigma Q_u) using the closed algebra of nested partitions.

L levels need only 2L+1 basis elements; community/catalog nesting uses a 5-by-5 matrix exponential.
Public probabilities use row-source indexing; the internal multiplication table keeps legacy column-source indexing.
Its P_l G corresponds to G P_l in the paper's row-source notation; retain this orientation to avoid transposition errors.
Use the exact closed representation without constructing a dense catalog matrix or truncating an expansion.
"""
import torch
from torch import nn
import torch.nn.functional as F


class ExactForwardKernel(nn.Module):
    """Own one UserGenerator and expose forward probabilities and terminal initialization distributions."""

    def __init__(self, generator):
        """Build the basis multiplication table; cache fixed-rate algebra or assemble it per user for learned rates."""
        super().__init__()
        self.generator = generator
        self.level_count = generator.level_count
        root = self.level_count - 1
        self.basis = (("I", -1), *(('P', level) for level in range(self.level_count)),
                      ("G", -1), *(('B', level) for level in range(root)))

        def product(left, right):
            """Close nested projections under multiplication; select the coarser level and absorb root bridges into root projections."""
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
            else:
                result = left
            return ("P", root) if result == ("B", root) else result

        table = torch.tensor([[self.basis.index(product(a, b)) for b in self.basis]
                              for a in self.basis])
        structure = F.one_hot(table, len(self.basis)).permute(0, 2, 1).double()
        self.register_buffer("algebra_structure", structure, persistent=False)
        # Cache fixed-rate algebra; learned rates must retain the gradient path from gate to matrix exponential.
        if not generator.rate_policy.is_learned:
            self.register_buffer("algebra", torch.einsum("l,ldr->dr", generator.algebra_coordinates(), structure))

    def coefficients(self, sigma, context):
        """Map [B] noise scales to [B,2L+1] coefficients, computing the small matrix exponential in float64."""
        generator = self.generator
        if generator.rate_policy.is_learned:
            coordinates = generator.algebra_coordinates(context)
            algebra = torch.einsum("bl,ldr->bdr", coordinates, self.algebra_structure)
            return torch.matrix_exp(sigma.double().reshape(-1, 1, 1) * algebra)[..., 0].to(generator.base.dtype)
        unique, inverse = torch.unique(sigma.double().reshape(-1), return_inverse=True)
        values = torch.matrix_exp(unique[:, None, None] * self.algebra)[..., 0]
        return values[inverse].to(generator.base.dtype)

    def probability(self, source, sigma, context):
        """Return [B,N] forward probabilities q_sigma(j|source,u) for noisy-item sampling.

        Accumulate identity, projections, shared reset, and bridge terms in the original order.
        Reject nonfinite or nonpositive probabilities before logarithms or sampling.
        """
        generator = self.generator
        partition = generator.partition
        weights = self.coefficients(sigma, context)
        result = weights[:, :1] * F.one_hot(source, generator.item_count)
        for level in range(self.level_count):
            result = result + weights[:, level + 1:level + 2] * partition.projection(
                source, level, context.rho, context.block_mass)
        reset = self.level_count + 1
        result = result + weights[:, reset:reset + 1] * generator.base
        for level in range(self.level_count - 1):
            index = reset + 1 + level
            result = result + weights[:, index:index + 1] * partition.bridge(
                level, context.rho, context.block_mass, generator.base)
        if not torch.isfinite(result).all() or torch.any(result <= 0):
            raise FloatingPointError("Invalid forward probability")
        return result

    def terminal_reference(self, sigma, context):
        """Compute normalized [B,N] terminal references b exp(sigma Q_u) for reverse initialization."""
        generator = self.generator
        sigma = torch.full((len(context.rho),), float(sigma), device=generator.base.device,
                           dtype=generator.base.dtype)
        weights = self.coefficients(sigma, context)
        reset = self.level_count + 1
        result = (weights[:, :1] + weights[:, reset:reset + 1]) * generator.base
        for level in range(self.level_count):
            coefficient = weights[:, level + 1:level + 2]
            if level < self.level_count - 1:
                index = reset + 1 + level
                coefficient = coefficient + weights[:, index:index + 1]
            result = result + coefficient * generator.partition.bridge(
                level, context.rho, context.block_mass, generator.base)
        return result / result.sum(-1, keepdim=True)
