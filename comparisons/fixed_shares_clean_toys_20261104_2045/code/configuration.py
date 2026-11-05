"""Resolve configuration: file defaults, explicit CLI overrides, then the effective run settings.

Configure initial_shares (community, catalog) and total_budget (Lambda).
Internally, level_rates = Lambda * initial_shares preserves the mathematical and checkpoint interfaces.
Legacy level_rates-only configurations remain valid without recomputing their floating-point values.
"""
from copy import deepcopy
import json
import math
from pathlib import Path


def validate_initial_shares(values):
    """Require two finite, strictly positive shares summing to one, with an absolute tolerance of 1e-9.

    Do not silently normalize invalid inputs, which would run a different experiment than requested.
    Both levels require positive shares; removing a level is a structural ablation, not a zero share.
    """
    if not isinstance(values, (list, tuple)) or len(values) != 2:
        raise ValueError("initial_shares must contain two values, ordered as community and catalog")
    shares = [float(value) for value in values]
    if any(not math.isfinite(value) or value <= 0 or value >= 1 for value in shares):
        raise ValueError("Community and catalog initial shares must be finite and strictly between 0 and 1")
    if not math.isclose(sum(shares), 1., rel_tol=0., abs_tol=1e-9):
        raise ValueError(f"Community and catalog initial shares must sum to 1; got {sum(shares):.12g}")
    return shares


def resolve_config(config, *, initial_shares=None, learnable_rates=None):
    """Copy and resolve settings; None means no CLI override, so the file default is retained.

    The result includes derived level_rates for training, structural construction, and resolved_config.json.
    learnable_rates=False is an explicit override, not an omitted value; avoid a truthiness check.
    """
    result = deepcopy(config)
    settings = result["diffusion"]
    if initial_shares is not None:
        settings["initial_shares"] = list(initial_shares)
    if "initial_shares" in settings:
        shares = validate_initial_shares(settings["initial_shares"])
        # An explicit share override on a legacy configuration retains its existing total budget.
        budget = float(settings.get("total_budget", sum(settings.get("level_rates", []))))
        if not math.isfinite(budget) or budget <= 0:
            raise ValueError("total_budget must be finite and positive; the default is -log(0.1)")
        settings["initial_shares"] = shares
        settings["total_budget"] = budget
        settings["level_rates"] = [budget * value for value in shares]
    elif "level_rates" not in settings:
        raise ValueError("Configuration is missing diffusion.initial_shares and diffusion.total_budget")
    if learnable_rates is not None:
        settings["dynamic_rate_gate"] = learnable_rates
    return result


def load_config(path, *, initial_shares=None, learnable_rates=None):
    """Read UTF-8 JSON, apply explicit overrides, and validate the initial shares."""
    config = json.loads(Path(path).read_text(encoding="utf-8"))
    return resolve_config(config, initial_shares=initial_shares, learnable_rates=learnable_rates)
