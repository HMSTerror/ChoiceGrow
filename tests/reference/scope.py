"""Train-only native community/catalog scopes and the historical packed partition."""
# Note: docs/decisions/implemented/feature/2026-11-02-native-community-two-level.md
from collections import Counter, defaultdict

import numpy as np
import torch


def directed_edges(training, item_count):
    edges = Counter()
    for history, target in zip(training["seq"].tolist(), training["target"].tolist()):
        chain = [v for v in history if v < item_count] + [target]
        edges.update(zip(chain[:-1], chain[1:]))
    return edges


def native_memberships(training, item_count, seed, restarts):
    """Select native Infomap modules using training-graph codelength only."""
    from infomap import Infomap
    if restarts < 1:
        raise ValueError("At least one Infomap restart is required")
    edges = directed_edges(training, item_count)
    best, trials = None, []
    for restart in range(restarts):
        trial_seed = seed + restart + 1
        model = Infomap(f"--directed --silent --num-trials 1 --seed {trial_seed}")
        for item in range(item_count):
            model.add_node(item)
        for (source, target), weight in sorted(edges.items()):
            model.add_link(source, target, float(weight))
        model.run()
        paths = {int(k): tuple(int(v) for v in path) for k, path in model.get_multilevel_modules().items()}
        trial = {"restart": restart, "seed": trial_seed, "codelength": float(model.codelength)}
        trials.append(trial)
        if best is None or trial["codelength"] < best[0]:
            best = trial["codelength"], paths, trial
    return {"selection": "minimum training graph codelength", "selected": best[2],
            "trials": trials, "memberships": best[1]}


def community_root_blocks(memberships, item_count):
    """Keep every native leaf community intact; add the global catalog root."""
    paths = {int(item): tuple(path) for item, path in memberships.items()}
    if set(paths) != set(range(item_count)) or any(not path for path in paths.values()):
        raise ValueError("Every catalog item must have a native community path")
    labels = {path: index for index, path in enumerate(sorted(set(paths.values())))}
    community = torch.tensor([labels[paths[item]] for item in range(item_count)], dtype=torch.long)
    return torch.stack((community, torch.zeros_like(community)))


def real_partition(training, item_count, fine_size, coarse_size, seed, restarts=3):
    """Directed Infomap modules, consensus ordering, and strictly nested packing."""
    from infomap import Infomap
    if fine_size < 1 or coarse_size < fine_size or coarse_size % fine_size or restarts < 1:
        raise ValueError("Use positive restarts and nested fine/coarse capacities")
    edges = directed_edges(training, item_count)
    memberships = []
    for time in (1., 2., 4., 8., 16.):
        best = None
        for restart in range(restarts):
            model = Infomap(f"--directed --two-level --silent --num-trials 1 --seed {seed + restart + 1} --markov-time {time}")
            for node in range(item_count):
                model.add_node(node)
            for (source, target), weight in sorted(edges.items()):
                model.add_link(source, target, float(weight))
            model.run()
            candidate = (float(model.codelength), {int(k): int(v) for k, v in model.get_modules().items()})
            if best is None or candidate[0] < best[0]:
                best = candidate
        memberships.append([best[1].get(i, item_count + i) for i in range(item_count)])
    fine = memberships[0]
    groups = defaultdict(list)
    for item, label in enumerate(fine):
        groups[label].append(item)

    def signature(items):
        return tuple(sorted(Counter(level[i] for i in items).items(), key=lambda row: (-row[1], row[0]))[0][0]
                     for level in memberships[1:])

    rng = np.random.default_rng(seed)
    ordered = []
    for _, items in sorted(groups.items(), key=lambda pair: (signature(pair[1]), pair[0])):
        ordered.extend(rng.permutation(items).tolist())
    rank = torch.empty(item_count, dtype=torch.long)
    rank[torch.tensor(ordered)] = torch.arange(item_count)
    return torch.stack((rank // fine_size, rank // coarse_size, torch.zeros_like(rank)))
