"""CPU-only history graph and capacity-constrained spectral partitions.

Only item labels vary. All resulting community partitions nest in one catalog
Root, retaining the existing projection algebra for later experiments.
"""
# Note: notes.md records why count balance is exact but mass balance is measured.
from dataclasses import dataclass
import json
import time
import numpy as np
from scipy import sparse
from scipy.optimize import linear_sum_assignment
from scipy.sparse.linalg import svds


def directed_graph(counts, item_count):
    """Convert lag-weighted counts to CSR without adding neighbor self priors."""
    rows, cols, data = [], [], []
    for source in sorted(counts):
        for target, weight in sorted(counts[source].items()):
            rows.append(source)
            cols.append(target)
            data.append(weight)
    return sparse.csr_matrix((data, (rows, cols)), shape=(item_count, item_count), dtype=np.float64)


def spectral_embedding(directed, emb_dim=32, seed=100):
    """Return L2-normalized leading left singular vectors of normalized W+W.T.

    Real repeated-item self edges are removed for the affinity embedding.
    Isolates get zero features. svds uses a deterministic starting generator.
    Largest singular values are used as requested, not largest algebraic
    eigenvalues; this distinction matters for a symmetric indefinite matrix.
    """
    affinity = (directed + directed.T).tocsr()
    affinity.setdiag(0.)
    affinity.eliminate_zeros()
    degree = np.asarray(affinity.sum(1)).ravel()
    inverse = 1. / np.sqrt(np.maximum(degree, 1e-12))
    normalized = (sparse.diags(inverse) @ affinity @ sparse.diags(inverse)).tocsr()
    if not 0 < emb_dim < affinity.shape[0]:
        raise ValueError('Embedding dimension must be between zero and N')
    u, singular, _ = svds(normalized, k=emb_dim, rng=np.random.default_rng(seed), tol=1e-7)
    order = np.argsort(-singular, kind='stable')
    u = u[:, order]
    # Fix sign ambiguity so persisted features are easier to compare.
    for column in range(u.shape[1]):
        pivot = np.argmax(np.abs(u[:, column]))
        if u[pivot, column] < 0:
            u[:, column] *= -1
    u[degree == 0] = 0.
    u /= np.linalg.norm(u, axis=1, keepdims=True).clip(1e-12)
    return u, {'singular_values': singular[order].tolist(), 'isolated_items': int((degree == 0).sum()),
               'directed_edges': directed.nnz, 'affinity_edges': affinity.nnz, 'svds_tolerance': 1e-7}


@dataclass
class FrequencyStrata:
    """Shared item ordering and slot capacities for spectral/random assignment."""
    items: list
    slots: list
    cluster_count: int

    @classmethod
    def build(cls, base, cluster_count=12, multiplier=20, seed=100):
        if cluster_count < 1 or multiplier < 1 or cluster_count > len(base):
            raise ValueError('Invalid stratum capacity')
        order = np.lexsort((np.arange(len(base)), -np.asarray(base)))
        width = cluster_count * multiplier
        items, slots = [], []
        for start in range(0, len(base), width):
            group = order[start:start+width]
            q, remainder = divmod(len(group), cluster_count)
            capacity = np.full(cluster_count, q, dtype=np.int64)
            extra = np.random.default_rng(seed+177).permutation(cluster_count)[:remainder]
            capacity[extra] += 1
            items.append(group)
            slots.append(np.repeat(np.arange(cluster_count), capacity))
        return cls(items, slots, cluster_count)

    def random_labels(self, seed=1099):
        labels = np.empty(sum(map(len, self.items)), dtype=np.int64)
        rng = np.random.default_rng(seed)
        for items, slots in zip(self.items, self.slots):
            labels[items] = rng.permutation(slots)
        return labels


def balanced_assignment(features, initial_centers, groups, iterations=10):
    """Spherical Lloyd iterations with exact optimal capacity assignment per group.

    The dense Hungarian problem has at most M*K rows for stratified groups.
    The size-only control instead has one full-catalog group; repeated columns
    are compressed by a transportation LP to avoid an N-by-N dense matrix.
    """
    from scipy.optimize import linprog
    centers = initial_centers.copy()
    k = len(centers)
    labels = np.zeros(len(features), dtype=np.int64)
    trace = []
    for iteration in range(iterations):
        started = time.perf_counter()
        similarities = features @ centers.T
        for items, slots in zip(groups.items, groups.slots):
            if len(items) <= 640:
                rows, columns = linear_sum_assignment(-similarities[items][:, slots])
                labels[items[rows]] = slots[columns]
            else:
                count = len(items)
                row_ids = np.arange(count*k)
                item_rows = np.repeat(np.arange(count), k)
                cluster_rows = np.tile(np.arange(k), count)
                constraints = sparse.coo_matrix((np.ones(2*count*k),
                    (np.concatenate((item_rows, count+cluster_rows)), np.tile(row_ids, 2))),
                    shape=(count+k, count*k)).tocsr()
                capacity = np.bincount(slots, minlength=k)
                fit = linprog(-similarities[items].ravel(), A_eq=constraints,
                              b_eq=np.concatenate((np.ones(count), capacity)), bounds=(0., 1.),
                              method='highs-ipm', options={'presolve': True})
                if not fit.success:
                    raise RuntimeError(f'Balanced transportation failed: {fit.message}')
                solution = fit.x.reshape(count, k)
                if np.max(np.abs(solution - np.round(solution))) > 1e-6:
                    raise RuntimeError('Expected an integral transportation solution')
                labels[items] = solution.argmax(1)
            assert np.array_equal(np.bincount(labels[items], minlength=k), np.bincount(slots, minlength=k))
        for cluster in range(k):
            members = features[labels == cluster]
            if len(members):
                center = members.mean(0)
                centers[cluster] = center / max(np.linalg.norm(center), 1e-12)
        trace.append({'iteration': iteration+1, 'mean_assigned_cosine': float(np.mean(np.sum(features*centers[labels], axis=1)))})
        print(json.dumps({'stage':'assignment_iteration','groups':len(groups.items),
              'iteration':iteration+1,'elapsed_seconds':time.perf_counter()-started}),flush=True)
    return labels, centers, trace


def build_stratified_partitions(directed, base, cluster_count=12, multiplier=20, emb_dim=32, iterations=10, seed=100):
    """Build hs, hr, and size-only spherical balanced-k-means on shared features."""
    print(json.dumps({'stage':'svds_start'}),flush=True)
    features, graph_info = spectral_embedding(directed, emb_dim, seed)
    print(json.dumps({'stage':'svds_complete'}),flush=True)
    strata = FrequencyStrata.build(base, cluster_count, multiplier, seed)
    rng = np.random.default_rng(seed)
    pool = np.concatenate(strata.items)[:min(len(base), cluster_count*20)]
    centers = features[rng.choice(pool, size=cluster_count, replace=False)].copy()
    centers /= np.linalg.norm(centers, axis=1, keepdims=True).clip(1e-12)
    hs, hs_centers, hs_trace = balanced_assignment(features, centers, strata, iterations)
    hr = strata.random_labels(seed+999)
    # One group imposes equal total sizes without frequency-stratum constraints.
    plain = FrequencyStrata.build(base, cluster_count, int(np.ceil(len(base)/cluster_count)), seed)
    balanced, balanced_centers, balanced_trace = balanced_assignment(features, centers, plain, iterations)
    for items, slots in zip(strata.items, strata.slots):
        expected = np.bincount(slots, minlength=cluster_count)
        assert np.array_equal(np.bincount(hs[items], minlength=cluster_count), expected)
        assert np.array_equal(np.bincount(hr[items], minlength=cluster_count), expected)
    return {'hs': hs, 'hr': hr, 'balanced_kmeans': balanced}, features, {
        'hs': hs_centers, 'balanced_kmeans': balanced_centers}, {
        'graph': graph_info, 'hs_iterations': hs_trace, 'balanced_iterations': balanced_trace,
        'strata_sizes': list(map(len, strata.items)),
        'shared_stratum_capacities': [np.bincount(s, minlength=cluster_count).tolist() for s in strata.slots],
        'exact_count_match': True, 'exact_weighted_mass_match_guaranteed': False,
        'remainder_policy': 'floor/ceiling counts shared by hs/hr; overall block sizes differ at most one'}, strata


def infomap_labels(directed, seed=100, restarts=3):
    """Select native leaf labels by directed training-graph codelength only."""
    from infomap import Infomap
    graph = directed.tocoo()
    best, trials = None, []
    for restart in range(restarts):
        model = Infomap(f'--directed --silent --num-trials 1 --seed {seed+restart+1}')
        for item in range(directed.shape[0]):
            model.add_node(item)
        for source, target, weight in zip(graph.row, graph.col, graph.data):
            model.add_link(int(source), int(target), float(weight))
        model.run()
        paths = model.get_multilevel_modules()
        unique = {path: i for i, path in enumerate(sorted(set(tuple(p) for p in paths.values())))}
        labels = np.array([unique[tuple(paths[i])] for i in range(directed.shape[0])])
        trial = {'seed': seed+restart+1, 'codelength': float(model.codelength), 'clusters': len(unique)}
        trials.append(trial)
        if best is None or trial['codelength'] < best[0]:
            best = trial['codelength'], labels
    return best[1], {'selection': 'minimum training-graph codelength', 'trials': trials}
