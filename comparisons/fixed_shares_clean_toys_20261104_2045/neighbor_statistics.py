"""Training-only neighbor statistics and sparse whole-history destinations.

Clean mode excludes each sample's final target from graph edges. All splits use
the same table; final targets are accepted only to reproduce legacy controls.
"""
# Note: notes.md explains self-inclusion, shared-table scope, and the review gate.
from collections import Counter, defaultdict
import numpy as np
from scipy import sparse


def count_transitions(sequence, targets, item_count, mode='nn_clean', lags=3):
    """Count every retained position pair once, with inverse-distance weights.

    nn_clean: history-to-history only; nn: history plus final target; no: the
    last up-to-three history positions to the final target only. Self-pairs
    from actual repeated item occurrences are retained here.
    """
    if mode not in ('nn_clean', 'nn', 'no') or lags < 1:
        raise ValueError('Use nn_clean, nn, or no and a positive lag window')
    rows = defaultdict(Counter)
    for history, target in zip(sequence, targets):
        valid = [int(x) for x in history if 0 <= x < item_count]
        if mode == 'no':
            for lag, source in enumerate(reversed(valid[-lags:]), 1):
                rows[source][int(target)] += 1. / lag
        else:
            chain = valid if mode == 'nn_clean' else valid + [int(target)]
            for position in range(1, len(chain)):
                for lag in range(1, min(lags, position) + 1):
                    rows[chain[position-lag]][chain[position]] += 1. / lag
    return dict(rows)


def normalized_row(counts, source, width=16, self_prior=1.):
    """Apply the legacy self prior, deterministic top-K, and normalization."""
    values = Counter(counts)
    values[source] += self_prior
    selected = sorted(((j, w) for j, w in values.items() if w > 1e-12), key=lambda p: (-p[1], p[0]))[:width]
    if not selected:
        return np.array([source]), np.ones(1)
    ids = np.array([p[0] for p in selected], dtype=np.int64)
    weights = np.array([p[1] for p in selected], dtype=np.float64)
    return ids, weights / weights.sum()


def build_table(counts, item_count, width=16, self_prior=1.):
    """Return one shared CSR table and its nonzero-support diagnostics."""
    indices, values, indptr = [], [], [0]
    for source in range(item_count):
        ids, weights = normalized_row(counts.get(source, {}), source, width, self_prior)
        indices.extend(ids)
        values.extend(weights)
        indptr.append(len(indices))
    table = sparse.csr_matrix((values, indices, indptr), shape=(item_count, item_count))
    np.testing.assert_allclose(np.asarray(table.sum(1)).ravel(), 1., atol=1e-14)
    return table


def history_weights(sequence, item_count, decay=.2):
    """Normalize exp(-decay*age) over all valid positions, combining repeats."""
    rows, cols, values = [], [], []
    for row, sequence_row in enumerate(sequence):
        valid = [(p, int(x)) for p, x in enumerate(sequence_row) if 0 <= x < item_count]
        if not valid:
            raise ValueError('At least one history item is required')
        weights = np.array([np.exp(-decay*(len(sequence_row)-1-p)) for p, _ in valid])
        weights /= weights.sum()
        for (_, item), weight in zip(valid, weights):
            rows.append(row)
            cols.append(item)
            values.append(weight)
    return sparse.csr_matrix((values, (rows, cols)), shape=(len(sequence), item_count))


def remove_own_final_edges(sequence, targets, counts, table, item_count):
    """Exact paired intervention: subtract only this sample's final-label edges.

    Retop each changed source row after subtraction, including self prior and
    tie rules. Earlier history edges stay in place; no approximation through
    subtraction from already normalized/truncated rows is used.
    """
    histories = history_weights(sequence, item_count)
    indices, values, indptr = [], [], [0]
    for row, (history, target) in enumerate(zip(sequence, targets)):
        valid = [int(x) for x in history if 0 <= x < item_count]
        changes = Counter()
        for lag, source in enumerate(reversed(valid[-3:]), 1):
            changes[source] += 1. / lag
        changed = {}
        for source, amount in changes.items():
            modified = Counter(counts.get(source, {}))
            modified[int(target)] -= amount
            if modified[int(target)] < -1e-9:
                raise AssertionError('Cannot remove unobserved final-target contribution')
            changed[source] = normalized_row(modified, source)
        # Sum nonnegative constituent rows afresh. Subtracting two normalized
        # sparse mixtures leaves roundoff residues at mathematically zero
        # entries, which can change ranks among equal background scores.
        combined = defaultdict(float)
        start, end = histories.indptr[row:row+2]
        for source, history_weight in zip(histories.indices[start:end], histories.data[start:end]):
            if source in changed:
                ids, weights = changed[source]
            else:
                left, right = table.indptr[source:source+2]
                ids, weights = table.indices[left:right], table.data[left:right]
            for target_id, weight in zip(ids, weights):
                combined[int(target_id)] += history_weight * weight
        for target_id in sorted(combined):
            indices.append(target_id)
            values.append(combined[target_id])
        indptr.append(len(indices))
    destination = sparse.csr_matrix((values, indices, indptr), shape=(len(sequence), item_count))
    assert destination.data.min(initial=0) >= 0
    np.testing.assert_allclose(np.asarray(destination.sum(1)).ravel(), 1., atol=1e-12)
    return destination
