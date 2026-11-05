"""Read-only trajectory and final-score diagnostics for three retained Toys fits.

This file imports the experiment's frozen numerical modules, never a Trainer.
The observed sampler keeps the original arithmetic and random-call order, adding
only clones of the initial state and the state after each outer noise interval.
No target is passed to the sampler or the score corrections.
"""
# Note: ../../docs/decisions/implemented/feature/2026-11-04-fixed-shares.md
import argparse
import csv
from pathlib import Path
import time
import traceback

import experiment as p
from data import loader
from evaluate import evaluation_rng


OUTPUT = p.ROOT / 'runs/reverse_diagnostics_20261105'
RUNS = (
    ('Root', p.ROOT / 'runs/patience_alignment/root_p300', 1810),
    ('hr', p.ROOT / 'runs/step2/hr_s01', 1770),
    ('hs', p.ROOT / 'runs/step2/hs_s01', 1870),
)
ETAS = (0., .05, .1, .2)


@p.torch.no_grad()
def sample_observed(model, sequence, guidance, kernel, settings):
    """Run frozen ReverseSampler arithmetic with 21 observational snapshots.

    Point0 is the terminal-reference draw at sigma10. Points1..20 are states
    after the20 outer intervals; adaptive internal jumps remain unchanged.
    """
    torch = p.torch
    context = kernel.generator.context(sequence)
    history = model.encode_history(sequence)
    initial = kernel.terminal_reference(settings['sigma_max'], context)
    current = torch.multinomial(initial, 1).squeeze(1)
    states = [current.clone()]
    grid = torch.linspace(settings['sigma_max'], settings['sigma_min'],
                          settings['reverse_intervals'] + 1, device=sequence.device)
    step_count = torch.zeros(len(sequence), device=sequence.device)
    cap_count = 0
    for high, low in zip(grid[:-1], grid[1:]):
        remaining = torch.full((len(sequence),), float(high - low), device=sequence.device)
        for _ in range(settings['max_substeps']):
            active = remaining > 1e-7
            if not active.any():
                break
            sigma = low + remaining
            scores = model.guided_scores(history, current, sigma, guidance)
            incoming = kernel.generator.incoming_rates(current, context)
            log_rates = incoming.log() + scores
            log_exit = torch.logsumexp(log_rates, -1)
            if not torch.isfinite(log_exit).all():
                raise FloatingPointError('Nonfinite reverse exit rate')
            effective = log_exit.clamp_max(settings['log_exit_cap'])
            cap_count += int(((effective < log_exit) & active).sum())
            dt = torch.minimum(remaining, settings['reverse_cfl'] * torch.exp(-effective))
            if torch.any(active & (dt <= 0)):
                raise FloatingPointError('Reverse step underflow')
            event = (torch.rand_like(remaining) < dt * effective.exp()) & active
            destination = torch.multinomial((log_rates - log_exit[:, None]).exp(), 1).squeeze(1)
            current = torch.where(event, destination, current)
            remaining = (remaining - dt).clamp_min(0)
            step_count += active
        if torch.any(remaining > 1e-7):
            raise RuntimeError('Reverse substep budget exceeded')
        states.append(current.clone())
    endpoint = torch.full((len(sequence),), settings['sigma_min'], device=sequence.device)
    scores = model.guided_scores(history, current, endpoint, guidance)
    return scores, {'mean_reverse_substeps': float(step_count.mean()),
                    'exit_cap_activations': cap_count}, torch.stack(states, 1), context


def final_rate_terms(generator, current, context):
    """Return logQ off the diagonal, and its community-only centered version.

    The zero diagonal is not a probability: Q[x,x] is a negative exit rate in
    the actual generator. The incoming-rate helper masks it to zero. Both
    ranking heuristics explicitly retain the existing self score instead.
    """
    torch = p.torch
    incoming = generator.incoming_rates(current, context)
    safe = incoming.scatter(1, current[:, None], 1.)
    if torch.any(safe <= 0):
        raise ValueError('Off-diagonal rates must be positive')
    raw = safe.log()
    root_rate = context.level_rates[-1]
    rho_x = context.rho.gather(1, current[:, None])
    root_mass = context.block_mass[-1].gather(1, current[:, None])
    outside = generator.reset_rate * generator.base[current, None] + root_rate * rho_x / root_mass
    centered = (raw - outside.log()).scatter(1, current[:, None], 0.)
    assert torch.all(raw.gather(1, current[:, None]) == 0)
    assert torch.all(centered.gather(1, current[:, None]) == 0)
    return raw, centered


def metrics_from_top10(top10, target, tail):
    """Compute stream means with the same full-catalog tail-row definition."""
    np = p.np
    result = {'streams': int(len(top10)), 'examples': int(len(target)), 'overall': {},
              'tail': {'examples': int(tail[target].sum())},
              'head': {'examples': int((~tail[target]).sum())}, 'per_stream': []}
    for ranked in top10:
        row = {'overall': {}, 'tail': {}, 'head': {}}
        for k in (5, 10):
            hits = ranked[:, :k] == target[:, None]
            found, rank = hits.any(-1), hits.argmax(-1) + 1
            for key, value in ((f'hr@{k}', found), (f'ndcg@{k}', found / np.log2(rank + 1))):
                row['overall'][key] = float(value.mean())
                row['tail'][key] = float(value[tail[target]].mean())
                row['head'][key] = float(value[~tail[target]].mean())
            row['overall'][f'tail_exposure@{k}'] = float(tail[ranked[:, :k]].mean())
        result['per_stream'].append(row)
    for group in ('overall', 'tail', 'head'):
        for key in result['per_stream'][0][group]:
            result[group][key] = float(np.mean([row[group][key] for row in result['per_stream']]))
    return result


def curve_rows(states, target, hs, tail, grid):
    """All models are measured against the SAME target hs block, including hr."""
    np = p.np
    correct = hs[states] == hs[target][None, :, None]
    exact = states == target[None, :, None]
    rows = []
    for index, sigma in enumerate(grid):
        row = {'step': index, 'sigma': float(sigma)}
        for name, mask in (('overall', np.ones(len(target), dtype=bool)), ('tail', tail[target])):
            hit = correct[:, mask, index]
            row[f'{name}_block_hit'] = float(hit.mean())
            row[f'{name}_item_hit'] = float(exact[:, mask, index].mean())
            row[f'{name}_block_hit_per_stream'] = hit.mean(1).tolist()
            if index:
                previous = correct[:, mask, index - 1]
                row[f'{name}_entered_fraction'] = float((~previous & hit).mean())
                row[f'{name}_left_fraction'] = float((previous & ~hit).mean())
                row[f'{name}_retention_given_previous_hit'] = float(hit[previous].mean()) if previous.any() else None
        rows.append(row)
    return rows


def parity_check(model, process, values, config, device):
    """Check observations do not change first-batch output or RNG consumption."""
    torch = p.torch
    sequence, _, _ = next(iter(loader(values, config['training']['eval_batch_size'])))
    sequence = sequence.to(device)
    with evaluation_rng(config['seed'], 'val', 0, device):
        original, telemetry = process.sample_scores(model, sequence, config['training']['guidance_strength'])
        original_rng = torch.cuda.get_rng_state(device).clone()
    with evaluation_rng(config['seed'], 'val', 0, device):
        observed, observed_telemetry, states, _ = sample_observed(
            model, sequence, config['training']['guidance_strength'], process.kernel, process.settings)
        observed_rng = torch.cuda.get_rng_state(device).clone()
    max_diff = float((original - observed).abs().max())
    same_rng = torch.equal(original_rng, observed_rng)
    assert torch.equal(original.topk(10, -1).indices, observed.topk(10, -1).indices)
    assert same_rng and telemetry == observed_telemetry
    assert max_diff < 1e-6
    assert states.shape == (len(sequence), 21)
    return {'max_abs_score_difference': max_diff, 'same_top10': True,
            'same_cuda_rng_after_sampling': same_rng, 'same_telemetry': True}


@p.torch.no_grad()
def evaluate_run(name, folder, epoch, output, hs, tail, device):
    torch, np = p.torch, p.np
    config = p.load_config(folder / 'resolved_config.json')
    torch.set_num_threads(config['threads'])
    process = p.ChoiceDiffusion.from_directory(folder, config, device)
    state = p.CheckpointStore(folder, config).load_best(process, device)
    assert state['epoch'] == epoch and not process.generator.rate_policy.is_learned
    model = p.make_denoiser(folder, config, process).to(device)
    model.load_state_dict(state['model']); model.eval()
    validation = p.load_split(folder, 'val')
    parity = parity_check(model, process, validation, config, device)
    result = {'run': name, 'source': str(folder), 'best_epoch': epoch,
              'best_validation_ndcg10': state['validation']['ndcg@10'],
              'checkpoint_sha256': p.sha256(folder / 'choice_best.pt'), 'sampler_parity': parity, 'splits': {}}
    for split in ('val', 'test'):
        p.write_json(output / 'status.json', {'state': 'evaluating', 'run': name, 'split': split, 'updated_at': p.now()})
        values = validation if split == 'val' else p.load_split(folder, split)
        targets = values['target'].numpy()
        streams = config['training']['validation_streams'] if split == 'val' else config['training']['test_streams']
        all_states, all_top10 = [], {}
        # Test receives eta0 only. Additional score coefficients never touch test data.
        variants = [('base', 0.)] + ([(mode, eta) for mode in ('raw_log_q', 'community_only')
                                     for eta in ETAS[1:]] if split == 'val' else [])
        self_stats, boosts = [], []
        all_telemetry = []
        for stream in range(streams):
            batches, ranks = [], {f'{mode}:{eta:g}': [] for mode, eta in variants}
            substeps, caps = 0., 0
            with evaluation_rng(config['seed'], split, stream, device):
                for sequence, _, target in loader(values, config['training']['eval_batch_size']):
                    scores, telemetry, states, context = sample_observed(model, sequence.to(device),
                        config['training']['guidance_strength'], process.kernel, process.settings)
                    current = states[:, -1]
                    assert torch.all(scores.gather(1, current[:, None]) == 0)
                    batches.append(states.cpu().numpy())
                    if split == 'val':
                        raw, centered = final_rate_terms(process.generator, current, context)
                        if name == 'Root':
                            assert torch.count_nonzero(centered) == 0
                        boost = centered.amax(-1)
                        boosts.append(boost.cpu().numpy())
                    base_rank = scores.topk(10, -1).indices
                    for mode, eta in variants:
                        key = f'{mode}:{eta:g}'
                        adjusted = scores if not eta else scores + eta * (raw if mode == 'raw_log_q' else centered)
                        assert torch.all(adjusted.gather(1, current[:, None]) == 0)
                        rank = base_rank if not eta else adjusted.topk(10, -1).indices
                        if name == 'Root' and mode == 'community_only':
                            assert torch.equal(rank, base_rank)
                        ranks[key].append(rank.cpu().numpy())
                    self_stats.append((base_rank[:, 0] == current).cpu().numpy())
                    substeps += telemetry['mean_reverse_substeps'] * len(target)
                    caps += telemetry['exit_cap_activations']
            all_states.append(np.concatenate(batches))
            for key, batch in ranks.items():
                all_top10.setdefault(key, []).append(np.concatenate(batch))
            all_telemetry.append({'mean_reverse_substeps': substeps / len(targets), 'exit_cap_activations': caps})
        all_states = np.stack(all_states)
        all_top10 = {key: np.stack(value) for key, value in all_top10.items()}
        original_cache = folder / ('validation_recommendations.npz' if split == 'val' else 'recommendations.npz')
        with np.load(original_cache) as saved:
            np.testing.assert_array_equal(saved['target'], targets)
            cache_agreement = float((saved['top10'] == all_top10['base:0']).mean())
        baseline = metrics_from_top10(all_top10['base:0'], targets, tail)
        saved_metrics = p.read(folder / ('validation_report.json' if split == 'val' else 'report.json'))[
            'validation' if split == 'val' else 'test']
        metric_delta = {key: baseline['overall'][key] - saved_metrics[key] for key in p.METRICS}
        # CPU recomputation of NDCG can differ by float32 summation rounding only.
        if cache_agreement != 1.:
            raise AssertionError(f'{name}/{split}: eta0 predictions differ from retained evaluation cache: {cache_agreement}')
        assert max(map(abs, metric_delta.values())) < 1e-7
        grid = torch.linspace(config['diffusion']['sigma_max'], config['diffusion']['sigma_min'], 21).numpy()
        rows = curve_rows(all_states, targets, hs, tail, grid)
        variant_metrics = {key: metrics_from_top10(rank, targets, tail) for key, rank in all_top10.items()}
        for key, rank in all_top10.items():
            variant_metrics[key]['top1_equals_final_state'] = float((rank[:, :, 0] == all_states[:, :, -1]).mean())
            variant_metrics[key]['top1_in_target_hs_block'] = float((hs[rank[:, :, 0]] == hs[targets][None]).mean())
            variant_metrics[key]['top10_entries_in_target_hs_block'] = float((hs[rank] == hs[targets][None, :, None]).mean())
        split_result = {'examples': len(targets), 'streams': streams, 'curve': rows,
                        'baseline_cache_exact': cache_agreement == 1., 'baseline_metric_difference': metric_delta,
                        'sampling_telemetry': all_telemetry, 'scores': variant_metrics}
        if boosts:
            boost_values = np.concatenate(boosts)
            split_result['community_log_boost'] = {key: float(value) for key, value in zip(
                ('mean', 'p05', 'p50', 'p95', 'max'),
                (boost_values.mean(), *np.quantile(boost_values, [.05, .5, .95]), boost_values.max()))}
        np.savez_compressed(output / f'{name}_{split}_states.npz', states=all_states, target=targets,
                            sigma=grid, hs_block=hs, tail_item=tail)
        np.savez_compressed(output / f'{name}_{split}_top10.npz', target=targets,
                            **{key.replace(':', '_'): value for key, value in all_top10.items()})
        result['splits'][split] = split_result
        p.write_json(output / f'{name}.json', result)
        print(p.json.dumps({'event': 'evaluated', 'run': name, 'split': split,
                           'baseline_ndcg10': baseline['overall']['ndcg@10'],
                           'final_block_hit': rows[-1]['overall_block_hit'], 'cache_exact': cache_agreement == 1.}), flush=True)
    assert result['checkpoint_sha256'] == p.sha256(folder / 'choice_best.pt')
    return result


def write_tables(results, output):
    curves, scores = [], []
    for result in results:
        for split, values in result['splits'].items():
            for row in values['curve']:
                curves.append({'run': result['run'], 'split': split, **{
                    key: value for key, value in row.items() if not isinstance(value, list)}})
            for variant, row in values['scores'].items():
                mode, eta = variant.split(':')
                scores.append({'run': result['run'], 'split': split, 'mode': mode, 'eta': float(eta),
                    **row['overall'], **{f'tail_{key}': value for key, value in row['tail'].items()},
                    'top1_equals_final_state': row['top1_equals_final_state']})
    for filename, rows in (('commitment.csv', curves), ('validation_score_sweep.csv', [r for r in scores if r['split'] == 'val'])):
        fields = list(dict.fromkeys(key for row in rows for key in row))
        with (output / filename).open('w', encoding='utf-8', newline='') as stream:
            writer = csv.DictWriter(stream, fields); writer.writeheader(); writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=OUTPUT)
    args = parser.parse_args()
    output = args.output
    output.mkdir(parents=True, exist_ok=False)
    p.write_json(output / 'status.json', {'state': 'preparing', 'started_at': p.now()})
    started = time.perf_counter()
    try:
        with p.np.load(p.STEP0 / 'partitions/Toys/partitions.npz') as saved:
            hs = saved['hs'].copy()
        hs_artifact = p.torch.load(RUNS[2][1] / 'structure.pt', map_location='cpu', weights_only=True)
        p.np.testing.assert_array_equal(hs, hs_artifact['blocks'][0].numpy())
        tail, cutoff = p.tail_mask(RUNS[0][1])
        assert cutoff == 12 and tail.sum() == 9367 and len(p.np.unique(hs)) == 12
        reference = p.torch.load(RUNS[0][1] / 'structure.pt', map_location='cpu', weights_only=True)
        for _, folder, _ in RUNS[1:]:
            artifact = p.torch.load(folder / 'structure.pt', map_location='cpu', weights_only=True)
            for key in ('base', 'neighbor_ids', 'neighbor_weights'):
                assert p.torch.equal(reference[key], artifact[key])
            assert p.split_hashes(folder) == p.split_hashes(RUNS[0][1])
        results = [evaluate_run(name, folder, epoch, output, hs, tail, p.torch.device('cuda:0'))
                   for name, folder, epoch in RUNS]
        write_tables(results, output)
        summary = {'completed_at': p.now(), 'elapsed_seconds': time.perf_counter() - started,
            'training_runs': 0, 'seed': 100, 'evaluation_streams': {'val': 1, 'test': 3},
            'curve_definition': 'P(hs_block(x_state)=hs_block(true_target)); terminal draw plus20 interval endpoints',
            'curve_steps': 21, 'hs_blocks': 12, 'tail_cutoff_exclusive': cutoff, 'tail_items': int(tail.sum()),
            'score_variants_split': 'validation only; no corrected score is evaluated on test',
            'score_modes': {'raw_log_q': 'z + eta*log(Q[y,x]), off-diagonal only, self keeps z[x]=0',
                'community_only': 'z + eta*log(Q[y,x]/outside_rate(x)); self keeps z[x]=0'},
            'results': results}
        p.write_json(output / 'summary.json', summary)
        p.write_json(output / 'status.json', {'state': 'complete', 'completed_at': p.now()})
        print(p.json.dumps({'event': 'complete', 'output': str(output)}), flush=True)
    except BaseException:
        p.write_json(output / 'status.json', {'state': 'failed', 'updated_at': p.now(), 'traceback': traceback.format_exc()})
        raise


if __name__ == '__main__':
    main()
