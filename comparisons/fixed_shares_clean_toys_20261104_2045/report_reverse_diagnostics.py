"""Render completed evaluation-only diagnostic artifacts into tables and curves."""
import csv
from pathlib import Path

import experiment as p
from diagnose_reverse import OUTPUT


def main():
    summary = p.read(OUTPUT / 'summary.json')
    assert p.read(OUTPUT / 'status.json')['state'] == 'complete'
    results = {row['run']: row for row in summary['results']}
    persistence = []
    for name in results:
        for split in ('val', 'test'):
            with p.np.load(OUTPUT / f'{name}_{split}_states.npz') as saved:
                hit = saved['hs_block'][saved['states']] == saved['hs_block'][saved['target']][None, :, None]
            given = hit[:, :, 10]
            persistence.append({'run': name, 'split': split,
                'given_target_block_at_step10_count': int(given.sum()),
                'final_block_hit_given_step10_hit': float(hit[:, :, -1][given].mean()),
                'all_step10_to20_observed_endpoints_in_target_block_given_step10_hit': float(hit[:, :, 10:].all(-1)[given].mean()),
                'final_given_step19_hit': float(hit[:, :, -1][hit[:, :, 19]].mean())})
    p.write_json(OUTPUT / 'persistence.json', {'meaning':
        'Persistence at outer interval endpoints; internal departure and return remain possible.', 'rows': persistence})
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.ticker import PercentFormatter

    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 10, 'axes.spines.top': False,
                         'axes.spines.right': False, 'svg.fonttype': 'none'})
    figure, axes = plt.subplots(2, 2, figsize=(11.5, 7.6), sharex=True)
    colors = {'Root': '#555B65', 'hr': '#4C8DBB', 'hs': '#D96F37'}
    for column, split in enumerate(('val', 'test')):
        for line, group in enumerate(('overall', 'tail')):
            axis = axes[line, column]
            for name, result in results.items():
                curve = result['splits'][split]['curve']
                axis.plot([row['step'] for row in curve], [row[f'{group}_block_hit'] for row in curve],
                          label=name, color=colors[name], linewidth=2, marker='o', markersize=3)
            axis.set_title(f"{'Validation (1 stream)' if split == 'val' else 'Test (3 streams)'} — {group}")
            axis.set_ylabel('State in true target hs block')
            axis.yaxis.set_major_formatter(PercentFormatter(1, decimals=1))
            axis.set_xticks(range(0, 21, 5)); axis.grid(axis='y', alpha=.18)
            axis.set_xlabel('Completed reverse intervals (0: sigma10; 20: sigma0.001)')
    axes[0, 0].legend(frameon=False, ncol=3)
    figure.suptitle('Toys seed100: commitment to the target spectral block (K=12)', fontsize=13)
    figure.tight_layout(rect=(0, 0, 1, .96))
    for suffix in ('png', 'svg'):
        figure.savefig(OUTPUT / f'commitment.{suffix}', dpi=180, bbox_inches='tight')
    plt.close(figure)

    lines = ['# Reverse-path and final-score diagnostics: Toys seed100', '',
        'Evaluation only. Root best1810, hr0.10 best1770 and hs0.10 best1870 retain the matched patience300 protocol. No weights or training settings change. All models use bn, nn_clean and gamma0.02. Validation uses1 stream and testing uses3 streams, with the original full-catalog ranking and tail definition.', '',
        'The state curves show modest enrichment of the true target block, with reversals rather than a monotone lock. hs ends only0.1546 percentage points above Root on validation and0.7724 points above Root on test; hr ends within0.0515 points of hs on test. Raw eta0.05 gives hs a small validation NDCG@10 gain (+0.1947%), while larger tested coefficients reduce overall NDCG@10. These diagnostics do not establish a strong spectral-specific commitment mechanism or a missing-rate bug.', '',
        '## What the implementation ranks', '',
        'Frozen code evidence:', '',
        '- `code/diffusion.py:70`: reverse jumps use `log_rates = incoming.log() + scores`.',
        '- `code/diffusion.py:89`: the endpoint returns only `model.guided_scores(...)`.',
        '- `code/evaluate.py:67` and `:74`: final Top10 comes directly from those returned scores.',
        '- `code/models.py:189`: guidance combines raw conditional/unconditional scores, applies a bounded transform, and sets the current item score to0.',
        '- `code/transition.py:215`: incoming rates implement Q[y,x], with a zero mask at self; this mask must not be mistaken for the negative generator diagonal.', '',
        'For off-diagonal candidates in two-level models:', '',
        r'`Q_u[y,x] = a_u(x) + c_u(x) 1{C(y)=C(x)}`,',
        r'`a_u(x) = reset*b[x] + r_R*rho_u[x]/rho_u(catalog)`,',
        r'`c_u(x) = r_C*rho_u[x]/rho_u(C(x))`.', '',
        'Thus the final scorer omits logQ from its ranking expression, but this is not sufficient to establish a bug. An ideal unbounded score-ratio predictor estimates `log p_sigma(y|u) - log p_sigma(x|u)`, which already ranks the marginal item probabilities. Multiplication by Q instead ranks instantaneous reverse-jump hazards. Bounded scores and guidance introduce approximation, so the validation sweep tests a heuristic rather than proving that a generator factor is missing.', '',
        '## Commitment curves', '',
        'The initial terminal-reference draw plus20 completed outer intervals give21 state observations. Adaptive internal jump counts are not replaced by20 jumps. Every method, including hr, is measured against the true target\'s hs label; labels are used after sampling only. State block-hit is different from final Top10 relevance or community accuracy. Curves need not be monotone.', '',
        '| Split | Step | Sigma | Root block-hit | hr block-hit | hs block-hit | hs minus Root (pp) |',
        '| --- | ---: | ---: | ---: | ---: | ---: | ---: |']
    for split in ('val', 'test'):
        for index in (0, 5, 10, 15, 20):
            rows = {name: result['splits'][split]['curve'][index] for name, result in results.items()}
            lines.append(f"| {split} | {index} | {rows['Root']['sigma']:.6f} | " +
                ' | '.join(f"{100*rows[name]['overall_block_hit']:.4f}%" for name in ('Root', 'hr', 'hs')) +
                f" | {100*(rows['hs']['overall_block_hit']-rows['Root']['overall_block_hit']):+.4f} |")
    lines += ['', '| Split | Method | Initial tail block-hit | Final tail block-hit | Final exact-item hit (overall) |',
              '| --- | --- | ---: | ---: | ---: |']
    for split in ('val', 'test'):
        for name, result in results.items():
            curve = result['splits'][split]['curve']
            lines.append(f"| {split} | {name} | {100*curve[0]['tail_block_hit']:.4f}% | "
                         f"{100*curve[-1]['tail_block_hit']:.4f}% | {100*curve[-1]['overall_item_hit']:.4f}% |")
    lines += ['', 'Persistence after an intermediate correct-block state:', '',
              '| Split | Method | Final hit given hit at step10 | All recorded endpoints10..20 hit, given hit at10 |',
              '| --- | --- | ---: | ---: |']
    for row in p.read(OUTPUT / 'persistence.json')['rows']:
        lines.append(f"| {row['split']} | {row['run']} | {100*row['final_block_hit_given_step10_hit']:.4f}% | "
                     f"{100*row['all_step10_to20_observed_endpoints_in_target_block_given_step10_hit']:.4f}% |")
    lines += ['', 'A path may also leave and return inside any outer interval. Therefore endpoint persistence is an upper bound on staying inside the target block at every internal time.', '',
        '![Commitment curves](commitment.png)', '',
        '## Validation-only final-score sweep', '',
        'Both modes retain the self score at0. Raw uses `z+eta*log(Q[y,x])` off diagonal, exactly as requested. Community-only uses `z+eta*log(Q[y,x]/a_u(x))` off diagonal. Since self does not receive the common offset, these are different self-versus-other heuristics. The latter isolates the structural factor `log(1+c/a)` and leaves Root rankings exactly unchanged. Neither changes the sampled path.', '',
        '| hs score | eta | Val HR@10 | Val NDCG@10 | Tail HR@10 | Tail NDCG@10 | Tail exposure@10 | Top1 equals final state |',
        '| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |']
    hs_scores = results['hs']['splits']['val']['scores']
    for key, row in hs_scores.items():
        mode, eta = key.split(':')
        lines.append(f"| {mode} | {eta} | {row['overall']['hr@10']:.9f} | {row['overall']['ndcg@10']:.9f} | "
            f"{row['tail']['hr@10']:.9f} | {row['tail']['ndcg@10']:.9f} | "
            f"{100*row['overall']['tail_exposure@10']:.4f}% | {100*row['top1_equals_final_state']:.4f}% |")
    baseline = hs_scores['base:0']
    lines += ['', 'Validation effects relative to hs eta0:', '']
    for key, row in hs_scores.items():
        if key == 'base:0':
            continue
        change = row['overall']['ndcg@10'] - baseline['overall']['ndcg@10']
        tail_change = row['tail']['ndcg@10'] - baseline['tail']['ndcg@10']
        lines.append(f"- {key}: overall NDCG@10 {change:+.9f} ({100*change/baseline['overall']['ndcg@10']:+.4f}%); "
                     f"tail NDCG@10 {tail_change:+.9f} ({100*tail_change/baseline['tail']['ndcg@10']:+.4f}%).")
    lines += ['', 'Root raw-logQ negative control:', '',
              '| eta | Val NDCG@10 | Tail NDCG@10 | Top1 equals final state |',
              '| ---: | ---: | ---: | ---: |']
    for key, row in results['Root']['splits']['val']['scores'].items():
        if key.startswith('community_only:'):
            continue
        lines.append(f"| {key.split(':')[1]} | {row['overall']['ndcg@10']:.9f} | {row['tail']['ndcg@10']:.9f} | "
                     f"{100*row['top1_equals_final_state']:.4f}% |")
    lines += ['', 'The hr raw correction also changes validation NDCG@10: eta0=0.049791061, eta0.05=0.050351643, eta0.10=0.050420287, eta0.20=0.049931859. Thus any small hs improvement is not unique to the spectral partition. These control sweeps are validation-only and do not select a deployed method.', '',
        '## Reproduction and limits', '',
        'First-batch observed-sampler scores, Top10, telemetry and CUDA RNG state agree exactly with the frozen sampler. Complete eta0 validation/test Top10 arrays match all retained caches exactly. CPU metric recomputation differs only by float32 reduction rounding below1e-7. Original checkpoint hashes remain unchanged.', '',
        'The evaluation has1941 validation rows (989 tail targets) and1942 test rows (1035 tail targets). Tail items have training-retained-history-plus-target frequency below12:9367 items. Corrected scores are evaluated on validation only; no eta is selected using test results and no corrected test score is reported. A single training seed cannot establish a stable cross-seed effect. These curves report state occupancy, not a theorem of monotonic commitment.', '',
        'Machine-readable evidence: `summary.json`, `{Root,hr,hs}.json`, `commitment.csv`, `validation_score_sweep.csv`, and the per-run/split state and Top10 NPZ files.']
    (OUTPUT / 'report.md').write_text('\n'.join(lines) + '\n', encoding='utf-8')
    print(str(OUTPUT / 'report.md'))


if __name__ == '__main__':
    main()
