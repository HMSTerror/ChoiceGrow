"""Audit stopping rules and train the missing Root patience300 control.

The original Root has no resumable optimizer state, so this control starts from
seed100 in a separate directory. Frozen numerical modules remain unchanged.
"""
# Note: ../../docs/decisions/implemented/feature/2026-11-04-fixed-shares.md
import argparse
from copy import deepcopy
import os
from pathlib import Path
import shutil
import traceback

import experiment as p
import run_experiments as suite
from full_state_training import StatefulTrainer

OUTPUT = p.ROOT / 'runs/patience_alignment'
ROOT_ORIGINAL = p.ROOT / 'runs/step1/1A_clean_g002'
ROOT_ALIGNED = OUTPUT / 'root_p300'
CONTROLS = (('Root 1A', ROOT_ORIGINAL), ('hr .10', p.ROOT / 'runs/step2/hr_s01'),
            ('hs .10', p.ROOT / 'runs/step2/hs_s01'))


def log_rows(folder):
    path = Path(folder) / 'choice_log.jsonl'
    return [p.json.loads(line) for line in path.read_text(encoding='utf-8').splitlines()
            if line.strip()] if path.exists() else []


def replay_stopping(rows, patience):
    """Replay strict improvement and the original validation-only stop check."""
    best = None
    for row in rows:
        metrics = row.get('validation')
        if metrics is None:
            continue
        if best is None or metrics['ndcg@10'] > best['ndcg@10']:
            best = {'epoch': row['epoch'], 'ndcg@10': metrics['ndcg@10']}
        if row['epoch'] - best['epoch'] >= patience:
            return {'stop_epoch': row['epoch'], 'best_validation': best, 'stop_reached': True}
    return {'stop_epoch': None, 'best_validation': best, 'stop_reached': False}


def audit_logs():
    result = {'checked_at': p.now(), 'simulated_patience150': [],
              'best_validation_at_or_before_1670': [], 'checkpoints': []}
    for name, folder in CONTROLS:
        rows = log_rows(folder)
        assert [row['epoch'] for row in rows] == list(range(1, len(rows) + 1))
        replay = replay_stopping(rows, 150)
        result['simulated_patience150'].append({'run': name, **replay})
        selected = max((row for row in rows if row['epoch'] <= 1670 and row.get('validation')),
                       key=lambda row: row['validation']['ndcg@10'])
        result['best_validation_at_or_before_1670'].append({
            'run': name, 'epoch': selected['epoch'], 'ndcg@10': selected['validation']['ndcg@10']})
        training = p.read(folder / 'training.json')
        patience = p.read(folder / 'resolved_config.json')['training']['patience']
        actual = replay_stopping(rows, patience)
        assert actual['stop_epoch'] == training['epochs_completed']
        assert actual['best_validation']['epoch'] == training['best_epoch']
        checkpoints = []
        for path in sorted(folder.glob('choice*.pt')):
            state = p.torch.load(path, map_location='cpu', weights_only=True)
            checkpoints.append({'file': path.name, 'epoch': state['epoch']})
        result['checkpoints'].append({'run': name, 'available': checkpoints,
            'hypothetical_patience150_best_available': any(
                row['epoch'] == replay['best_validation']['epoch'] for row in checkpoints)})
    OUTPUT.mkdir(parents=True, exist_ok=True)
    p.write_json(OUTPUT / 'log_audit.json', result)
    lines = ['# Patience alignment: log audit', '',
        'Strictly larger validation NDCG@10 updates the best epoch; stopping is checked on validation epochs.', '',
        '| Run | Hypothetical patience150 stop | Selected epoch | Selected val N10 | Best val N10 through1670 | Epoch |',
        '| --- | ---: | ---: | ---: | ---: | ---: |']
    for replay, common in zip(result['simulated_patience150'], result['best_validation_at_or_before_1670']):
        best = replay['best_validation']
        lines.append(f"| {replay['run']} | {replay['stop_epoch']} | {best['epoch']} | {best['ndcg@10']:.9f} | {common['ndcg@10']:.9f} | {common['epoch']} |")
    lines += ['', 'The hypothetical hs1450/hr510 weights were overwritten by later best checkpoints; their test scores cannot be recovered from logs.',
        'Root has only EMA evaluation weights and no complete optimizer/RNG state. Its patience300 control is a separate fresh seed100 replay, never an EMA warm start.']
    (OUTPUT / 'log_audit.md').write_text('\n'.join(lines) + '\n', encoding='utf-8')
    return result


def trace_audit():
    """Report numerical agreement on the shared original/replayed log prefix."""
    reference, actual = log_rows(ROOT_ORIGINAL), log_rows(ROOT_ALIGNED)
    pairs = list(zip(reference, actual))
    mismatches = [a['epoch'] for a, b in pairs if a != b]
    loss_deltas = [abs(a['score_entropy'] - b['score_entropy']) for a, b in pairs]
    validation_pairs = [(a, b) for a, b in pairs if a.get('validation') and b.get('validation')]
    result = {'checked_at': p.now(), 'shared_epochs': len(pairs),
              'expected_original_epochs': len(reference), 'rerun_epochs': len(actual),
              'all_shared_log_records_exact': not mismatches,
              'first_differing_epoch': mismatches[0] if mismatches else None,
              'differing_epoch_count': len(mismatches),
              'max_abs_training_loss_difference': max(loss_deltas, default=0.),
              'compared_validation_epochs': len(validation_pairs),
              'max_abs_validation_ndcg10_difference': max((abs(a['validation']['ndcg@10'] - b['validation']['ndcg@10'])
                                                          for a, b in validation_pairs), default=0.),
              'interpretation': 'Same configured protocol replay; no claim of exact original-trajectory continuation when logs differ'}
    p.write_json(OUTPUT / 'trajectory_audit.json', result)
    return result


def summary():
    rows = []
    for name, folder in (('Root patience300', ROOT_ALIGNED), *CONTROLS[1:]):
        row = {'run': name, 'output': str(folder), 'patience': 300, **p.progress(folder)}
        if (folder / 'worker_status.json').exists():
            row['worker_status'] = p.read(folder / 'worker_status.json')
        if (folder / 'training.json').exists():
            row['training'] = p.read(folder / 'training.json')
        if all((folder / path).exists() for path in ('report.json', 'long_tail.json')):
            test = p.read(folder / 'report.json')['test']
            tail = p.read(folder / 'long_tail.json')
            row['test'] = {key: test[key] for key in ('hr@10', 'ndcg@10', 'streams')}
            row['tail'] = {**tail['target_groups']['tail'], 'exposure@10': tail['tail_exposure@10']}
        rows.append(row)
    complete = all(row.get('worker_status', {}).get('state') == 'complete' and 'test' in row for row in rows)
    result = {'updated_at': p.now(), 'state': 'complete' if complete else 'root_training_or_evaluation_pending',
              'training_seed': 100, 'patience': 300, 'aggregation': 'one fit per configuration, test mean of3 streams',
              'root_method': 'fresh seed100 rerun because original Root lacks full optimizer/RNG state',
              'trajectory_audit': trace_audit(), 'rows': rows}
    p.write_json(OUTPUT / 'comparison.json', result)
    lines = ['# Matched patience300 comparison', '', f"State: {result['state']}", '',
        '| Run | Stop epoch | Best val NDCG@10 | Best epoch | Test HR@10 | Test NDCG@10 | Tail HR@10 | Tail NDCG@10 | Tail exposure@10 |',
        '| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |']
    for row in rows:
        best = row['best_validation']
        metrics = ([f"{row['test'][key]:.6f}" for key in ('hr@10', 'ndcg@10')] +
                   [f"{row['tail'][key]:.6f}" for key in ('hr@10', 'ndcg@10', 'exposure@10')]) if 'test' in row else ['pending'] * 5
        values = [row['run'], row.get('training', {}).get('epochs_completed', 'pending'),
                  f"{best['ndcg@10']:.6f}" if best else 'pending', best['epoch'] if best else 'pending', *metrics]
        lines.append('| ' + ' | '.join(map(str, values)) + ' |')
    lines += ['', 'All three configurations use patience300, strict validation improvement, the same validation interval and streams.',
              'Root is a fresh replay. trajectory_audit.json records whether the first1670 epochs reproduce the original Root log exactly.']
    (OUTPUT / 'comparison.md').write_text('\n'.join(lines) + '\n', encoding='utf-8')
    return result


def prepare():
    if (ROOT_ALIGNED / 'job.json').exists():
        raise FileExistsError('Alignment job is already prepared')
    manifest = p.read(suite.OUTPUT / 'manifest.json')
    assert suite.code_hashes() == manifest['frozen_code_sha256']
    ROOT_ALIGNED.mkdir(parents=True, exist_ok=False)
    config = deepcopy(p.read(ROOT_ORIGINAL / 'resolved_config.json'))
    config['training']['patience'] = 300
    assert config['training']['epochs'] == 0
    assert config['destination']['neighbor_mix'] == .02 and config['neighbor_statistics'] == 'nn_clean'
    assert config['background_statistics'] == 'bn' and config['diffusion']['fixed_shares'] == [1.]
    assert not config['diffusion']['dynamic_rate_gate']
    for name in ('config.json', 'resolved_config.json'):
        p.write_json(ROOT_ALIGNED / name, config)
    for name in ('data_source.json', 'structure.pt', 'artifacts.json'):
        shutil.copy2(ROOT_ORIGINAL / name, ROOT_ALIGNED / name)
    artifact_meta = p.read(ROOT_ALIGNED / 'artifacts.json')
    artifact_meta['training_overrides'] = {'patience': 300}
    p.write_json(ROOT_ALIGNED / 'artifacts.json', artifact_meta)
    job = {'id': 'root_patience300', 'output': str(ROOT_ALIGNED), 'state': 'prepared',
           'fresh_seed100_replay': True, 'source': str(ROOT_ORIGINAL), 'only_config_change': {'training.patience': [150, 300]},
           'structure_sha256': p.sha256(ROOT_ALIGNED / 'structure.pt'),
           'source_structure_sha256': p.sha256(ROOT_ORIGINAL / 'structure.pt'),
           'original_training_state_exists': (ROOT_ORIGINAL / 'training_state.pt').exists(),
           'frozen_code_sha256': manifest['frozen_code_sha256']}
    assert job['structure_sha256'] == job['source_structure_sha256']
    p.write_json(ROOT_ALIGNED / 'job.json', job)
    audit_logs()
    summary()
    return job


class AuditedCheckpoints:
    """Delegate the original writer and compare retained original epoch weights."""
    def __init__(self, original):
        self.original = original
        self.best_path = original.best_path

    def save(self, filename, model, process, epoch, metrics, ema_updates):
        self.original.save(filename, model, process, epoch, metrics, ema_updates)
        reference_name = {1520: 'choice_best.pt', 1670: 'choice_newst.pt'}.get(epoch)
        if filename != 'choice_newst.pt' or reference_name is None:
            return
        reference = p.torch.load(ROOT_ORIGINAL / reference_name, map_location='cpu', weights_only=True)
        current = model.state_dict()
        equal = all(p.torch.equal(value.detach().cpu(), reference['model'][key]) for key, value in current.items())
        path = OUTPUT / 'retained_weight_audit.json'
        report = p.read(path) if path.exists() else {'checks': []}
        report['checks'].append({'epoch': epoch, 'reference': reference_name, 'all_model_tensors_exact': equal})
        p.write_json(path, report)


def worker(resume=False):
    import psutil
    for process in psutil.process_iter(['pid', 'cmdline']):
        command = process.info['cmdline'] or []
        if process.pid != os.getpid() and '--worker' in command and any(
                str(Path(arg)).replace('\\', '/') == str(Path(__file__).resolve()).replace('\\', '/') for arg in command):
            raise RuntimeError('An alignment worker is already alive')
    job = p.read(ROOT_ALIGNED / 'job.json')
    assert suite.code_hashes() == job['frozen_code_sha256']
    config = p.load_config(ROOT_ALIGNED / 'resolved_config.json')
    p.torch.set_num_threads(config['threads'])
    if config['tf32']:
        p.torch.set_float32_matmul_precision('high'); p.torch.backends.cudnn.allow_tf32 = True
    status = {'pid': os.getpid(), 'updated_at': p.now(), 'state': 'training', 'fresh_rerun': True,
              'patience': 300, 'resumed_from_full_state': resume}
    p.write_json(ROOT_ALIGNED / 'worker_status.json', status)
    try:
        trainer = StatefulTrainer(ROOT_ALIGNED, config, p.torch.device('cuda:0'))
        trainer.checkpoints = AuditedCheckpoints(trainer.checkpoints)
        trainer.fit(resume=resume)
        p.write_json(ROOT_ALIGNED / 'worker_status.json', {**status, 'state': 'evaluating', 'updated_at': p.now()})
        trace_audit()
        selected = p.torch.load(ROOT_ALIGNED / 'choice_best.pt', map_location='cpu', weights_only=True)
        original = p.torch.load(ROOT_ORIGINAL / 'choice_best.pt', map_location='cpu', weights_only=True)
        identical = (selected['epoch'] == original['epoch'] and selected['validation'] == original['validation'] and
                     all(p.torch.equal(value, original['model'][key]) for key, value in selected['model'].items()))
        if identical:
            for name in ('report.json', 'recommendations.npz', 'long_tail.json', 'validation_report.json',
                         'validation_recommendations.npz', 'long_tail_val.json'):
                shutil.copy2(ROOT_ORIGINAL / name, ROOT_ALIGNED / name)
        p.write_json(ROOT_ALIGNED / 'evaluation_reuse.json', {
            'reused': identical, 'source': str(ROOT_ORIGINAL) if identical else None,
            'reason': 'Same selected epoch, validation and bitwise-equal EMA model; same structure/data/evaluation settings'
                      if identical else 'Selected model changed; run full3-stream test and tail evaluation',
            'selected_epoch': selected['epoch']})
        p.finish_evaluation(ROOT_ALIGNED)
        p.write_json(ROOT_ALIGNED / 'worker_status.json', {**status, 'state': 'complete', 'updated_at': p.now()})
        result = summary()
        print(p.json.dumps({'event': 'alignment_complete', 'summary': result}), flush=True)
    except BaseException:
        p.write_json(ROOT_ALIGNED / 'worker_status.json', {**status, 'state': 'failed', 'updated_at': p.now(),
                     'traceback': traceback.format_exc()})
        summary()
        raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--prepare', action='store_true')
    parser.add_argument('--worker', action='store_true')
    parser.add_argument('--resume', action='store_true', help='Restore the complete epoch-boundary training state')
    parser.add_argument('--audit-only', action='store_true')
    args = parser.parse_args()
    if args.resume and not args.worker:
        parser.error('--resume requires --worker')
    if args.prepare:
        print(p.json.dumps(prepare()), flush=True)
    elif args.worker:
        worker(resume=args.resume)
    elif args.audit_only:
        print(p.json.dumps(audit_logs()), flush=True)
    else:
        print(p.json.dumps(summary()), flush=True)
