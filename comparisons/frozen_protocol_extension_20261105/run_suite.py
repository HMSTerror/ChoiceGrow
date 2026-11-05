"""Run the three approved frozen-protocol batches with two continuous GPU slots.

Numerical modules are byte-identical snapshots of the reviewed Toys suite.
This entry point owns data audits, artifact preparation, scheduling and reports.
See ../../docs/decisions/implemented/feature/2026-11-04-fixed-shares.md.
"""
import argparse
from collections import Counter
from copy import deepcopy
import json
import math
import os
from pathlib import Path
import statistics
import subprocess
import sys
import time
import traceback

import psutil
import experiment as p
import full_state_training as fst
from train import append_log

ROOT = Path(__file__).resolve().parent
OUTPUT = ROOT / 'runs'
OLD = p.PROJECT / 'comparisons/fixed_shares_clean_toys_20261104_2045'
BASELINE = OLD / 'runs/patience_alignment/root_p300'
SOURCES = p.PROJECT / 'runs/raw_seed100_lr0006_patience150_20261103'
REUSED = {'h0': BASELINE, 'hr': OLD / 'runs/step2/hr_s01', 'hs': OLD / 'runs/step2/hs_s01'}
K = {'Toys': 12, 'Beauty': 12, 'Sports': 16}
REQUIRED = ('training.json', 'report.json', 'validation_report.json',
            'long_tail.json', 'long_tail_val.json', 'efficiency_summary.json')


def process_record(pid=None):
    process = psutil.Process(pid or os.getpid())
    return {'pid': process.pid, 'created': process.create_time(), 'command': process.cmdline()}


def alive(record):
    """Reject recycled PIDs and changed commands, rather than trusting a PID alone."""
    try:
        process = psutil.Process(record['pid'])
        return abs(process.create_time() - record['created']) < .01 and process.cmdline() == record['command']
    except (psutil.Error, TypeError, KeyError):
        return False


def matching_workers(spec):
    exact = os.path.normcase(os.path.abspath(spec))
    found = []
    for process in psutil.process_iter(['pid', 'cmdline']):
        args = process.info['cmdline'] or []
        if '--worker-job' in args:
            index = args.index('--worker-job')
            if index + 1 < len(args) and os.path.normcase(os.path.abspath(args[index + 1])) == exact:
                found.append(process_record(process.pid))
    return found


def code_hashes():
    return {str(path.relative_to(ROOT)): p.sha256(path)
            for path in sorted(ROOT.rglob('*.py')) if 'runs' not in path.relative_to(ROOT).parts}


def background_counts(sequence, targets, item_count, mode):
    """bn counts every retained successor once; bo counts final targets only."""
    if mode == 'bo':
        return p.np.bincount(targets, minlength=item_count)
    if mode != 'bn':
        raise ValueError(mode)
    counts = p.np.zeros(item_count, dtype=p.np.int64)
    for history, target in zip(sequence, targets):
        chain = [int(x) for x in history if 0 <= x < item_count] + [int(target)]
        p.np.add.at(counts, chain[1:], 1)
    return counts


def dataset_artifacts(dataset):
    """Reconstruct float32 training artifacts and verify the reviewed Step 0 files."""
    source = SOURCES / dataset
    train = p.load_split(source, 'train')
    n = p.metadata(source)['item_count']
    counts = background_counts(train['seq'].numpy(), train['target'].numpy(), n, 'bn')
    base = (p.torch.from_numpy(counts).float() + 1).sqrt()
    base /= base.sum()
    partition_path = p.STEP0 / 'partitions' / dataset / 'partitions.npz'
    with p.np.load(partition_path) as saved:
        p.np.testing.assert_array_equal(counts, saved['bn_counts'])
        p.np.testing.assert_allclose(base.numpy(), saved['base'], rtol=3e-7, atol=1e-10)
        labels = {mode: p.torch.from_numpy(saved[mode].copy()).long() for mode in ('hs', 'hr')}
    for label in labels.values():
        assert label.shape == (n,) and len(label.unique()) == K[dataset]
        assert int(label.min()) == 0 and int(label.max()) == K[dataset] - 1
    clean = p.count_transitions(train['seq'].numpy(), train['target'].numpy(), n, 'nn_clean', 3)
    rows, cols, values = [], [], []
    for source_id, row in clean.items():
        for target_id, weight in row.items():
            rows.append(source_id); cols.append(target_id); values.append(weight)
    graph = p.sparse.csr_matrix((values, (rows, cols)), shape=(n, n))
    delta = graph - p.sparse.load_npz(partition_path.parent / 'clean_directed_graph.npz')
    assert p.np.abs(delta.data).max(initial=0) == 0
    ids, weights = p.tensor_table(clean, n)
    table = p.sparse.csr_matrix((weights.numpy().ravel(),
        (p.np.repeat(p.np.arange(n), ids.shape[1]), ids.numpy().ravel())), shape=(n, n))
    table.eliminate_zeros()
    reviewed_table = p.sparse.load_npz(partition_path.parent / 'clean_neighbor_table.npz')
    assert (table.astype(bool) != reviewed_table.astype(bool)).nnz == 0
    delta = table - reviewed_table
    assert p.np.abs(delta.data).max(initial=0) < 2e-7
    bo_counts = background_counts(train['seq'].numpy(), train['target'].numpy(), n, 'bo')
    bo = (p.torch.from_numpy(bo_counts).float() + 1).sqrt(); bo /= bo.sum()
    tail, cutoff = p.tail_mask(SOURCES / dataset)
    audit = {'dataset': dataset, 'catalog_items': n, 'K': K[dataset],
        'split_rows': {s: len(p.load_split(SOURCES / dataset, s)['target']) for s in ('train', 'val', 'test')},
        'prepared_source': str(p.prepared_root(SOURCES / dataset).resolve()),
        'split_sha256': p.split_hashes(SOURCES / dataset),
        'partitions_sha256': p.sha256(partition_path), 'partition_seed': 100,
        'bn_counts_match_step0': True, 'clean_graph_matches_step0': True,
        'clean_table_support_matches_step0': True, 'clean_table_max_float_error': float(p.np.abs(delta.data).max(initial=0)),
        'tail_rule': f'training retained-history-plus-target frequency < {cutoff}',
        'tail_catalog_items': int(tail.sum())}
    return {'base': base, 'bo': bo, 'neighbor_ids': ids, 'neighbor_weights': weights,
            'labels': labels}, audit


def job_spec(batch, dataset, seed, partition, background='bn', neighbor='nn_clean'):
    identity = f'batch{batch}/{dataset}/seed_{seed}/{background}_{neighbor}_{partition}'
    folder = OUTPUT / identity
    return {'id': identity, 'batch': batch, 'dataset': dataset, 'seed': seed, 'partition': partition,
        'background': background, 'neighbor': neighbor, 'gamma': 0. if neighbor == 'n0' else .02,
        'community_share': 0. if partition == 'h0' else .1, 'K': K[dataset],
        'output': str(folder), 'spec': str(folder / 'job.json'), 'state': 'queued', 'attempts': 0}


def validate_protocol(config, job):
    reference = p.read(BASELINE / 'resolved_config.json')
    assert config['training'] == reference['training']
    assert config['training']['patience'] == 300 and config['training']['epochs'] == 0
    assert config['training']['validation_streams'] == 1 and config['training']['test_streams'] == 3
    assert config['destination'] == {**reference['destination'], 'neighbor_mix': job['gamma']}
    assert config['diffusion']['dynamic_rate_gate'] is False
    shares = [1.] if job['partition'] == 'h0' else [.1, .9]
    assert config['diffusion']['fixed_shares'] == shares
    expected = deepcopy(reference['diffusion'])
    expected.update(fixed_shares=shares, level_rates=[expected['total_budget'] * s for s in shares])
    assert config['diffusion'] == expected
    allowed = {'dataset', 'seed', 'experiment', 'partition', 'neighbor_statistics', 'background_statistics',
               'diffusion', 'destination'}
    assert {k: v for k, v in config.items() if k not in allowed} == {
        k: v for k, v in reference.items() if k not in allowed}


def prepare():
    """Prepare all authorized jobs once; later batches remain behind the batch barrier."""
    if (OUTPUT / 'manifest.json').exists():
        raise FileExistsError('Suite already prepared')
    assert p.read(p.STEP0 / 'diagnostics/verification.json')['status'] == 'passed'
    artifacts, audits = {}, {}
    for dataset in ('Toys', 'Beauty', 'Sports'):
        artifacts[dataset], audits[dataset] = dataset_artifacts(dataset)
    jobs = []
    # Interleave datasets and keep both slots busy without a method-pair barrier.
    for partition in ('h0', 'hr', 'hs'):
        for dataset in ('Beauty', 'Sports'):
            jobs.append(job_spec(1, dataset, 100, partition))
    jobs += [job_spec(2, 'Toys', 100, 'h0', neighbor='n0'),
             job_spec(2, 'Toys', 100, 'h0', background='bo')]
    for partition in ('h0', 'hr', 'hs'):
        for seed in (101, 102):
            jobs.append(job_spec(3, 'Toys', seed, partition))
    for job in jobs:
        directory = Path(job['output']); directory.mkdir(parents=True, exist_ok=True)
        values = artifacts[job['dataset']]
        config = deepcopy(p.read(BASELINE / 'resolved_config.json'))
        config.update(dataset=job['dataset'], seed=job['seed'], experiment=job['id'],
            partition=job['partition'], background_statistics=job['background'], neighbor_statistics=job['neighbor'])
        config['destination']['neighbor_mix'] = job['gamma']
        shares = [1.] if job['partition'] == 'h0' else [.1, .9]
        config['diffusion'].update(fixed_shares=shares,
            level_rates=[config['diffusion']['total_budget'] * s for s in shares])
        validate_protocol(config, job)
        n = len(values['base'])
        blocks = p.torch.zeros((1, n), dtype=p.torch.long) if job['partition'] == 'h0' else p.torch.stack(
            (values['labels'][job['partition']], p.torch.zeros(n, dtype=p.torch.long)))
        structure = {'base': values['base'] if job['background'] == 'bn' else values['bo'],
            'blocks': blocks, 'neighbor_ids': values['neighbor_ids'], 'neighbor_weights': values['neighbor_weights']}
        p.torch.save(structure, directory / 'structure.pt')
        p.write_json(directory / 'data_source.json', {'path': audits[job['dataset']]['prepared_source']})
        for name in ('config.json', 'resolved_config.json'):
            p.write_json(directory / name, config)
        p.write_json(directory / 'artifacts.json', {**audits[job['dataset']], 'statistics_split': 'train',
            'background': job['background'], 'neighbor_statistics': job['neighbor'],
            'same_neighbor_table_all_splits': True, 'no_final_target_in_clean_edges': True,
            'partition': job['partition'], 'fixed_shares': shares, 'eta': 0., 'dynamic_rate_gate': False,
            'partition_source': str(p.STEP0 / 'partitions' / job['dataset'] / 'partitions.npz')})
        job['inputs_sha256'] = {name: p.sha256(directory / name) for name in
            ('resolved_config.json', 'structure.pt', 'data_source.json')}
        p.write_json(job['spec'], job)
    references = []
    for partition, folder in REUSED.items():
        config = p.read(folder / 'resolved_config.json')
        ref = job_spec(3, 'Toys', 100, partition)
        validate_protocol(config, ref)
        assert all((folder / name).exists() for name in REQUIRED[:-1])
        references.append({**ref, 'output': str(folder), 'state': 'complete', 'reused_training': True,
            'checkpoint_sha256': p.sha256(folder / 'choice_best.pt')})
    p.write_json(OUTPUT / 'input_audit.json', {'checked_at': p.now(), 'datasets': audits,
        'partition_rebuilds': 0, 'new_fits': 14, 'reused_seed100_fits': 3})
    p.write_json(OUTPUT / 'execution_control.json', {'hold_new_jobs': False, 'max_workers': 2,
        'strict_batch_barrier': True, 'automatic_next_batch': True, 'automatic_version_selection': False,
        'eta': 0., 'patience': 300, 'github_upload': False})
    p.write_json(OUTPUT / 'manifest.json', {'created_at': p.now(), 'jobs': jobs, 'references': references,
        'frozen_code_sha256': code_hashes(), 'batch_reports': {}, 'state': 'prepared'})
    summarize()


def append_timing(directory, record):
    with (Path(directory) / 'efficiency.jsonl').open('a', encoding='utf-8') as stream:
        stream.write(json.dumps({'at': p.now(), **record}) + '\n')


class TimedEvaluator(p.Evaluator):
    """Measure the unchanged evaluator; timers never draw random numbers."""
    def __init__(self, config, device, output):
        super().__init__(config, device)
        self.output = Path(output)

    def evaluate(self, model, process, values, split, streams, predictions_path=None):
        p.torch.cuda.synchronize(self.device)
        started = time.perf_counter()
        result = super().evaluate(model, process, values, split, streams, predictions_path)
        p.torch.cuda.synchronize(self.device)
        append_timing(self.output, {'stage': 'evaluation', 'split': split, 'streams': streams,
            'seconds': time.perf_counter() - started, 'examples': len(values['target']),
            'context': 'training worker; may share GPU with another fit'})
        return result


class TimedTrainer(fst.StatefulTrainer):
    """Add compute and full-epoch wall timers without replacing training arithmetic."""
    def __init__(self, output, config, device):
        super().__init__(output, config, device)
        self.evaluator = TimedEvaluator(config, device, output)
        self.timing_epoch = 0

    def train_epoch(self):
        p.torch.cuda.synchronize(self.device)
        p.torch.cuda.reset_peak_memory_stats(self.device)
        self.epoch_started = time.perf_counter()
        result = super().train_epoch()
        p.torch.cuda.synchronize(self.device)
        self.timing_epoch += 1
        self.epoch_compute = {'stage': 'epoch', 'epoch': self.timing_epoch,
            'training_seconds': time.perf_counter() - self.epoch_started,
            'peak_allocated_bytes': p.torch.cuda.max_memory_allocated(self.device),
            'peak_reserved_bytes': p.torch.cuda.max_memory_reserved(self.device)}
        return result


def timing_summary(directory):
    path = Path(directory) / 'efficiency.jsonl'
    rows = [json.loads(line) for line in path.read_text().splitlines()] if path.exists() else []
    epochs = {row['epoch']: row for row in rows if row['stage'] == 'epoch'}
    selected = list(epochs.values())
    return {'epochs_measured': len(selected), 'timing_scope': 'same GPU, up to two concurrent training workers',
        'training_sec_per_epoch_mean': statistics.mean(r['training_seconds'] for r in selected) if selected else None,
        'full_epoch_sec_mean': statistics.mean(r['wall_seconds'] for r in selected) if selected else None,
        'full_epoch_includes': 'training, scheduled validation, EMA checkpoints, complete AdamW/RNG snapshot',
        'peak_training_allocated_bytes': max((r['peak_allocated_bytes'] for r in selected), default=None),
        'peak_training_reserved_bytes': max((r['peak_reserved_bytes'] for r in selected), default=None),
        'evaluations': [row for row in rows if row['stage'] == 'evaluation']}


def worker(spec, resume=False):
    """Train once, select on validation, then evaluate the selected EMA checkpoint."""
    job = p.read(spec); directory = Path(job['output'])
    config = p.load_config(directory / 'resolved_config.json'); validate_protocol(config, job)
    manifest = p.read(OUTPUT / 'manifest.json')
    assert manifest['frozen_code_sha256'] == code_hashes(), 'Frozen source changed'
    assert job['inputs_sha256'] == {name: p.sha256(directory / name) for name in job['inputs_sha256']}
    record = process_record()
    def status(state, **extra):
        p.write_json(directory / 'worker_status.json', {'state': state, 'at': p.now(),
            'job': job['id'], 'process': record, **extra})
    try:
        p.torch.set_num_threads(config['threads'])
        p.torch.backends.cuda.matmul.allow_tf32 = config['tf32']
        p.torch.backends.cudnn.allow_tf32 = config['tf32']
        device = p.torch.device(config['device'])
        p.write_json(directory / 'hardware.json', {'gpu': p.torch.cuda.get_device_name(device),
            'torch': str(p.torch.__version__), 'cuda': p.torch.version.cuda, 'tf32': config['tf32'],
            'process': record, 'seed': config['seed']})
        if not (directory / 'training.json').exists():
            status('training', resume=resume)
            trainer = TimedTrainer(directory, config, device)
            if resume:
                state = fst.TrainingStateStore(directory).read(config)
                trainer.timing_epoch = state['epoch']
            original_save = fst.save_full_state
            def timed_save(*args, **kwargs):
                original_save(*args, **kwargs)
                append_timing(directory, {**trainer.epoch_compute,
                    'wall_seconds': time.perf_counter() - trainer.epoch_started})
            fst.save_full_state = timed_save
            trainer.fit(resume=resume)
            fst.save_full_state = original_save
            del trainer
            p.torch.cuda.empty_cache()
        status('evaluating')
        original_evaluator = p.Evaluator
        p.Evaluator = lambda cfg, dev: TimedEvaluator(cfg, dev, directory)
        try:
            p.finish_evaluation(directory)
        finally:
            p.Evaluator = original_evaluator
        p.write_json(directory / 'efficiency_summary.json', timing_summary(directory))
        status('complete', progress=p.progress(directory))
    except BaseException:
        status('failed', traceback=traceback.format_exc())
        raise


def result_row(job):
    directory = Path(job['output'])
    row = {key: job[key] for key in ('id', 'dataset', 'seed', 'partition', 'background', 'neighbor', 'gamma')}
    row.update(state=job['state'], output=str(directory), reused_training=job.get('reused_training', False))
    row.update(p.progress(directory))
    if all((directory / name).exists() for name in REQUIRED[:-1]):
        training = p.read(directory / 'training.json')
        assert p.read(directory / 'resolved_config.json')['training']['patience'] == 300
        assert training.get('configured_patience', 300) == 300
        assert training['epochs_completed'] - training['best_epoch'] >= 300
        row.update(stop_epoch=training['epochs_completed'], best_epoch=training['best_epoch'],
            validation=p.read(directory / 'validation_report.json')['validation'],
            test=p.read(directory / 'report.json')['test'],
            validation_tail=p.read(directory / 'long_tail_val.json'),
            test_tail=p.read(directory / 'long_tail.json'))
    return row


def mean_sd(rows):
    """Each input is one training seed, already averaged over sampling streams."""
    assert sorted(row['seed'] for row in rows) == [100, 101, 102]
    values = {}
    for split in ('validation', 'test'):
        for metric in p.METRICS:
            values[f'{split}.{metric}'] = [row[split][metric] for row in rows]
            values[f'{split}.tail.{metric}'] = [row[split + '_tail']['target_groups']['tail'][metric] for row in rows]
        values[f'{split}.tail_exposure@10'] = [row[split + '_tail']['tail_exposure@10'] for row in rows]
    return {key: {'mean': statistics.mean(value), 'sd': statistics.stdev(value), 'n_training_seeds': 3,
                  'values': value} for key, value in values.items()}


def markdown_table(rows):
    lines = ['| Dataset / seed | Method | Stop | Best epoch | Val N10 | Val tail N10 | Test HR10 | Test N10 | Tail HR10 | Tail N10 | Tail exposure10 |',
             '|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|']
    for row in rows:
        if 'test' not in row:
            lines.append(f"| {row['dataset']} / {row['seed']} | {row['background']} + {row['neighbor']} + {row['partition']} | {row['state']} | | | | | | | | |")
            continue
        test, tail = row['test'], row['test_tail']
        label = f"{row['background']} + {row['neighbor']} + {row['partition']}"
        lines.append(f"| {row['dataset']} / {row['seed']} | {label} | {row['stop_epoch']} | {row['best_epoch']} | "
            f"{row['validation']['ndcg@10']:.6f} | {row['validation_tail']['target_groups']['tail']['ndcg@10']:.6f} | "
            f"{test['hr@10']:.6f} | {test['ndcg@10']:.6f} | {tail['target_groups']['tail']['hr@10']:.6f} | "
            f"{tail['target_groups']['tail']['ndcg@10']:.6f} | {tail['tail_exposure@10']:.6f} |")
    return '\n'.join(lines) + '\n'


def summarize():
    manifest = p.read(OUTPUT / 'manifest.json')
    jobs = manifest['jobs']; references = manifest['references']
    rows = [result_row(job) for job in jobs]
    refs = [result_row(job) for job in references]
    completed = []
    for batch in (1, 2, 3):
        selected = [row for row in rows if next(j['batch'] for j in jobs if j['id'] == row['id']) == batch]
        done = all(row['state'] == 'complete' and 'test' in row for row in selected)
        comparison = selected + ([row for row in refs if row['partition'] == 'h0'] if batch == 2 else refs if batch == 3 else [])
        report = {'batch': batch, 'complete': done, 'updated_at': p.now(), 'rows': comparison,
            'eta': 0., 'patience': 300, 'test_sampling_streams': 3, 'selection': 'best validation NDCG@10'}
        if done:
            completed.append(batch)
        if batch == 3 and done:
            report['three_seed_mean_sample_sd'] = {mode: mean_sd([r for r in comparison if r['partition'] == mode])
                                                   for mode in ('h0', 'hr', 'hs')}
        p.write_json(OUTPUT / f'batch{batch}_summary.json', report)
        text = f'# Batch {batch}: frozen protocol\n\n' + markdown_table(comparison)
        text += '\nValidation: one sampling stream. Test: mean over three sampling streams per trained model. Tail metrics use the full-catalog ranking and training-only tail definition.\n'
        if 'three_seed_mean_sample_sd' in report:
            text += '\n## Toys three-training-seed mean ± sample SD\n\n| Method | Test N10 | Tail N10 | Test HR10 |\n|---|---:|---:|---:|\n'
            for mode, stats in report['three_seed_mean_sample_sd'].items():
                cells = [f"{stats[k]['mean']:.6f} ± {stats[k]['sd']:.6f}" for k in ('test.ndcg@10', 'test.tail.ndcg@10', 'test.hr@10')]
                text += f"| {mode} | " + ' | '.join(cells) + ' |\n'
        (OUTPUT / f'batch{batch}_summary.md').write_text(text, encoding='utf-8')
    p.write_json(OUTPUT / 'status.json', {'at': p.now(), 'state': manifest['state'],
        'completed_fits': sum(j['state'] == 'complete' for j in jobs), 'total_new_fits': len(jobs),
        'completed_batches': completed, 'running': [row for row in rows if row['state'] == 'running'],
        'failed': [row for row in rows if row['state'] == 'failed']})
    return rows, completed


def eligible_jobs(jobs):
    """Admit any queued job in the first unfinished batch, regardless of pairs."""
    unfinished = [job['batch'] for job in jobs if job['state'] != 'complete']
    batch = min(unfinished, default=None)
    return [job for job in jobs if job['batch'] == batch and job['state'] == 'queued']


def launch(job):
    directory = Path(job['output'])
    with (directory / 'worker.log').open('ab') as stdout, (directory / 'worker.err.log').open('ab') as stderr:
        child = subprocess.Popen([str(p.PYTHON), '-B', '-u', str(ROOT / 'run_suite.py'), '--worker-job', job['spec']],
            cwd=p.PROJECT, stdout=stdout, stderr=stderr, creationflags=subprocess.CREATE_NO_WINDOW)
    job.update(state='running', attempts=job['attempts'] + 1, process=process_record(child.pid), started_at=p.now())
    p.write_json(directory / 'launcher.json', job['process'])


def coordinator():
    lock = OUTPUT / 'coordinator.json'
    if lock.exists() and alive(p.read(lock)):
        raise RuntimeError('Coordinator is already alive')
    p.write_json(lock, process_record())
    manifest = p.read(OUTPUT / 'manifest.json')
    assert manifest['frozen_code_sha256'] == code_hashes(), 'Frozen source changed'
    try:
        while True:
            control = p.read(OUTPUT / 'execution_control.json')
            assert control['max_workers'] == 2 and control['strict_batch_barrier'] and control['eta'] == 0.
            for job in manifest['jobs']:
                if job['state'] in ('queued', 'running'):
                    matches = matching_workers(job['spec'])
                    if len(matches) > 1:
                        raise RuntimeError(f"Duplicate workers: {job['id']}")
                    if matches:
                        job.update(state='running', process=matches[0])
                        continue
                    if job['state'] == 'running':
                        status = p.read(Path(job['output']) / 'worker_status.json') if (Path(job['output']) / 'worker_status.json').exists() else {}
                        if status.get('state') == 'complete' and all((Path(job['output']) / name).exists() for name in REQUIRED):
                            job.update(state='complete', completed_at=p.now())
                        else:
                            job.update(state='failed', failure=status or {'reason': 'Worker exited without completion; inspect actual stderr and saved state'})
            running = sum(job['state'] == 'running' for job in manifest['jobs'])
            failed = any(job['state'] == 'failed' for job in manifest['jobs'])
            if not control['hold_new_jobs'] and not failed:
                for job in eligible_jobs(manifest['jobs'])[:max(0, 2 - running)]:
                    launch(job)
            manifest['state'] = 'needs_attention' if failed else 'running'
            manifest['updated_at'] = p.now()
            p.write_json(OUTPUT / 'manifest.json', manifest)
            _, completed = summarize()
            for batch in completed:
                if str(batch) not in manifest['batch_reports']:
                    manifest['batch_reports'][str(batch)] = {'completed_at': p.now(), 'notification_pending': True,
                        'report': str(OUTPUT / f'batch{batch}_summary.md')}
                    print(json.dumps({'event': 'batch_complete', 'batch': batch, 'at': p.now()}), flush=True)
            p.write_json(OUTPUT / 'manifest.json', manifest)
            if len(completed) == 3:
                manifest['state'] = 'benchmarking'
                p.write_json(OUTPUT / 'manifest.json', manifest)
                subprocess.run([str(p.PYTHON), '-B', '-u', str(ROOT / 'run_suite.py'), '--benchmark'],
                               cwd=p.PROJECT, check=True)
                manifest['state'] = 'complete'
                p.write_json(OUTPUT / 'manifest.json', manifest)
                summarize()
                return
            if failed and running == 0:
                return
            time.sleep(20)
    except BaseException:
        p.write_json(OUTPUT / 'coordinator_failure.json', {'at': p.now(), 'traceback': traceback.format_exc()})
        raise


def benchmark():
    """Isolated evaluation-only benchmark on the retained matched Toys seed100 fits."""
    if any(matching_workers(job['spec']) for job in p.read(OUTPUT / 'manifest.json')['jobs']):
        raise RuntimeError('Benchmark requires idle training workers')
    results = []
    for mode in ('h0', 'hs'):
        folder = REUSED[mode]; config = p.load_config(folder / 'resolved_config.json')
        p.torch.set_num_threads(config['threads'])
        p.torch.backends.cuda.matmul.allow_tf32 = config['tf32']
        p.torch.backends.cudnn.allow_tf32 = config['tf32']
        device = p.torch.device(config['device'])
        process = p.ChoiceDiffusion.from_directory(folder, config, device)
        state = p.CheckpointStore(folder, config).load_best(process, device)
        model = p.make_denoiser(folder, config, process).to(device); model.load_state_dict(state['model'])
        evaluator = p.Evaluator(config, device)
        for split, streams in (('val', 1), ('test', 3)):
            values = p.load_split(folder, split)
            evaluator.evaluate(model, process, values, split, streams)
            for repeat in range(3):
                p.torch.cuda.reset_peak_memory_stats(device); p.torch.cuda.synchronize(device)
                started = time.perf_counter()
                metrics = evaluator.evaluate(model, process, values, split, streams)
                p.torch.cuda.synchronize(device)
                expected = p.read(folder / ('report.json' if split == 'test' else 'validation_report.json'))['test' if split == 'test' else 'validation']
                assert abs(metrics['ndcg@10'] - expected['ndcg@10']) < 1e-7
                results.append({'method': mode, 'split': split, 'streams': streams, 'repeat': repeat,
                    'seconds': time.perf_counter() - started, 'peak_allocated_bytes': p.torch.cuda.max_memory_allocated(device),
                    'peak_reserved_bytes': p.torch.cuda.max_memory_reserved(device), 'metrics': metrics})
        del model, process, evaluator
        p.torch.cuda.empty_cache()
    manifest = p.read(OUTPUT / 'manifest.json')
    training = [{'id': job['id'], **p.read(Path(job['output']) / 'efficiency_summary.json')}
                for job in manifest['jobs'] if job['batch'] == 3 and job['partition'] in ('h0', 'hs')]
    p.write_json(OUTPUT / 'efficiency_comparison.json', {'at': p.now(), 'gpu': p.torch.cuda.get_device_name(),
        'evaluation_context': 'one process; no suite training active; one full-split warmup then three repeats; checkpoint loading excluded',
        'training_context': 'two concurrent GPU fits; per-process CUDA memory; full epoch includes validation and state writes',
        'isolated_evaluation': results, 'training': training, 'extra_training_epochs': 0})
    lines = ['# End-to-end efficiency\n', '| Method | Split | Streams | Mean seconds | Peak allocated MiB |', '|---|---|---:|---:|---:|']
    for mode in ('h0', 'hs'):
        for split in ('val', 'test'):
            group = [r for r in results if r['method'] == mode and r['split'] == split]
            lines.append(f"| {mode} | {split} | {group[0]['streams']} | {statistics.mean(r['seconds'] for r in group):.3f} | {max(r['peak_allocated_bytes'] for r in group)/2**20:.1f} |")
    lines += ['', '| Training run | Compute sec/epoch | Full sec/epoch | Peak allocated MiB |', '|---|---:|---:|---:|']
    for row in training:
        lines.append(f"| {row['id']} | {row['training_sec_per_epoch_mean']:.3f} | {row['full_epoch_sec_mean']:.3f} | {row['peak_training_allocated_bytes']/2**20:.1f} |")
    lines.append('\nTraining timings include GPU contention from two simultaneous fits. Full epoch includes scheduled validation, lightweight checkpoints, and full AdamW/RNG state writes. Evaluation timings use an otherwise idle suite, exclude loading, and retain eta0. These measurements describe end-to-end cost; they do not isolate or prove the theoretical coefficient-complexity bound.')
    (OUTPUT / 'efficiency_comparison.md').write_text('\n'.join(lines), encoding='utf-8')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument('--prepare', action='store_true')
    mode.add_argument('--coordinate', action='store_true')
    mode.add_argument('--worker-job', type=Path)
    mode.add_argument('--summary-only', action='store_true')
    mode.add_argument('--benchmark', action='store_true')
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args()
    if args.prepare: prepare()
    elif args.coordinate: coordinator()
    elif args.worker_job: worker(args.worker_job, args.resume)
    elif args.benchmark: benchmark()
    else: summarize()
