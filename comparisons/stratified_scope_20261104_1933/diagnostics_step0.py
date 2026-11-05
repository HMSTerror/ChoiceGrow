"""Read-only CPU diagnostics for hierarchy, self-inclusion, and stopping claims.

Run this before any new GPU experiment. All writes stay under this comparison.
"""
# Note: notes.md describes the mathematical corrections and Step 1 review gate.
import os
os.environ['CUDA_VISIBLE_DEVICES'] = ''
os.environ['OMP_NUM_THREADS'] = '2'
os.environ['MKL_NUM_THREADS'] = '2'
os.environ['OPENBLAS_NUM_THREADS'] = '2'

from collections import Counter
from datetime import datetime
from pathlib import Path
import argparse
import hashlib
import json
import math
import time
import sys
import faulthandler

import torch  # Import before NumPy to share the existing OpenMP runtime.
import numpy as np
from scipy import sparse
from scipy.stats import spearmanr
from threadpoolctl import threadpool_limits

from neighbors import count_transitions, build_table, history_weights, remove_own_final_edges
from scope import directed_graph, build_stratified_partitions, infomap_labels

ROOT = Path(__file__).resolve().parent
PROJECT = ROOT.parents[1]
REFERENCE = PROJECT / 'runs/raw_seed100_lr0006_patience150_20261103'
NEW_RUNS = PROJECT / 'comparisons/all_positions_three_datasets_20261104/runs/main'


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n', encoding='utf-8')


def notify(stage, **values):
    row = {'time': datetime.now().astimezone().isoformat(), 'stage': stage, **values}
    print(json.dumps(row), flush=True)
    write(ROOT / 'diagnostics/progress.json', row)


def load_data(dataset, split):
    with np.load(REFERENCE / dataset / 'prepared' / f'{split}.npz') as values:
        return {key: values[key].copy() for key in values.files}


def frequencies(training, n):
    raw = np.concatenate((training['seq'].ravel(), training['target']))
    full = np.bincount(raw[(raw >= 0) & (raw < n)], minlength=n)
    counts = np.zeros(n, dtype=np.int64)
    for history, target in zip(training['seq'], training['target']):
        chain = [int(x) for x in history if 0 <= x < n] + [int(target)]
        np.add.at(counts, chain[1:], 1)
    base = np.sqrt(counts + 1.)
    base /= base.sum()
    cutoff = int(np.sort(full)[::-1][math.ceil(.2*n)-1])
    return base, full, full < cutoff, counts, cutoff


def log_rows(directory):
    rows = []
    for line in (Path(directory) / 'choice_log.jsonl').read_text(encoding='utf-8').splitlines():
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return rows


def ema_age(steps):
    age = 0.
    for step in range(1, steps+1):
        decay = min(.9999, (1+step)/(10+step))
        age = decay*(age+1)
    decay = min(.9999, (1+steps)/(10+steps))
    return {'steps': steps, 'actual_decay': decay, 'instantaneous_1_over_1_minus_decay': 1/(1-decay),
            'exact_parameter_weight_mean_age_steps': age}


def check_a():
    """Align exact logged epochs; missing entries are never forward-filled."""
    epochs = (100, 200, 280, 300, 400, 500, 600, 800, 1000, 1200)
    paths = {'h0': PROJECT/'comparisons/toys_root_only_20261104/runs/root',
             'hn': NEW_RUNS/'Toys', 'ho': PROJECT/'comparisons/toys_random_three_levels_20261104/runs/random_three'}
    result = {'matched_epochs': [], 'runs': [], 'paired_hn_vs_root': []}
    epoch_values = {}
    for kind, parent in paths.items():
        for seed in (100, 101, 102):
            directory = parent/f'seed_{seed}'
            training = read(directory/'training.json')
            logs = log_rows(directory)
            vals = {row['epoch']: row for row in logs if row.get('validation')}
            epoch_values[kind, seed] = vals
            best = max(vals.values(), key=lambda row: row['validation']['ndcg@10'])
            assert best['epoch'] == training['best_epoch']
            result['runs'].append({'kind': kind, 'seed': seed, 'best_epoch': best['epoch'],
                'best_validation_ndcg10': best['validation']['ndcg@10'],
                'stop_epoch': training['epochs_completed'], 'source': str(directory),
                'best_pre_stop_epoch_gap': training['epochs_completed']-best['epoch']})
            for epoch in epochs:
                row = vals.get(epoch)
                shares = row.get('rate_shares') if row else None
                if shares:
                    layer_shares = {k: v['mean'] for k, v in shares.items() if isinstance(v, dict) and 'mean' in v}
                    if 'root_mean' in shares:
                        layer_shares['catalog'] = shares['root_mean']
                else:
                    layer_shares = {'catalog': 1.} if kind == 'h0' and row else None
                result['matched_epochs'].append({'kind': kind, 'seed': seed, 'epoch': epoch,
                    'validation_ndcg10': row['validation']['ndcg@10'] if row else None,
                    'shares': layer_shares,
                    'missing_reason': None if row else 'training ended before this epoch or no exact validation record'})
    for seed in (100, 101, 102):
        shared = sorted(set(epoch_values['hn', seed]) & set(epoch_values['h0', seed]))
        deltas = [epoch_values['hn',seed][t]['validation']['ndcg@10']-epoch_values['h0',seed][t]['validation']['ndcg@10'] for t in shared]
        result['paired_hn_vs_root'].append({'seed': seed, 'shared_validation_epochs': len(shared),
            'hn_higher_epochs': int(np.sum(np.array(deltas)>0)), 'root_higher_epochs': int(np.sum(np.array(deltas)<0)),
            'mean_hn_minus_root': float(np.mean(deltas)), 'last_shared_epoch': shared[-1],
            'last_shared_hn_minus_root': deltas[-1]})
    # Include the same 12 completed Toys runs used by the prior summary, adding
    # the now-complete Root seed102 separately. Spearman association is not causation.
    evidence = read(PROJECT/'docs/experiments/hierarchy_discussion_20261104/evidence.json')
    toys = [r for r in evidence['completed_runs'] if r['dataset']=='Toys']
    rho, p = spearmanr([r['best_epoch'] for r in toys], [r['test']['ndcg@10'] for r in toys])
    result['best_epoch_test_rank_association_original_12'] = {'n':len(toys),'spearman_rho':float(rho),'nominal_p':float(p),
        'interpretation':'exploratory association across different methods and seeds; not proof of stopping confounding'}
    evaluate_source = (PROJECT/'evaluate.py').read_text(encoding='utf-8')
    train_source = (PROJECT/'train.py').read_text(encoding='utf-8')
    fork = torch.random.get_rng_state().clone()
    def sample_validation_rng():
        with torch.random.fork_rng(devices=[]):
            payload = b'100:val:0'
            torch.manual_seed(int.from_bytes(hashlib.sha256(payload).digest()[:8], 'big')%(2**31-1))
            return torch.rand(32)
    a, b = sample_validation_rng(), sample_validation_rng()
    result['protocol_audit'] = {
        'existing_validation_rng_already_resets': 'payload = f"{seed}:{split}:{stream}"' in evaluate_source and 'fork_rng' in evaluate_source,
        'cpu_rng_repeat_identical': bool(torch.equal(a,b)), 'cpu_training_rng_restored':bool(torch.equal(fork,torch.random.get_rng_state())),
        'caveat':'Same random-number stream is reused, but model-dependent probabilities and adaptive substep counts do not imply identical realized trajectories or zero sampling error.',
        'existing_ema_ramp_detected': 'min(self.decay, (1 + self.updates) / (10 + self.updates))' in train_source,
        'ema_cap_first_reached_at_step':89990,
        'ema_at_18000':ema_age(18000),
        'minimum_step_epochs':{d: math.ceil(18000/math.ceil(read(REFERENCE/d/'prepared/metadata.json')['splits']['train']['used_rows']/256)) for d in ('Toys','Beauty','Sports')},
        'min_steps_changes_existing_completed_toys_runs':False,
        'min_steps_reason':'All nine completed h0/hn/ho runs already exceed 18000 steps; even hn seed101 stops at 430*61=26230 steps.'}
    result['interpretation'] = 'Matched curves test whether a gap is persistent or changes over time. They cannot establish a causal EMA/gate explanation without counterfactual training.'
    write(ROOT/'diagnostics/check_a.json',result)
    return result


def vector_stats(values):
    values = np.asarray(values,dtype=float)
    return {'mean':float(values.mean()),'std':float(values.std()),'median':float(np.median(values)),
            'p10':float(np.quantile(values,.1)),'p90':float(np.quantile(values,.9))}


def target_mass_and_rank(destination, targets, base, gamma=.02):
    """Exact full-catalog target rank without constructing rows of dense scores.

    Start with sorted background counts, then correct only sparse neighbor
    entries. Ties are broken by lower item ID for deterministic diagnostics.
    """
    target_mass = np.asarray(destination[np.arange(len(targets)),targets]).ravel()
    baseline = (1-gamma)*base
    sorted_baseline = np.sort(baseline)
    target_score = baseline[targets] + gamma*target_mass
    ranks = np.empty(len(targets),dtype=np.int64)
    for row, target in enumerate(targets):
        threshold = target_score[row]
        rank = len(base)-np.searchsorted(sorted_baseline,threshold,side='right')
        rank += int(np.sum((baseline[:target] == threshold)))
        start,end = destination.indptr[row:row+2]
        ids = destination.indices[start:end]
        added = gamma*destination.data[start:end]
        before = (baseline[ids]>threshold) | ((baseline[ids]==threshold)&(ids<target))
        after_scores = baseline[ids]+added
        after = (after_scores>threshold) | ((after_scores==threshold)&(ids<target))
        ranks[row] = 1+rank+int(after.sum())-int(before.sum())
    return target_mass,ranks


def group_rows(values, tail_mask):
    return {name: {'rows':int(mask.sum()), **{key:vector_stats(np.asarray(v)[mask]) for key,v in values.items()}}
            for name,mask in [('all',np.ones(len(tail_mask),bool)),('head',~tail_mask),('tail',tail_mask)]}


def block_predictability(destination, targets, labels):
    k=int(labels.max())+1
    indicator=sparse.csr_matrix((np.ones(len(labels)),(np.arange(len(labels)),labels)),shape=(len(labels),k))
    block_mass=destination@indicator
    return np.asarray(block_mass[np.arange(len(targets)),labels[targets]]).ravel()


def check_b(training, validation, base, tail, native_labels, tables, counts, clean_labels, adjacent_labels):
    n=len(base)
    result={'table_protocol':'One training-built table per mode shared by training, validation, and test.',
            'tail_definition':'retained training history plus target counts; same rule as existing long-tail evaluation',
            'statistics':[],'paired_own_final_edge_removal':[],
            'causal_caveat':'Train/validation cohorts differ. A gap alone is not a significance test or proof of test-data leakage. Paired own-edge deletion isolates a neighbor self-inclusion effect; bn still includes training targets.'}
    for mode,table in tables.items():
        partition=native_labels if mode!='nn_clean' else clean_labels
        for split,data in [('train',training),('val',validation)]:
            destination=(history_weights(data['seq'],n)@table).tocsr()
            mass,ranks=target_mass_and_rank(destination,data['target'],base)
            last=np.array([row[row<n][-1] for row in data['seq']])
            metrics={'neighbor_target_mass':mass,'rho_target_rank_gamma002':ranks,
                     'same_native_community':(partition[data['target']]==partition[last]).astype(float),
                     'same_original_community_fixed':(native_labels[data['target']]==native_labels[last]).astype(float),
                     'same_clean_adjacent_community':(adjacent_labels[data['target']]==adjacent_labels[last]).astype(float)}
            _,ranks10=target_mass_and_rank(destination,data['target'],base,.10)
            metrics['rho_target_rank_gamma010']=ranks10
            result['statistics'].append({'mode':mode,'split':split,'groups':group_rows(metrics,tail[data['target']])})
            if split=='train' and mode in ('nn','no'):
                notify('B_own_final_edges',mode=mode)
                removed=remove_own_final_edges(data['seq'],data['target'],counts[mode],table,n)
                removed_mass,removed_rank=target_mass_and_rank(removed,data['target'],base)
                delta=mass-removed_mass
                paired={'own_edge_target_mass_increase':delta,'rank_improvement_from_own_edges':removed_rank-ranks,
                        'target_mass_with_own_edges':mass,'target_mass_without_own_edges':removed_mass,
                        'positive_own_edge_effect':(delta>1e-12).astype(float)}
                result['paired_own_final_edge_removal'].append({'mode':mode,'groups':group_rows(paired,tail[data['target']])})
    write(ROOT/'diagnostics/check_b.json',result)
    return result


def partition_metrics(labels,base,tail,features,destination,validation,centers=None):
    k=int(labels.max())+1
    sizes=np.bincount(labels,minlength=k)
    masses=np.bincount(labels,weights=base,minlength=k)
    head_counts=np.bincount(labels,weights=~tail,minlength=k)
    head_ratio=head_counts/sizes
    if centers is None:
        sums=np.zeros((k,features.shape[1]))
        np.add.at(sums,labels,features)
        centers=sums/sizes[:,None]
        centers/=np.linalg.norm(centers,axis=1,keepdims=True).clip(1e-12)
    similarity=features@centers.T
    ranks=np.argsort(-similarity,axis=1,kind='stable')
    nonzero=np.linalg.norm(features,axis=1)>1e-10
    top1=ranks[:,0]==labels
    top3=np.any(ranks[:,:min(3,k)]==labels[:,None],axis=1)
    fidelity={}
    for label,mask in [('all',np.ones(len(labels),bool)),('head',~tail),('tail',tail)]:
        usable=mask&nonzero
        fidelity[label]={'items':int(mask.sum()),'isolates':int((mask&~nonzero).sum()),
                         'top1':float(top1[mask].mean()),'top3':float(top3[mask].mean()),
                         'top1_nonzero_features':float(top1[usable].mean()) if usable.any() else None,
                         'top3_nonzero_features':float(top3[usable].mean()) if usable.any() else None}
    # The supplied clustering reference defines Head as the top floor(0.2*N)
    # by bn background, whereas existing long-tail evaluation uses a strict
    # training-interaction frequency boundary. Report both without conflating.
    b_order=np.lexsort((np.arange(len(base)),-base))
    b_head=np.zeros(len(base),dtype=bool)
    b_head[b_order[:int(.2*len(base))]]=True
    b_rank_fidelity={}
    for label,mask in [('head',b_head),('tail',~b_head)]:
        usable=mask&nonzero
        b_rank_fidelity[label]={'items':int(mask.sum()),'isolates':int((mask&~nonzero).sum()),
            'top1':float(top1[mask].mean()),'top3':float(top3[mask].mean()),
            'top1_nonzero_features':float(top1[usable].mean()) if usable.any() else None,
            'top3_nonzero_features':float(top3[usable].mean()) if usable.any() else None}
    predictable=block_predictability(destination,validation['target'],labels)
    metrics={'clusters':k,'block_size_mean':float(sizes.mean()),'block_size_std':float(sizes.std()),
        'block_size_min':int(sizes.min()),'block_size_max':int(sizes.max()),'singletons':int((sizes==1).sum()),
        'blocks_size_le5':int((sizes<=5).sum()),'items_in_blocks_size_le5':int(sizes[sizes<=5].sum()),
        'block_mass_mean':float(masses.mean()),'block_mass_std':float(masses.std()),
        'block_mass_min':float(masses.min()),'block_mass_max':float(masses.max()),
        'head_fraction_std':float(head_ratio.std()),'head_fraction_min':float(head_ratio.min()),'head_fraction_max':float(head_ratio.max()),
        'fidelity':fidelity,'validation_block_predictability':group_rows({'mass':predictable},tail[validation['target']]),
        'b_rank_fidelity':b_rank_fidelity,
        'b_rank_head_fraction_std':float((np.bincount(labels,weights=b_head,minlength=k)/sizes).std()),
        'block_masses':masses.tolist(),'block_sizes':sizes.tolist()}
    return metrics


def markdown_table(headers,rows):
    return '\n'.join(['| '+' | '.join(headers)+' |','| '+' | '.join(['---']*len(headers))+' |',
                      *['| '+' | '.join(map(str,row))+' |' for row in rows]])


def render_report(a,b,c):
    out=['# Step 0: CPU diagnostic report','',f"Checked: {datetime.now().astimezone().isoformat()}",'',
         'All work is CPU-only. Existing datasets, runs, source files, and task queues are unchanged. Step 1 has not started.','',
         '## A. Matched-epoch validation and gate trajectories','']
    for seed in (100,101,102):
        rows=[]
        for epoch in (100,200,280,300,400,500,600,800,1000,1200):
            row=[epoch]
            for kind in ('h0','hn','ho'):
                value=next(x for x in a['matched_epochs'] if (x['kind'],x['seed'],x['epoch'])==(kind,seed,epoch))
                row.append('not reached' if value['validation_ndcg10'] is None else f"{value['validation_ndcg10']:.6f}")
                s=value['shares']
                if kind!='h0':
                    row.append('—' if not s else '/'.join(f'{100*x:.3f}%' for x in s.values()))
            rows.append(row)
        out.extend([f'### Toys seed {seed}','',markdown_table(['Epoch','Root val N10','hn val N10','hn shares','ho val N10','ho shares'],rows),''])
    out.extend(['Native shares are community/catalog; random shares are fine/coarse/catalog. Missing exact epochs are not filled from neighboring epochs.','',
        markdown_table(['Seed','Common validation epochs','hn wins','Root wins','Last common epoch','Last hn−Root'],
            [[v['seed'],v['shared_validation_epochs'],v['hn_higher_epochs'],v['root_higher_epochs'],v['last_shared_epoch'],f"{v['last_shared_hn_minus_root']:+.6f}"] for v in a['paired_hn_vs_root']]),'',
        'Existing validation already resets its RNG for every call. CRN reuses random numbers, not identical model-dependent sampled paths and not zero Monte Carlo error.','',
        f"EMA uses a ramp, reaching its 0.9999 cap only at step 89,990. At step 18,000 its decay is {a['protocol_audit']['ema_at_18000']['actual_decay']:.7f}, and exact mean parameter age is {a['protocol_audit']['ema_at_18000']['exact_parameter_weight_mean_age_steps']:.1f} steps. A fixed 10,000-step lag from the start is incorrect.",'',
        f"18,000 steps correspond to {a['protocol_audit']['minimum_step_epochs']} epochs. Every completed h0/hn/ho Toys run already passed this threshold; adding it would not have prevented their recorded early stops.",'',
        f"Across the prior 12 heterogeneous Toys runs, selected-epoch/test-NDCG Spearman rho = {a['best_epoch_test_rank_association_original_12']['spearman_rho']:.4f}. This association is not a causal early-stop estimate.",'',
        '## B. Target self-inclusion diagnostics','',
        'All groups use the existing training-frequency tail definition. Probability/rank entries below are row means. Smaller rank is better. The mode-specific Infomap is original for nn/no and history-only lag<=3 for nn_clean.',''])
    rows=[]
    for r in b['statistics']:
        for group in ('head','tail'):
            g=r['groups'][group]
            rows.append([r['mode'],r['split'],group,g['rows'],f"{g['neighbor_target_mass']['mean']:.6f}",
                f"{g['rho_target_rank_gamma002']['mean']:.1f}",f"{g['rho_target_rank_gamma010']['mean']:.1f}",
                f"{100*g['same_native_community']['mean']:.2f}%",f"{100*g['same_clean_adjacent_community']['mean']:.2f}%"])
    out.extend([markdown_table(['Mode','Split','Target','Rows','n_u(target)','rho rank g=.02','rho rank g=.10','Same community','Clean-adjacent same community'],rows),'',
        '### Paired deletion of each training sample\'s own final-target edges','',
        'Each table is rebuilt locally after subtracting only that sample\'s final-label contributions, before self prior and top-16 selection. Other samples and this sample\'s internal history edges remain. This directly quantifies neighbor self-inclusion without confusing it with population differences.',''])
    rows=[]
    for r in b['paired_own_final_edge_removal']:
        for group in ('head','tail'):
            g=r['groups'][group]
            rows.append([r['mode'],group,f"{g['target_mass_with_own_edges']['mean']:.6f}",
                f"{g['target_mass_without_own_edges']['mean']:.6f}",f"{g['own_edge_target_mass_increase']['mean']:.6f}",
                f"{g['rank_improvement_from_own_edges']['mean']:.1f}",f"{100*g['positive_own_edge_effect']['mean']:.2f}%"])
    out.extend([markdown_table(['Mode','Target','With own edges','Without own edges','Mass increase','Mean rank improvement','Positive-effect rows'],rows),'',
        'Train/validation gaps alone are descriptive. No validation/test label was used to construct any table. The nn_clean intervention removes direct final-label edges but retains training-history self-influence; bn remains target-inclusive by design.','',
        '## C. Partition balance, assignment fidelity, and validation block predictability','',
        'Fidelity and Head-share SD use the requested top floor(0.2*N) items by bn background as Head; the remaining items are Tail. Block predictability uses the existing training-frequency evaluation boundary. Both definitions and both Head-share SDs are saved in JSON. Fidelity is cosine assignment relative to this collaborative embedding, not independently verified product semantics. Zero-feature isolates have tie-dependent fidelity, so JSON also reports fidelity excluding them.',''])
    for dataset,values in c.items():
        out.extend([f'### {dataset}',''])
        rows=[]
        for mode,g in values['partitions'].items():
            rows.append([mode,g['clusters'],f"{g['block_size_min']}–{g['block_size_max']}",f"{g['block_mass_mean']:.6f}",
                f"{g['block_mass_std']:.6f}",f"{g['block_mass_min']:.6f}–{g['block_mass_max']:.6f}",f"{g['b_rank_head_fraction_std']:.6f}"])
        out.extend([markdown_table(['Partition','K','Block sizes','Mass mean','Mass SD','Mass min–max','Head-share SD'],rows),''])
        rows=[]
        for mode,g in values['partitions'].items():
            rows.append([mode,*[f"{100*g['b_rank_fidelity'][group][key]:.2f}%" for group,key in [('head','top1'),('head','top3'),('tail','top1'),('tail','top3')]],
                *[f"{g['validation_block_predictability'][group]['mass']['mean']:.6f}" for group in ('all','head','tail')]])
        out.extend([markdown_table(['Partition','Head Top1','Head Top3','Tail Top1','Tail Top3','Val block mass','Head block mass','Tail block mass'],rows),'',
            f"hs/hr validation block-mass ratio: {values['hs_vs_hr_predictability_ratio']:.4f}. Exact per-stratum counts match; maximum corresponding-block weighted-mass difference: {values['paired_balance']['max_corresponding_block_mass_difference']:.6f}.",'',
            f"History-only spectral graph isolates: {values['construction']['graph']['isolated_items']}. Historical Real-Scope: {values['historical_real_scope']['status']}.",''])
    out.extend(['## Interpretation and review gate','',
        '- Fixed strata enforce matching **counts per frequency band**; weighted mass equality and identical within-band frequencies are not mathematical guarantees. The final incomplete stratum uses the same floor/ceiling capacities in both arms.','- A positive global reset prevents strictly closed communities. Small blocks can change actual exit rates and predictive support; isolation and performance degradation must be measured rather than inferred from block count alone.','- Old supervised-target graph construction is training-only, not held-out-label leakage. Its self-inclusion can nevertheless create a train/held-out structural asymmetry; the paired deletion results quantify the neighbor part. A full leave-one-user-out Infomap intervention was not run, so community-level causal attribution remains limited.','- CRN is already present. A proposed 18,000-step guard does not alter the recorded stopping times of completed matched Toys runs.','- No Step 1 or Step 2 GPU run has started. Review the tables and measured deviations before deciding how to proceed.','',
        'Full arrays, distributions, group counts, construction settings, graph/partition artifacts, and source paths are in the adjacent JSON and NPZ files.'])
    (ROOT/'diagnostics/report.md').write_text('\n'.join(out)+'\n',encoding='utf-8')


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--report-only',action='store_true')
    args=parser.parse_args()
    torch.set_num_threads(2)
    if args.report_only:
        render_report(read(ROOT/'diagnostics/check_a.json'),read(ROOT/'diagnostics/check_b.json'),read(ROOT/'diagnostics/check_c.json'))
        return
    started=time.perf_counter()
    (ROOT/'diagnostics').mkdir(exist_ok=True)
    notify('A_start')
    a=check_a()
    c={}
    toys_inputs=None
    for dataset,k in [('Toys',12),('Beauty',12),('Sports',16)]:
        notify('dataset_start',dataset=dataset)
        training,validation=load_data(dataset,'train'),load_data(dataset,'val')
        n=read(REFERENCE/dataset/'prepared/metadata.json')['item_count']
        base,full,tail,bn_counts,cutoff=frequencies(training,n)
        clean_counts=count_transitions(training['seq'],training['target'],n,'nn_clean')
        graph=directed_graph(clean_counts,n)
        clean_table=build_table(clean_counts,n)
        destination=(history_weights(validation['seq'],n)@clean_table).tocsr()
        output=ROOT/'partitions'/dataset
        output.mkdir(parents=True,exist_ok=True)
        sparse.save_npz(output/'clean_directed_graph.npz',graph)
        sparse.save_npz(output/'clean_neighbor_table.npz',clean_table)
        notify('spectral_and_balanced_assignment',dataset=dataset,items=n,clusters=k)
        labels,features,centers,construction,strata=build_stratified_partitions(graph,base,k)
        native=torch.load(REFERENCE/dataset/'structure.pt',map_location='cpu',weights_only=True)['blocks'][0].numpy()
        labels['hn_original']=native
        notify('partition_metrics',dataset=dataset)
        measurements={name:partition_metrics(ids,base,tail,features,destination,validation,centers.get(name)) for name,ids in labels.items()}
        masses_hs=np.bincount(labels['hs'],weights=base,minlength=k)
        masses_hr=np.bincount(labels['hr'],weights=base,minlength=k)
        result={'dataset':dataset,'item_count':n,'tail_items':int(tail.sum()),'tail_frequency_boundary':cutoff,
            'settings':{'K':k,'M':20,'emb_dim':32,'num_iters':10,'partition_seed':100,'graph':'history-only directed lag<=3, inverse-distance; affinity W+W.T without self loops'},
            'source':str(REFERENCE/dataset),'construction':construction,'partitions':measurements,
            'historical_real_scope':{'status':'not found in searched Grow comparisons/docs and NewChoice source; no substitute fabricated'},
            'paired_balance':{'identical_per_stratum_counts':True,'identical_block_sizes':bool(np.array_equal(np.bincount(labels['hs']),np.bincount(labels['hr']))),
                'max_corresponding_block_mass_difference':float(np.max(np.abs(masses_hs-masses_hr)))},
            'hs_vs_hr_predictability_ratio':measurements['hs']['validation_block_predictability']['all']['mass']['mean']/measurements['hr']['validation_block_predictability']['all']['mass']['mean']}
        np.savez_compressed(output/'partitions.npz',base=base,tail=tail,features=features,bn_counts=bn_counts,**labels)
        c[dataset]=result
        write(ROOT/'diagnostics/check_c.json',c)
        if dataset=='Toys':
            toys_inputs=(training,validation,base,tail,native,clean_table,clean_counts)
        notify('dataset_complete',dataset=dataset,elapsed_seconds=time.perf_counter()-started)
    training,validation,base,tail,native,clean_table,clean_counts=toys_inputs
    n=len(base)
    notify('B_clean_infomap_start')
    clean_labels,clean_info=infomap_labels(directed_graph(clean_counts,n))
    adjacent_counts=count_transitions(training['seq'],training['target'],n,'nn_clean',lags=1)
    adjacent_labels,adjacent_info=infomap_labels(directed_graph(adjacent_counts,n))
    write(ROOT/'diagnostics/clean_infomap.json',{'lags3':clean_info,'adjacent_only':adjacent_info})
    np.savez_compressed(ROOT/'partitions/Toys/clean_infomap.npz',lag3=clean_labels,adjacent=adjacent_labels)
    counts={'nn_clean':clean_counts}
    tables={'nn_clean':clean_table}
    for mode in ('nn','no'):
        counts[mode]=count_transitions(training['seq'],training['target'],n,mode)
        tables[mode]=build_table(counts[mode],n)
    notify('B_measure_start')
    b=check_b(training,validation,base,tail,native,tables,counts,clean_labels,adjacent_labels)
    render_report(a,b,c)
    write(ROOT/'diagnostics/manifest.json',{'created_at':datetime.now().astimezone().isoformat(),
        'elapsed_seconds':time.perf_counter()-started,'device':'cpu','cuda_visible_devices':os.environ['CUDA_VISIBLE_DEVICES'],
        'training_started':False,'existing_outputs_modified':False,'step1_requires_review':True,
        'sources':{d:str(REFERENCE/d) for d in ('Toys','Beauty','Sports')},
        'checks':['A matched logs and protocol audit','B train/validation and paired own-edge removal','C all three datasets spectral/random/balanced/native']})
    notify('complete',elapsed_seconds=time.perf_counter()-started)


if __name__=='__main__':
    # Write directly instead of relying on PowerShell's buffered native redirection.
    sys.stdout=(ROOT/'engine.log').open('w',encoding='utf-8',buffering=1)
    trace=(ROOT/'engine_stacks.log').open('w',encoding='utf-8')
    faulthandler.dump_traceback_later(60,repeat=True,file=trace)
    with threadpool_limits(limits=2):
        main()
    faulthandler.cancel_dump_traceback_later()
