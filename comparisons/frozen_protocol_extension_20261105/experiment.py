"""Frozen Toys protocol: structural artifacts and final split-specific metrics.

Training and validation selection call the unchanged Trainer and Evaluator.
This module only prepares the authorized factors and adds final tail reports.
"""
# Note: ../../docs/decisions/implemented/feature/2026-11-04-fixed-shares.md
from collections import Counter
from copy import deepcopy
from datetime import datetime
import hashlib
import json
import math
from pathlib import Path
import shutil
import sys

ROOT=Path(__file__).resolve().parent
PROJECT=ROOT.parents[1]
sys.path.insert(0,str(ROOT/'code'))
import numpy as np
import torch
from scipy import sparse
from configuration import load_config
from data import load_split, metadata, prepared_root, write_json
from checkpoint import CheckpointStore, make_denoiser
from diffusion import ChoiceDiffusion
from evaluate import Evaluator
from train import Trainer
from neighbor_statistics import count_transitions

PYTHON=Path('E:/anaco/python.exe')
BASELINE=PROJECT/'comparisons/toys_root_only_20261104/runs/root/seed_100'
STEP0=PROJECT/'comparisons/stratified_scope_20261104_1933'
METRICS=('hr@5','hr@10','ndcg@5','ndcg@10')


def now():
    return datetime.now().astimezone().isoformat()


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def progress(directory):
    latest,best=0,None
    path=Path(directory)/'choice_log.jsonl'
    if path.exists():
        for line in path.read_text(encoding='utf-8').splitlines():
            try:
                row=json.loads(line)
            except json.JSONDecodeError:
                continue
            latest=max(latest,row.get('epoch',0))
            metrics=row.get('validation')
            if metrics and (best is None or metrics['ndcg@10']>best['ndcg@10']):
                best={'epoch':row['epoch'],'ndcg@10':metrics['ndcg@10']}
    return {'latest_epoch':latest,'best_validation':best}


def split_hashes(directory):
    return {split:sha256(prepared_root(directory)/'prepared'/f'{split}.npz') for split in ('train','val','test')}


def tensor_table(counts,n,width=16,self_prior=1.):
    """Apply the same self prior, item-ID tie rule and float32 normalization as nn."""
    width=min(width,n)
    ids=torch.arange(n)[:,None].expand(-1,width).clone()
    weights=torch.zeros(n,width);weights[:,0]=1.
    for source,raw in counts.items():
        values=Counter(raw);values[source]+=self_prior
        selected=sorted(values.items(),key=lambda pair:(-pair[1],pair[0]))[:width]
        ids[source,:len(selected)]=torch.tensor([j for j,_ in selected])
        probabilities=torch.tensor([w for _,w in selected],dtype=torch.float32)
        weights[source].zero_()
        weights[source,:len(selected)]=probabilities/probabilities.sum()
    return ids,weights


def audit_inputs():
    """Check the reused fit, actual splits, bn counts and reviewed partition mapping."""
    config=read(BASELINE/'resolved_config.json')
    assert config['seed']==100 and config['dataset']=='Toys'
    assert config['training']['patience']==150 and config['training']['validation_interval']==10
    assert config['training']['validation_streams']==1 and config['training']['test_streams']==3
    assert config['training']['learning_rate']==.0006 and 'min_steps' not in config['training']
    assert config['destination']['neighbor_mix']==.02
    assert not config['diffusion']['dynamic_rate_gate']
    assert math.isclose(config['diffusion']['total_budget'],-math.log(.1),abs_tol=1e-14)
    assert config['diffusion']['reset_rate']==.035
    training=read(BASELINE/'training.json')
    best=progress(BASELINE)['best_validation']
    assert best=={'epoch':training['best_epoch'],'ndcg@10':training['best_validation']['ndcg@10']}
    checkpoint=torch.load(BASELINE/'choice_best.pt',map_location='cpu',weights_only=True)
    assert checkpoint['epoch']==training['best_epoch'] and checkpoint['config']==config
    assert checkpoint['validation']==training['best_validation'] and 'rate_gate' not in checkpoint
    for name in ('train.py','evaluate.py','diffusion.py','kernel.py','models.py','checkpoint.py','data.py','configuration.py'):
        assert (ROOT/'code'/name).read_bytes()==(PROJECT/'comparisons/toys_root_only_20261104/code'/name).read_bytes(),name
    values=load_split(BASELINE,'train');n=metadata(BASELINE)['item_count']
    artifact=torch.load(BASELINE/'structure.pt',map_location='cpu',weights_only=True)
    assert artifact['blocks'].shape==(1,n) and not artifact['blocks'].any()
    counts=np.zeros(n,dtype=np.int64)
    for history,target in zip(values['seq'].tolist(),values['target'].tolist()):
        chain=[i for i in history if 0<=i<n]+[target]
        np.add.at(counts,chain[1:],1)
    expected=(torch.from_numpy(counts).float()+1).sqrt();expected/=expected.sum()
    assert torch.equal(expected,artifact['base'])
    with np.load(STEP0/'partitions/Toys/partitions.npz') as saved:
        np.testing.assert_array_equal(saved['bn_counts'],counts)
        np.testing.assert_allclose(saved['base'],expected.numpy(),rtol=3e-7,atol=1e-10)
        for mode in ('hs','hr'):
            assert len(saved[mode])==n and len(np.unique(saved[mode]))==12
    legacy=count_transitions(values['seq'].numpy(),values['target'].numpy(),n,'nn')
    ids,weights=tensor_table(legacy,n)
    assert torch.equal(ids,artifact['neighbor_ids']) and torch.equal(weights,artifact['neighbor_weights'])
    clean=count_transitions(values['seq'].numpy(),values['target'].numpy(),n,'nn_clean')
    rows,cols,weights=[],[],[]
    for source,row in clean.items():
        for target,weight in row.items():
            rows.append(source);cols.append(target);weights.append(weight)
    graph=sparse.csr_matrix((weights,(rows,cols)),shape=(n,n))
    delta=graph-sparse.load_npz(STEP0/'partitions/Toys/clean_directed_graph.npz')
    assert np.abs(delta.data).max(initial=0)==0
    clean_ids,clean_weights=tensor_table(clean,n)
    assert torch.isfinite(clean_weights).all()
    torch.testing.assert_close(clean_weights.sum(-1),torch.ones(n))
    return {'checked_at':now(),'prepared_root':str(prepared_root(BASELINE).resolve()),
        'prepared_split_sha256':split_hashes(BASELINE),'baseline_best_validation':best,
        'baseline_checkpoint_sha256':sha256(BASELINE/'choice_best.pt'),
        'step0_partitions_sha256':sha256(STEP0/'partitions/Toys/partitions.npz'),
        'unchanged_validation_and_training':True,'clean_graph_matches_step0':True,
        'legacy_neighbor_reconstruction_bitwise_equal':True,'bn_count_total':int(counts.sum()),
        'new_training_fits':8,'reused_training_candidates':1}


def prepare_job(job,neighbor=None,gamma=None):
    directory=Path(job['output']);directory.mkdir(parents=True,exist_ok=True)
    config=deepcopy(read(BASELINE/'resolved_config.json'))
    config.update(experiment=job['id'],neighbor_statistics=neighbor,background_statistics='bn',
        partition=job['partition'])
    config['destination']['neighbor_mix']=gamma
    shares=[1.] if job['partition']=='h0' else [job['community_share'],1-job['community_share']]
    config['diffusion'].update(dynamic_rate_gate=False,fixed_shares=shares,
        level_rates=[config['diffusion']['total_budget']*s for s in shares])
    config['diffusion'].pop('initial_shares',None)
    overrides=job.get('training_overrides',{})
    if not set(overrides)<= {'patience','epochs'}:
        raise ValueError('This experiment only authorizes patience and epoch-limit overrides')
    assert config['training']==read(BASELINE/'resolved_config.json')['training']
    config['training'].update(overrides)
    original=torch.load(BASELINE/'structure.pt',map_location='cpu',weights_only=True)
    artifact={key:value.clone() for key,value in original.items()}
    if neighbor=='nn_clean':
        train=load_split(BASELINE,'train');n=len(artifact['base'])
        counts=count_transitions(train['seq'].numpy(),train['target'].numpy(),n,'nn_clean')
        artifact['neighbor_ids'],artifact['neighbor_weights']=tensor_table(counts,n,
            config['destination']['neighbor_count'],config['destination']['neighbor_self_prior'])
    else:
        assert neighbor=='nn'
    if job['partition']!='h0':
        with np.load(STEP0/'partitions/Toys/partitions.npz') as saved:
            labels=torch.from_numpy(saved[job['partition']].copy())
        artifact['blocks']=torch.stack((labels,torch.zeros_like(labels)))
    write_json(directory/'data_source.json',{'path':str(prepared_root(BASELINE).resolve())})
    write_json(directory/'config.json',config);write_json(directory/'resolved_config.json',config)
    torch.save(artifact,directory/'structure.pt')
    write_json(directory/'artifacts.json',{'statistics_split':'train','neighbor_statistics':neighbor,
        'same_table_all_splits':True,'background':'bn; every retained successor once',
        'partition':job['partition'],'partition_source':str(STEP0/'partitions/Toys/partitions.npz'),
        'fixed_shares':shares,'dynamic_rate_gate':False,'neighbor_mix':gamma,
        'prepared_split_sha256':split_hashes(BASELINE),'training_overrides':overrides,
        'validation_and_early_stopping':'original validation; explicitly recorded patience/epoch overrides only'})
    job.update(neighbor=neighbor,gamma=gamma,state='queued')
    write_json(job['spec'],job)


def tail_mask(directory):
    train=load_split(directory,'train');n=metadata(directory)['item_count']
    positions=np.concatenate((train['seq'].numpy().ravel(),train['target'].numpy()))
    counts=np.bincount(positions[(positions>=0)&(positions<n)],minlength=n)
    cutoff=int(np.sort(counts)[::-1][math.ceil(.2*n)-1])
    return counts<cutoff,cutoff


def cached_metrics(directory,split,predictions):
    """Conditional tail accuracy uses full-catalog Top10, not tail-only reranking."""
    tail,cutoff=tail_mask(directory)
    with np.load(predictions) as saved:
        ranked,target=saved['top10'],saved['target']
    np.testing.assert_array_equal(target,load_split(directory,split)['target'].numpy())
    assert ranked.shape==(1 if split=='val' else 3,len(target),10)
    assert ranked.min()>=0 and ranked.max()<len(tail)
    result={'split':split,'streams':ranked.shape[0],'examples':len(target),'catalog_items':len(tail),
        'tail_catalog_items':int(tail.sum()),'target_groups':{},
        'protocol':{'counts':'training retained history plus final targets','tail_rule':f'frequency < {cutoff}',
                    'ranking':'full catalog; subset rows by true target','aggregation':'mean of stream metrics; one training seed'}}
    for k in (5,10):
        hits=ranked[:,:,:k]==target[None,:,None]
        found=hits.any(-1);rank=hits.argmax(-1)+1
        metrics={f'hr@{k}':found.mean(0),f'ndcg@{k}':(found/np.log2(rank+1)).mean(0)}
        result.update({key:float(v.mean()) for key,v in metrics.items()})
        result[f'tail_exposure@{k}']=float(tail[ranked[:,:,:k]].mean())
        for label,mask in (('tail',tail[target]),('head',~tail[target])):
            result['target_groups'].setdefault(label,{'examples':int(mask.sum())})
            result['target_groups'][label].update({key:float(v[mask].mean()) for key,v in metrics.items()})
    return result


def finish_evaluation(directory):
    directory=Path(directory);config=load_config(directory/'resolved_config.json')
    device=torch.device('cuda:0');evaluator=Evaluator(config,device)
    training=read(directory/'training.json')
    if not all((directory/name).exists() for name in ('report.json','recommendations.npz')):
        evaluator.test_best(directory)
    if not all((directory/name).exists() for name in ('validation_report.json','validation_recommendations.npz')):
        process=ChoiceDiffusion.from_directory(directory,config,device)
        state=CheckpointStore(directory,config).load_best(process,device)
        model=make_denoiser(directory,config,process).to(device)
        model.load_state_dict(state['model'])
        metrics=evaluator.evaluate(model,process,load_split(directory,'val'),'val',1,
            predictions_path=directory/'validation_recommendations.npz')
        assert abs(metrics['ndcg@10']-training['best_validation']['ndcg@10'])<1e-7
        write_json(directory/'validation_report.json',{'best_epoch':state['epoch'],
            'validation':metrics,'logged_best_validation':training['best_validation'],
            'selection_split':'val','additional_evaluation':'final selected checkpoint only; unchanged training-time validation'})
    for split,name,cache in (('val','long_tail_val.json','validation_recommendations.npz'),
                            ('test','long_tail.json','recommendations.npz')):
        metrics=cached_metrics(directory,split,directory/cache)
        overall=read(directory/('validation_report.json' if split=='val' else 'report.json'))['validation' if split=='val' else 'test']
        for key in METRICS:
            assert abs(metrics[key]-overall[key])<1e-7
        write_json(directory/name,metrics)
    assert read(directory/'report.json')['best_epoch']==training['best_epoch']
