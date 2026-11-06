"""Independently check completed CPU diagnostic artifacts and report coverage."""
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
from pathlib import Path
from datetime import datetime
import json
import math
import importlib.util
import torch
import numpy as np
from scipy import sparse
from neighbors import count_transitions, build_table

ROOT=Path(__file__).resolve().parent


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def main():
    manifest=read(ROOT/'diagnostics/manifest.json')
    assert manifest['device']=='cpu' and manifest['training_started'] is False
    assert manifest['step1_requires_review'] is True
    a,b,c=[read(ROOT/f'diagnostics/check_{key}.json') for key in 'abc']
    assert len(a['runs'])==9 and len(a['matched_epochs'])==90
    assert a['protocol_audit']['existing_validation_rng_already_resets']
    assert a['protocol_audit']['cpu_rng_repeat_identical']
    assert a['protocol_audit']['cpu_training_rng_restored']
    for row in a['runs']:
        source=read(Path(row['source'])/'training.json')
        assert source['best_epoch']==row['best_epoch']
        assert source['best_validation']['ndcg@10']==row['best_validation_ndcg10']
        assert row['stop_epoch']*61>18000
    assert len(b['statistics'])==6
    assert len(b['paired_own_final_edge_removal'])==2
    for row in b['statistics']:
        for group in row['groups'].values():
            assert 0<=group['neighbor_target_mass']['mean']<=1
            assert 1<=group['rho_target_rank_gamma002']['mean']<=11924
            assert 0<=group['same_native_community']['mean']<=1
    # Legacy controls must reproduce the tables actually used by old fits.
    source=Path('E:/Grow/runs/raw_seed100_lr0006_patience150_20261103/Toys')
    with np.load(source/'prepared/train.npz') as training:
        seq,target=training['seq'].copy(),training['target'].copy()
    module_path=ROOT.parents[1]/'comparisons/all_positions_three_datasets_20261104/code/neighbors.py'
    spec=importlib.util.spec_from_file_location('legacy_neighbors_verification',module_path)
    legacy=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(legacy)
    legacy_checks={}
    for mode,directory in [('no',source),('nn',Path('E:/Grow/comparisons/all_positions_three_datasets_20261104/runs/main/Toys/seed_100'))]:
        table=build_table(count_transitions(seq,target,11924,mode),11924)
        artifact=torch.load(directory/'structure.pt',map_location='cpu',weights_only=True)
        rebuilt_ids,rebuilt_weights=legacy.build_neighbors(torch.from_numpy(seq),torch.from_numpy(target),11924,
            statistics='last_target' if mode=='no' else 'all_positions')
        assert torch.equal(rebuilt_ids,artifact['neighbor_ids'])
        assert torch.equal(rebuilt_weights,artifact['neighbor_weights'])
        ids=artifact['neighbor_ids'].numpy()
        weights=artifact['neighbor_weights'].numpy()
        expected=sparse.coo_matrix((weights.ravel(),(np.repeat(np.arange(11924),ids.shape[1]),ids.ravel())),shape=table.shape).tocsr()
        delta=table-expected
        # Diagnostic probabilities are float64; saved training tables used
        # float32 normalization. Verify exact legacy replay above, support
        # equality here, and measure the expected rounding difference.
        expected.eliminate_zeros()
        assert (table.astype(bool)!=expected.astype(bool)).nnz==0
        maximum=float(np.abs(delta.data).max(initial=0))
        assert maximum<2*np.finfo(np.float32).eps
        legacy_checks[mode]={'saved_float32_replay_bitwise_equal':True,
            'float64_support_identical':True,'float64_vs_float32_max_difference':maximum}
    numerical={}
    for dataset,k in [('Toys',12),('Beauty',12),('Sports',16)]:
        folder=ROOT/'partitions'/dataset
        with np.load(folder/'partitions.npz') as saved:
            base,hs,hr=saved['base'],saved['hs'],saved['hr']
            assert hs.min()==hr.min()==0 and hs.max()==hr.max()==k-1
            assert np.isclose(base.sum(),1)
            order=np.lexsort((np.arange(len(base)),-base))
            for start in range(0,len(base),20*k):
                ids=order[start:start+20*k]
                counts=np.bincount(hs[ids],minlength=k)
                random_counts=np.bincount(hr[ids],minlength=k)
                np.testing.assert_array_equal(counts,random_counts)
                assert counts.max()-counts.min()<=1
            for variant in ('hs','hr','balanced_kmeans','hn_original'):
                labels=saved[variant]
                metrics=c[dataset]['partitions'][variant]
                sizes=np.bincount(labels)
                mass=np.bincount(labels,weights=base)
                assert sizes.min()==metrics['block_size_min'] and sizes.max()==metrics['block_size_max']
                assert abs(mass.std()-metrics['block_mass_std'])<1e-14
                b_head=np.zeros(len(base),dtype=bool)
                b_head[order[:int(.2*len(base))]]=True
                head_share=np.bincount(labels,weights=b_head)/sizes
                assert abs(head_share.std()-metrics['b_rank_head_fraction_std'])<1e-14
                assert metrics['b_rank_fidelity']['head']['items']==int(.2*len(base))
            numerical[dataset]={'mass_sd_hs':c[dataset]['partitions']['hs']['block_mass_std'],
                'mass_sd_hr':c[dataset]['partitions']['hr']['block_mass_std'],
                'strata_count_match':True,'overall_size_difference':int(np.ptp(np.bincount(hs)))}
        table=sparse.load_npz(folder/'clean_neighbor_table.npz')
        np.testing.assert_allclose(np.asarray(table.sum(1)).ravel(),1.,atol=1e-13)
    report=(ROOT/'diagnostics/report.md').read_text(encoding='utf-8')
    for phrase in ('Toys','Beauty','Sports','Same community','Mass SD','Head Top1','Tail Top3','Step 1 has not started'):
        assert phrase in report
    assert 'TODO' not in report and 'TBD' not in report
    chinese=(ROOT/'diagnostics/report_zh.md').read_text(encoding='utf-8')
    assert all(phrase in chinese for phrase in ('Step 1 尚未启动','目标自包含','89,990','1800.1','Real-Scope','2.256995'))
    historical=read(ROOT/'diagnostics/historical_real_scope.json')
    assert set(historical)=={'Toys','Beauty','Sports'}
    for value in historical.values():
        assert len(value['levels'])==3 and value['fidelity_and_current_validation_predictability'] is None
    result={'verified_at':datetime.now().astimezone().isoformat(),'status':'passed',
        'checks':['9 completed training sources and 90 exact epoch slots','6 neighbor/split groups and paired deletion summaries',
                  '3 datasets matching hs/hr stratum capacities and balanced block sizes','mass statistics recomputed from saved labels',
                  'all clean table rows normalized','report tables and review-gate coverage'],
        'numerical':numerical,'legacy_neighbor_replay':legacy_checks,'gpu_training_started':False}
    (ROOT/'diagnostics/verification.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(result,indent=2))


if __name__=='__main__':
    main()
