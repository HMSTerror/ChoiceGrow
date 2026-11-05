"""Finish CPU-only neighbor diagnostics using already saved partition results.

No clustering, model training, or inference is repeated. Positive-only mixture
reconstruction prevents numerical cancellation at zero-probability targets.
"""
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
from datetime import datetime
import time
import torch
import numpy as np
from scipy import sparse
from threadpoolctl import threadpool_limits
from diagnostics_step0 import (ROOT, REFERENCE, read, write, notify, load_data,
    frequencies, history_weights, partition_metrics, count_transitions,
    build_table, check_b, render_report)


def main():
    started=time.perf_counter()
    torch.set_num_threads(2)
    a=read(ROOT/'diagnostics/check_a.json')
    c=read(ROOT/'diagnostics/check_c.json')
    for dataset in ('Toys','Beauty','Sports'):
        validation=load_data(dataset,'val')
        folder=ROOT/'partitions'/dataset
        table=sparse.load_npz(folder/'clean_neighbor_table.npz')
        with np.load(folder/'partitions.npz') as saved:
            base,tail,features=saved['base'],saved['tail'],saved['features']
            destination=(history_weights(validation['seq'],len(base))@table).tocsr()
            for name in ('hs','hr','balanced_kmeans','hn_original'):
                c[dataset]['partitions'][name]=partition_metrics(
                    saved[name],base,tail,features,destination,validation)
    write(ROOT/'diagnostics/check_c.json',c)
    training,validation=load_data('Toys','train'),load_data('Toys','val')
    n=c['Toys']['item_count']
    base,_,tail,_,_=frequencies(training,n)
    with np.load(ROOT/'partitions/Toys/partitions.npz') as saved:
        native=saved['hn_original'].copy()
    with np.load(ROOT/'partitions/Toys/clean_infomap.npz') as saved:
        clean,adjacent=saved['lag3'].copy(),saved['adjacent'].copy()
    counts={mode:count_transitions(training['seq'],training['target'],n,mode)
            for mode in ('nn_clean','nn','no')}
    tables={mode:build_table(values,n) for mode,values in counts.items()}
    notify('B_positive_mixture_rebuild_start')
    b=check_b(training,validation,base,tail,native,tables,counts,clean,adjacent)
    render_report(a,b,c)
    write(ROOT/'diagnostics/manifest.json',{
        'created_at':datetime.now().astimezone().isoformat(),
        'partition_results_reused':True,'finalization_seconds':time.perf_counter()-started,
        'device':'cpu','cuda_visible_devices':os.environ['CUDA_VISIBLE_DEVICES'],
        'training_started':False,'existing_outputs_modified':False,'step1_requires_review':True,
        'sources':{d:str(REFERENCE/d) for d in ('Toys','Beauty','Sports')},
        'checks':['A matched logs and protocol audit','B train/validation and paired own-edge removal',
                  'C all three datasets spectral/random/balanced/native'],
        'paired_deletion_numerics':'sum nonnegative corrected constituent rows; no cancellation residues'})
    notify('complete',finalization_seconds=time.perf_counter()-started)


if __name__=='__main__':
    with threadpool_limits(limits=2):
        main()
