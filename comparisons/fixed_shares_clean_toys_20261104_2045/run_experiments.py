"""Run the approved stages, respecting the manual Step 1 decision hold."""
import argparse
from copy import deepcopy
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
import traceback
import psutil
import experiment as p

ROOT=p.ROOT
OUTPUT=ROOT/'runs'


def process_record(pid=None):
    process=psutil.Process(pid or os.getpid())
    return {'pid':process.pid,'created':process.create_time(),'command':process.cmdline()}


def alive(record):
    try:
        return bool(record) and abs(psutil.Process(record['pid']).create_time()-record['created'])<.01
    except psutil.Error:
        return False


def matching_worker(spec):
    exact=os.path.normcase(os.path.abspath(spec))
    found=[]
    for process in psutil.process_iter(['pid','name','cmdline']):
        args=process.info['cmdline'] or []
        if process.pid!=os.getpid() and '--worker-job' in args:
            if any(os.path.normcase(os.path.abspath(arg))==exact for arg in args):
                found.append(process.pid)
    return found


def code_hashes():
    return {str(path.relative_to(ROOT)):p.sha256(path)
            for path in sorted((ROOT/'code').glob('*.py'))} | {'neighbor_statistics.py':p.sha256(ROOT/'neighbor_statistics.py')}


def job_spec(identity,stage,partition,share=0.,reused=False):
    folder=OUTPUT/identity
    return {'id':identity,'stage':stage,'partition':partition,'community_share':share,
        'seed':100,'dataset':'Toys','output':str(folder),'spec':str(folder/'job.json'),
        'state':'evaluation_pending' if reused else 'blocked_on_step1' if stage==2 else 'queued',
        'reused_training':reused,'attempts':0,'pid':None}


def prepare():
    if (OUTPUT/'manifest.json').exists():
        raise FileExistsError('This suite is already prepared; refusing to overwrite it')
    audit=p.audit_inputs()
    jobs=[job_spec('step1/1A_clean_g002',1,'h0'),job_spec('step1/1B_clean_g010',1,'h0'),
          job_spec('step1/1C_nn_g002_reused',1,'h0',reused=True)]
    for partition in ('hs','hr'):
        for share in (.3,.5,.8):
            jobs.append(job_spec(f'step2/{partition}_s{int(share*10):02d}',2,partition,share))
    p.prepare_job(jobs[0],'nn_clean',.02)
    p.prepare_job(jobs[1],'nn_clean',.10)
    reused=Path(jobs[2]['output']);reused.mkdir(parents=True)
    for name in ('choice_best.pt','training.json','choice_log.jsonl','structure.pt','resolved_config.json',
                 'config.json','report.json','recommendations.npz'):
        shutil.copy2(p.BASELINE/name,reused/name)
    p.write_json(reused/'data_source.json',{'path':str(p.prepared_root(p.BASELINE).resolve())})
    p.write_json(reused/'reuse.json',{'source':str(p.BASELINE),'checkpoint_sha256':audit['baseline_checkpoint_sha256'],
        'verified':['prepared splits','bn','nn tensors','Root generator','training settings','checkpoint','full validation log']})
    jobs[2].update(neighbor='nn',gamma=.02,source=str(p.BASELINE))
    for job in jobs:
        p.write_json(job['spec'],job)
    manifest={'created_at':p.now(),'state':'prepared','dataset':'Toys','seed':100,'new_training_fits':8,
        'reused_training_candidates':1,'max_suite_workers':2,'step1_selection':None,
        'selection_rule':'highest logged best validation ndcg@10; ties prefer 1A then 1B then reused 1C; test excluded',
        'training_and_validation':'original Trainer/Evaluator; no min_steps or CRN changes',
        'step0_review':'approved by user','frozen_code_sha256':code_hashes(),'input_audit':audit,'jobs':jobs}
    p.write_json(OUTPUT/'input_verification.json',audit)
    p.write_json(OUTPUT/'execution_control.json',{'hold_new_jobs':False,'stop_requested':False,
        'max_suite_workers':2,'step1_max_workers':2,'step2_max_workers':1,
        'automatic_step2':True,'extra_seeds':False})
    save(manifest,'prepared')
    summarize(manifest)
    print(p.json.dumps({'event':'prepared','new_fits':8,'reused_candidates':1,'output':str(OUTPUT)}),flush=True)


def select_neighbors(jobs):
    """Read training selection metrics only; test and tail metrics never enter the decision."""
    candidates=[]
    for job in jobs:
        if job['stage']!=1:
            continue
        training=p.read(Path(job['output'])/'training.json')
        logged=p.progress(job['output'])['best_validation']
        assert logged=={'epoch':training['best_epoch'],'ndcg@10':training['best_validation']['ndcg@10']}
        candidates.append({'id':job['id'],'neighbor':job['neighbor'],'gamma':job['gamma'],
            'best_epoch':training['best_epoch'],'validation_ndcg10':logged['ndcg@10'],
            'reused_training':job['reused_training']})
    winner=max(candidates,key=lambda row:row['validation_ndcg10'])
    return {'selected_at':p.now(),'criterion':'logged best validation Overall NDCG@10 only',
        'test_used':False,'candidates':candidates,'winner':winner,
        'optional_1C':'reuse completed nn Root gamma=.02; no additional nn retraining'}


def summarize(manifest):
    rows=[]
    for job in manifest['jobs']:
        folder=Path(job['output'])
        row={key:job[key] for key in ('id','stage','state','partition','community_share','reused_training')}
        row.update(neighbor=job.get('neighbor'),gamma=job.get('gamma'),progress=p.progress(folder))
        row['training_protocol']=job.get('training_protocol','validation_early_stopping')
        if (folder/'resolved_config.json').exists():
            settings=p.read(folder/'resolved_config.json')['training']
            row['patience']=settings['patience']
            row['epoch_limit']=settings['epochs'] or None
        if all((folder/name).exists() for name in ('long_tail.json','long_tail_val.json','training.json')):
            training=p.read(folder/'training.json')
            val,test=p.read(folder/'long_tail_val.json'),p.read(folder/'long_tail.json')
            row.update(best_epoch=training['best_epoch'],epochs_completed=training['epochs_completed'],
                logged_best_validation_ndcg10=training['best_validation']['ndcg@10'],
                validation={'overall_ndcg10':val['ndcg@10'],'tail_ndcg10':val['target_groups']['tail']['ndcg@10'],
                    'tail_exposure10':val['tail_exposure@10'],'streams':1},
                test={'overall_hr10':test['hr@10'],'overall_ndcg10':test['ndcg@10'],
                    'tail_hr10':test['target_groups']['tail']['hr@10'],'tail_ndcg10':test['target_groups']['tail']['ndcg@10'],
                    'tail_exposure10':test['tail_exposure@10'],'streams':3},
                tail_protocol={'tail_catalog_items':test['tail_catalog_items'],
                    'validation_tail_targets':val['target_groups']['tail']['examples'],
                    'test_tail_targets':test['target_groups']['tail']['examples'],**test['protocol']})
        rows.append(row)
    result={'updated_at':p.now(),'state':manifest['state'],'training_seed':100,'rows':rows,
        'step1_selection':manifest['step1_selection'],'aggregation':'one training seed; val 1 stream; test 3 stream mean'}
    p.write_json(OUTPUT/'summary.json',result)
    lines=['# Toys seed100: clean neighbors and fixed hierarchy shares','',
        f"State: {manifest['state']}; completed entries: {sum(j['state']=='complete' for j in manifest['jobs'])}/9 (8 new fits + 1 reused fit).",'',
        'Original validation streams and interval are retained. Recorded stopping rules distinguish patience-based fits from the fixed-horizon hs.30 rerun. Final test: three sampling streams of one trained model.','',
        '| Run | State | Neighbor | gamma | Partition | Community share | Stop epoch | Best epoch | Val Overall N10 | Val Tail N10 | Test Overall N10 | Test Tail N10 |',
        '| --- | --- | --- | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |']
    for row in rows:
        metrics=[f"{row[split][key]:.6f}" for split in ('validation','test') for key in ('overall_ndcg10','tail_ndcg10')] if 'test' in row else ['pending']*4
        lines.append('| '+' | '.join(map(str,[row['id'],row['state'],row['neighbor'],row['gamma'],row['partition'],
            row['community_share'],row.get('epochs_completed','pending'),row.get('best_epoch','pending'),*metrics]))+' |')
    if manifest['step1_selection']:
        winner=manifest['step1_selection']['winner']
        lines+=['',f"Selected neighbor: {winner['neighbor']}; gamma={winner['gamma']}; validation NDCG@10={winner['validation_ndcg10']:.6f} at epoch {winner['best_epoch']}."]
    lines+=['','Tail accuracy subsets users by the true target after full-catalog ranking. The strict popularity boundary comes only from training histories + targets. Exact counts are in summary.json.']
    (OUTPUT/'summary.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    if manifest['step1_selection']:
        chosen=manifest['step1_selection']['winner']['id']
        comparison=[row for row in rows if row['id']==chosen or row['stage']==2]
        p.write_json(OUTPUT/'step2_comparison.json',{
            'updated_at':result['updated_at'],'state':manifest['state'],'training_seed':100,
            'selection':manifest['step1_selection'],'aggregation':result['aggregation'],'rows':comparison})
        table=['# Step 2: selected Root and six fixed-share structures','',
            f"Root control: {chosen}; reused without retraining. Test: three streams per seed100 model.",'',
            '| Run | State | Partition | s_C | Stop rule | Stop epoch | Best val NDCG@10 | Best epoch | Test HR@10 | Test NDCG@10 | Tail HR@10 | Tail NDCG@10 | Tail exposure@10 |',
            '| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |']
        for row in comparison:
            progress=row['progress'];best=progress['best_validation']
            stop_rule=(f"fixed {row['epoch_limit']}" if row['training_protocol']=='fixed_horizon_full_state'
                       else f"patience {row.get('patience','pending')}")
            values=[row['id'],row['state'],row['partition'],row['community_share'],stop_rule,
                row.get('epochs_completed','pending'),f"{best['ndcg@10']:.6f}" if best else 'pending',
                best['epoch'] if best else 'pending']
            values += [f"{row['test'][key]:.6f}" if 'test' in row else 'pending' for key in
                ('overall_hr10','overall_ndcg10','tail_hr10','tail_ndcg10','tail_exposure10')]
            table.append('| '+' | '.join(map(str,values))+' |')
        (OUTPUT/'step2_comparison.md').write_text('\n'.join(table)+'\n',encoding='utf-8')
    if all(job['state']=='complete' for job in manifest['jobs']):
        plot(result)
    return result


def plot(summary):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    chosen=summary['step1_selection']['winner']['id']
    root=next(row for row in summary['rows'] if row['id']==chosen)
    fig,axes=plt.subplots(2,2,figsize=(10,7),constrained_layout=True)
    for axis,(split,key,title) in zip(axes.ravel(),[
        ('validation','overall_ndcg10','Validation Overall NDCG@10'),('validation','tail_ndcg10','Validation Tail NDCG@10'),
        ('test','overall_ndcg10','Test Overall NDCG@10'),('test','tail_ndcg10','Test Tail NDCG@10')]):
        for partition,color in (('hs','#1768AC'),('hr','#D17A22')):
            rows=sorted([r for r in summary['rows'] if r['partition']==partition],key=lambda r:r['community_share'])
            axis.plot([0.]+[r['community_share'] for r in rows],[root[split][key]]+[r[split][key] for r in rows],
                '-o',label=partition,color=color)
        shares=sorted({0.}|{r['community_share'] for r in summary['rows'] if r['stage']==2})
        axis.set(title=title,xlabel='Fixed community share',xticks=shares);axis.grid(alpha=.2);axis.legend()
    fig.suptitle('Toys seed100; user-selected Root at share 0; fixed-share structures')
    fig.savefig(OUTPUT/'ndcg_by_fixed_share.png',dpi=180)
    fig.savefig(OUTPUT/'ndcg_by_fixed_share.svg')
    plt.close(fig)


def save(manifest,state):
    manifest.update(state=state,updated_at=p.now())
    for job in manifest['jobs']:
        job['progress']=p.progress(job['output'])
        p.write_json(job['spec'],job)
    p.write_json(OUTPUT/'manifest.json',manifest)
    p.write_json(OUTPUT/'status.json',{'state':state,'updated_at':p.now(),'coordinator':process_record(),
        'completed':sum(j['state']=='complete' for j in manifest['jobs']),'total':9,
        'selection':manifest['step1_selection'],'jobs':manifest['jobs']})


def worker(spec, resume=False):
    if matching_worker(spec):
        raise RuntimeError('A worker for this exact job is already alive')
    job=p.read(spec);folder=Path(job['output'])
    config=p.load_config(folder/'resolved_config.json')
    p.torch.set_num_threads(config['threads'])
    if config['tf32']:
        p.torch.set_float32_matmul_precision('high');p.torch.backends.cudnn.allow_tf32=True
    if code_hashes()!=p.read(OUTPUT/'manifest.json')['frozen_code_sha256']:
        raise RuntimeError('Frozen numerical source changed')
    try:
        if resume or not (folder/'training.json').exists():
            protocol=job.get('training_protocol')
            if resume and protocol not in ('full_state','fixed_horizon_full_state'):
                raise ValueError('This job has no full-state training protocol; EMA weights are not resumable')
            if not resume and any((folder/name).exists() for name in ('choice_best.pt','choice_newst.pt','choice_log.jsonl')):
                raise RuntimeError('Partial training has no optimizer state; refusing a restart or fake resume')
            p.write_json(folder/'worker_status.json',{'state':'training','pid':os.getpid(),'updated_at':p.now()})
            if protocol in ('full_state','fixed_horizon_full_state'):
                from full_state_training import FixedHorizonTrainer, StatefulTrainer
                trainer=FixedHorizonTrainer if protocol=='fixed_horizon_full_state' else StatefulTrainer
                trainer(folder,config,p.torch.device('cuda:0')).fit(resume=resume)
            else:
                p.Trainer(folder,config,p.torch.device('cuda:0')).fit()
        p.write_json(folder/'worker_status.json',{'state':'evaluating','pid':os.getpid(),'updated_at':p.now()})
        p.finish_evaluation(folder)
        p.write_json(folder/'worker_status.json',{'state':'complete','pid':os.getpid(),'updated_at':p.now()})
    except BaseException:
        p.write_json(folder/'worker_status.json',{'state':'failed','pid':os.getpid(),'updated_at':p.now(),
            'traceback':traceback.format_exc()})
        raise


def gpu_available():
    output=subprocess.check_output(['nvidia-smi','--query-gpu=memory.free','--format=csv,noheader,nounits'],text=True)
    free=int(output.strip().splitlines()[0])
    return free>=1800,free


def current_stage(manifest):
    """Do not admit structural jobs until all Step 1 fits and reports finish."""
    return 1 if any(j['stage']==1 and j['state']!='complete' for j in manifest['jobs']) else 2


def available_jobs(manifest,control):
    """Admit the current stage and, when enabled, only the current share pair."""
    if control.get('hold_new_jobs') or control.get('stop_requested'):
        return []
    stage=current_stage(manifest)
    if stage==2 and not control.get('automatic_step2',True):
        return []
    maximum=int(control.get('step1_max_workers',2) if stage==1 else control.get('step2_max_workers',1))
    active=sum(job['state']=='running' for job in manifest['jobs'])
    slots=max(0,min(maximum,int(control.get('max_suite_workers',2)))-active)
    candidates=[job for job in manifest['jobs'] if job['stage']==stage]
    if stage==2 and control.get('step2_pair_by_share'):
        pending=[job for job in candidates if job['state']!='complete']
        if not pending:
            return []
        current_share=min(job['community_share'] for job in pending)
        candidates=[job for job in candidates if job['community_share']==current_share]
    return [job for job in candidates
            if job['state'] in ('queued','evaluation_pending')][:slots]


def synchronize_jobs(manifest):
    """Adopt live workers when replacing only their coordinator; never restart a fit."""
    completed=False
    for job in manifest['jobs']:
        folder=Path(job['output'])
        if job['state']=='complete':
            continue
        live=matching_worker(job['spec'])
        if len(live)>1:
            raise RuntimeError(f"Duplicate live workers: {job['id']}: {live}")
        if live:
            job.update(state='running',pid=live[0],process=process_record(live[0]))
            continue
        status=p.read(folder/'worker_status.json') if (folder/'worker_status.json').exists() else {}
        complete=all((folder/name).exists() for name in
            ('training.json','choice_best.pt','validation_report.json','report.json',
             'long_tail_val.json','long_tail.json','validation_recommendations.npz','recommendations.npz'))
        if complete:
            job.update(state='complete',finished_at=p.now());completed=True
            continue
        if status.get('state')=='failed':
            job['state']='failed'
            raise RuntimeError(f"Worker failed: {job['id']}; {status.get('traceback','inspect worker.err.log')}")
        if job['state']=='running':
            job['state']='failed'
            raise RuntimeError(f"Worker exited without complete results: {job['id']}; partial training is not restarted")
    return completed


def coordinate():
    manifest=p.read(OUTPUT/'manifest.json')
    lock=OUTPUT/'launcher.lock'
    if lock.exists():
        if alive(p.read(lock)):
            raise RuntimeError('This coordinator is already alive')
        lock.unlink()
    with lock.open('x',encoding='utf-8') as stream:
        p.json.dump(process_record(),stream)
    p.write_json(OUTPUT/'launcher.json',process_record())
    (OUTPUT/'launcher.pid').write_text(str(os.getpid())+'\n',encoding='utf-8')
    try:
        if code_hashes()!=manifest['frozen_code_sha256']:
            raise RuntimeError('Frozen numerical source changed')
        # A coordinator replacement leaves independent training workers alive.
        # Reattach by exact spec command and process identity before admission.
        while True:
            completed=synchronize_jobs(manifest)
            control=p.read(OUTPUT/'execution_control.json')
            manifest['max_suite_workers']=int(control.get('max_suite_workers',2))
            if current_stage(manifest)==2 and not control.get('automatic_step2',True):
                save(manifest,'awaiting_user_decision');summarize(manifest);return
            if current_stage(manifest)==2 and manifest['step1_selection'] is None:
                if not control.get('automatic_selection',True):
                    save(manifest,'awaiting_user_decision');summarize(manifest);return
                selection=select_neighbors(manifest['jobs'])
                manifest['step1_selection']=selection
                p.write_json(OUTPUT/'step1_selection.json',selection)
                winner=selection['winner']
                for later in manifest['jobs']:
                    if later['stage']==2:
                        p.prepare_job(later,winner['neighbor'],winner['gamma'])
                save(manifest,'running')
            if all(job['state']=='complete' for job in manifest['jobs']):
                save(manifest,'complete');summarize(manifest);return
            active=any(job['state']=='running' for job in manifest['jobs'])
            if control.get('stop_requested') and not active:
                save(manifest,'stopped');summarize(manifest);return
            for job in available_jobs(manifest,control):
                ready,free=gpu_available()
                if not ready:
                    manifest['waiting_reason']=f'GPU free memory {free} MiB < 1800 MiB'
                    break
                if matching_worker(job['spec']):
                    raise RuntimeError(f"Unexpected live worker before launch: {job['id']}")
                manifest.pop('waiting_reason',None)
                folder=Path(job['output'])
                command=[str(p.PYTHON),'-B','-u',str(Path(__file__).resolve()),'--worker-job',job['spec']]
                with (folder/'worker.log').open('a',encoding='utf-8') as stdout, (folder/'worker.err.log').open('a',encoding='utf-8') as stderr:
                    process=subprocess.Popen(command,cwd=ROOT,stdout=stdout,stderr=stderr,
                        env={**os.environ,'PYTHONIOENCODING':'utf-8'},
                        creationflags=subprocess.CREATE_NO_WINDOW if os.name=='nt' else 0)
                job.update(state='running',pid=process.pid,process=process_record(process.pid),
                    command=command,started_at=p.now(),attempts=job['attempts']+1)
                save(manifest,'running')
                print(p.json.dumps({'event':'worker_started','job':job['id'],'pid':process.pid,'time':p.now()}),flush=True)
            active=any(job['state']=='running' for job in manifest['jobs'])
            save(manifest,'running' if active else 'waiting')
            if completed:
                summarize(manifest)
            time.sleep(15)
    except BaseException:
        p.write_json(OUTPUT/'coordinator_error.json',{'at':p.now(),'traceback':traceback.format_exc()})
        save(manifest,'failed');summarize(manifest)
        raise
    finally:
        lock.unlink(missing_ok=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--prepare',action='store_true')
    parser.add_argument('--summary-only',action='store_true')
    parser.add_argument('--worker-job',type=Path)
    parser.add_argument('--resume',action='store_true',help='Resume an explicit worker from its complete training_state.pt')
    args=parser.parse_args()
    if args.resume and not args.worker_job:
        parser.error('--resume requires --worker-job')
    if args.prepare:
        prepare()
    elif args.worker_job:
        worker(args.worker_job,resume=args.resume)
    elif args.summary_only:
        summarize(p.read(OUTPUT/'manifest.json'))
    else:
        coordinate()
