"""Test an immutable Step 1 validation-best checkpoint without touching live training."""
import argparse
import io
import os
from pathlib import Path
import shutil
import time
import traceback
import experiment as p


def freeze(path):
    """Capture one complete save even if training writes a new best concurrently."""
    for attempt in range(8):
        before=path.stat()
        content=path.read_bytes()
        after=path.stat()
        if (before.st_size,before.st_mtime_ns)!=(after.st_size,after.st_mtime_ns):
            time.sleep(.2)
            continue
        try:
            state=p.torch.load(io.BytesIO(content),map_location='cpu',weights_only=True)
        except (OSError,EOFError,RuntimeError):
            if attempt==7:
                raise
            time.sleep(.2)
            continue
        return content,state
    raise RuntimeError('The best checkpoint changed during every capture attempt')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run',choices=('1A','1B'),default='1B')
    args=parser.parse_args()
    run_name,gamma={'1A':('1A_clean_g002',.02),'1B':('1B_clean_g010',.10)}[args.run]
    source=p.ROOT/'runs/step1'/run_name
    stamp=p.datetime.now().astimezone().strftime('%Y%m%d_%H%M%S')
    output=p.ROOT/'runs/snapshot_tests'/f'{args.run}_{stamp}'
    output.mkdir(parents=True,exist_ok=False)
    p.write_json(output/'status.json',{'state':'capturing','pid':os.getpid(),'started_at':p.now()})
    try:
        content,state=freeze(source/'choice_best.pt')
        config=p.read(source/'resolved_config.json')
        assert state['config']==config
        assert config['destination']['neighbor_mix']==gamma and config['neighbor_statistics']=='nn_clean'
        assert config['diffusion']['fixed_shares']==[1.] and 'rate_gate' not in state
        (output/'choice_best.pt').write_bytes(content)
        shutil.copy2(source/'structure.pt',output/'structure.pt')
        p.write_json(output/'resolved_config.json',config)
        p.write_json(output/'data_source.json',{'path':str(p.prepared_root(source).resolve())})
        snapshot={'source':str(source),'captured_at':p.now(),'checkpoint_epoch':state['epoch'],
            'checkpoint_validation':state['validation'],'checkpoint_sha256':p.hashlib.sha256(content).hexdigest(),
            'evaluation_split':'test','streams':3,'selection_split':'val',
            'training_settings_changed':False,'used_for_step1_selection':False}
        p.write_json(output/'snapshot.json',snapshot)
        print(p.json.dumps({'event':'captured','output':str(output),**snapshot}),flush=True)
        del content,state
        p.torch.set_num_threads(config['threads'])
        if config['tf32']:
            p.torch.set_float32_matmul_precision('high');p.torch.backends.cudnn.allow_tf32=True
        p.write_json(output/'status.json',{'state':'evaluating','pid':os.getpid(),'updated_at':p.now()})
        report=p.Evaluator(config,p.torch.device('cuda:0')).test_best(output)
        assert report['best_epoch']==snapshot['checkpoint_epoch'] and report['test']['streams']==3
        tail=p.cached_metrics(output,'test',output/'recommendations.npz')
        for key in p.METRICS:
            assert abs(tail[key]-report['test'][key])<1e-7
        p.write_json(output/'long_tail.json',tail)
        baseline=p.read(p.BASELINE/'report.json')
        baseline_tail=p.cached_metrics(p.BASELINE,'test',p.BASELINE/'recommendations.npz')
        result={'snapshot':snapshot,'test':report['test'],'long_tail':tail,
            'baseline':{'source':str(p.BASELINE),'checkpoint_epoch':baseline['best_epoch'],
                'test':baseline['test'],'long_tail':baseline_tail},
            'overall_ndcg10_change':report['test']['ndcg@10']-baseline['test']['ndcg@10'],
            'overall_ndcg10_relative_percent':100*(report['test']['ndcg@10']/baseline['test']['ndcg@10']-1),
            'tail_ndcg10_change':tail['target_groups']['tail']['ndcg@10']-baseline_tail['target_groups']['tail']['ndcg@10'],
            'tail_ndcg10_relative_percent':100*(tail['target_groups']['tail']['ndcg@10']/baseline_tail['target_groups']['tail']['ndcg@10']-1)}
        p.write_json(output/'comparison.json',result)
        p.write_json(output/'status.json',{'state':'complete','pid':os.getpid(),'finished_at':p.now(),
            'checkpoint_epoch':snapshot['checkpoint_epoch']})
        print(p.json.dumps({'event':'complete','output':str(output),'result':result}),flush=True)
    except BaseException:
        p.write_json(output/'status.json',{'state':'failed','pid':os.getpid(),'updated_at':p.now(),
            'traceback':traceback.format_exc()})
        raise


if __name__=='__main__':
    main()
