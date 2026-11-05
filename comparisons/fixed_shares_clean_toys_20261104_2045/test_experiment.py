"""Check validation-only selection and conditional tail ranking semantics."""
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import experiment as p
import run_experiments as runner


class ExperimentTests(unittest.TestCase):
    def test_selection_uses_full_validation_best_and_ignores_test(self):
        with tempfile.TemporaryDirectory() as temporary:
            jobs=[]
            for i,(score,test_score) in enumerate(((.02,.99),(.03,.98),(.04,0.))):
                folder=Path(temporary)/str(i);folder.mkdir()
                p.write_json(folder/'training.json',{'best_epoch':10,'best_validation':{'ndcg@10':score}})
                rows=[{'epoch':10,'validation':{'ndcg@10':score}},
                      {'epoch':20,'validation':{'ndcg@10':score-.01}}]
                (folder/'choice_log.jsonl').write_text('\n'.join(p.json.dumps(v) for v in rows)+'\n',encoding='utf-8')
                p.write_json(folder/'report.json',{'test':{'ndcg@10':test_score}})
                jobs.append({'id':str(i),'stage':1,'output':str(folder),'neighbor':'nn' if i==2 else 'nn_clean',
                    'gamma':.1 if i==1 else .02,'reused_training':i==2})
            selected=runner.select_neighbors(jobs)
            self.assertEqual(selected['winner']['id'],'2')
            self.assertFalse(selected['test_used'])
            self.assertEqual(selected['winner']['best_epoch'],10)

    def test_tail_ndcg_is_full_catalog_target_subset_not_tail_reranking(self):
        with tempfile.TemporaryDirectory() as temporary:
            folder=Path(temporary)
            ranked=p.np.array([[[0,1,2,3,4,5,6,7,8,9],[0,1,2,3,4,5,6,7,8,9]]])
            target=p.np.array([10,1]);tail=p.np.zeros(12,dtype=bool);tail[10:]=True
            cache=folder/'val.npz';p.np.savez(cache,top10=ranked,target=target)
            with patch.object(p,'tail_mask',return_value=(tail,12)),patch.object(p,'load_split',
                    return_value={'target':p.torch.from_numpy(target)}):
                metrics=p.cached_metrics(folder,'val',cache)
            self.assertEqual(metrics['target_groups']['tail']['ndcg@10'],0.)
            self.assertAlmostEqual(metrics['ndcg@10'],.5/p.np.log2(3))
            self.assertEqual(metrics['target_groups']['tail']['examples'],1)
            self.assertEqual(metrics['tail_exposure@10'],0.)

    def test_step2_grid_and_stage_barrier(self):
        identities=[runner.job_spec(f'{kind}/{share}',2,kind,share)
            for kind in ('hs','hr') for share in (.3,.5,.8)]
        self.assertEqual(len(identities),6)
        self.assertTrue(all(job['state']=='blocked_on_step1' for job in identities))
        self.assertTrue(all(job['seed']==100 for job in identities))

    def test_parallel_step1_admission_and_serial_step2(self):
        jobs=[{'id':'A','stage':1,'state':'running'},
              {'id':'B','stage':1,'state':'queued'},
              {'id':'C','stage':1,'state':'evaluation_pending'},
              {'id':'hs','stage':2,'state':'queued'},
              {'id':'hr','stage':2,'state':'queued'}]
        manifest={'jobs':jobs}
        control={'max_suite_workers':2,'step1_max_workers':2,'step2_max_workers':1}
        self.assertEqual([j['id'] for j in runner.available_jobs(manifest,control)],['B'])
        jobs[1]['state']='running'
        self.assertEqual(runner.available_jobs(manifest,control),[])
        for job in jobs[:3]:job['state']='complete'
        self.assertEqual([j['id'] for j in runner.available_jobs(manifest,control)],['hs'])
        jobs[3]['state']='running'
        self.assertEqual(runner.available_jobs(manifest,control),[])

    def test_coordinator_adopts_live_fit_without_launching(self):
        with tempfile.TemporaryDirectory() as temporary:
            job={'id':'A','state':'running','output':temporary,'spec':str(Path(temporary)/'job.json')}
            with patch.object(runner,'matching_worker',return_value=[12345]),patch.object(runner,
                    'process_record',return_value={'pid':12345,'created':7.}):
                self.assertFalse(runner.synchronize_jobs({'jobs':[job]}))
            self.assertEqual(job['pid'],12345)
            self.assertEqual(job['state'],'running')

    def test_step2_pairs_wait_for_both_reports_before_next_share(self):
        jobs=[{'id':f'{mode}_{share}','stage':2,'community_share':share,'state':'queued'}
              for share in (.3,.5,.8) for mode in ('hs','hr')]
        manifest={'jobs':jobs}
        control={'automatic_step2':True,'step2_pair_by_share':True,
                 'max_suite_workers':2,'step2_max_workers':2}
        self.assertEqual([j['id'] for j in runner.available_jobs(manifest,control)],['hs_0.3','hr_0.3'])
        jobs[0]['state']='running'
        self.assertEqual([j['id'] for j in runner.available_jobs(manifest,control)],['hr_0.3'])
        jobs[0]['state']='complete';jobs[1]['state']='running'
        self.assertEqual(runner.available_jobs(manifest,control),[])
        jobs[1]['state']='complete'
        self.assertEqual([j['id'] for j in runner.available_jobs(manifest,control)],['hs_0.5','hr_0.5'])
        jobs[2]['state']='failed'
        self.assertEqual([j['id'] for j in runner.available_jobs(manifest,control)],['hr_0.5'])

    def test_manual_decision_hold_prevents_step2_selection_and_launch(self):
        jobs=[{'id':'A','stage':1,'state':'complete'},
              {'id':'hs','stage':2,'state':'queued'}]
        manifest={'jobs':jobs,'step1_selection':None,'frozen_code_sha256':{}}
        control={'automatic_step2':False}
        self.assertEqual(runner.available_jobs(manifest,control),[])
        with tempfile.TemporaryDirectory() as temporary:
            output=Path(temporary)
            p.write_json(output/'manifest.json',manifest)
            p.write_json(output/'execution_control.json',control)
            with patch.object(runner,'OUTPUT',output),patch.object(runner,'code_hashes',return_value={}), \
                    patch.object(runner,'synchronize_jobs',return_value=False), \
                    patch.object(runner,'select_neighbors') as select, \
                    patch.object(p,'prepare_job') as prepare, \
                    patch.object(runner.subprocess,'Popen') as launch, \
                    patch.object(runner,'save') as save,patch.object(runner,'summarize'):
                runner.coordinate()
                save.assert_called_once_with(manifest | {'max_suite_workers':2},'awaiting_user_decision')
                select.assert_not_called()
                prepare.assert_not_called()
                launch.assert_not_called()


if __name__=='__main__':
    p.torch.set_num_threads(2)
    unittest.main()
