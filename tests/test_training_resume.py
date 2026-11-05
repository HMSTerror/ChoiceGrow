"""Compare uninterrupted training with disk-restored training on actual model updates."""
from contextlib import redirect_stdout
from copy import deepcopy
import io
import json
from pathlib import Path
import random
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch

import run
from data import write_json
from train import Trainer
from training_state import TrainingStateStore, validate_resume_config
from tests.test_equivalence import artifact, settings


def prepare_run(folder, config):
    (folder / "prepared").mkdir(parents=True)
    values = {
        "seq": np.array([[12,12,0,1], [12,2,3,4], [4,5,6,7], [12,8,9,10], [0,2,4,6]]),
        "length": np.array([2,3,4,3,4]), "target": np.array([2,5,8,11,7]),
    }
    for split in ("train", "val"):
        np.savez(folder / "prepared" / f"{split}.npz", **values)
    write_json(folder / "prepared" / "metadata.json", {"item_count":12})
    write_json(folder / "resolved_config.json", config)
    torch.save(artifact(), folder / "structure.pt")


def nested_equal(test, a, b):
    if isinstance(a, torch.Tensor):
        test.assertTrue(torch.equal(a, b))
    elif isinstance(a, dict):
        test.assertEqual(a.keys(), b.keys())
        for key in a:
            nested_equal(test, a[key], b[key])
    elif isinstance(a, (list, tuple)):
        test.assertEqual(len(a), len(b))
        for left, right in zip(a, b):
            nested_equal(test, left, right)
    else:
        test.assertEqual(a, b)


class ResumeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def test_actual_fixed_and_learned_models_match_after_resume(self):
        for learned in (False, True):
            with self.subTest(learned=learned), tempfile.TemporaryDirectory() as temporary:
                root=Path(temporary);whole=root/'whole';split=root/'split'
                config=settings(learned)
                config.update(device='cpu',threads=2)
                config['training'].update(epochs=4,patience=100,validation_interval=2)
                short=deepcopy(config);short['training']['epochs']=2
                prepare_run(whole, config);prepare_run(split, short)
                with redirect_stdout(io.StringIO()):
                    random.seed(100)
                    Trainer(whole, config, torch.device('cpu')).fit()
                    random.seed(100)
                    Trainer(split, short, torch.device('cpu')).fit()
                    # Simulate an interrupted output write newer than the full snapshot.
                    with (split/'choice_log.jsonl').open('a',encoding='utf-8') as log:
                        log.write(json.dumps({'epoch':3,'validation':{'ndcg@10':999.}})+'\n')
                    write_json(split/'report.json',{'stale':True})
                    torch.save({'invalid_newer_best':True}, split/'choice_best.pt')
                    Trainer(split, config, torch.device('cpu')).fit(resume=True)
                expected=torch.load(whole/'training_state.pt',map_location='cpu',weights_only=True)
                resumed=torch.load(split/'training_state.pt',map_location='cpu',weights_only=True)
                for key in ('live_model','ema_model','optimizer','process','ema_updates','torch_rng',
                            'shuffle_rng','numpy_rng','python_rng','best_validation','best_epoch'):
                    nested_equal(self, expected[key], resumed[key])
                self.assertEqual((whole/'choice_log.jsonl').read_bytes(),(split/'choice_log.jsonl').read_bytes())
                best_a=torch.load(whole/'choice_best.pt',map_location='cpu',weights_only=True)
                best_b=torch.load(split/'choice_best.pt',map_location='cpu',weights_only=True)
                nested_equal(self,best_a['model'],best_b['model'])
                if learned:
                    nested_equal(self,best_a['rate_gate'],best_b['rate_gate'])
                self.assertFalse((split/'report.json').exists())
                self.assertTrue(list((split/'resume_history').glob('*/report.json')))

    def test_rejects_ema_only_state_and_non_stopping_changes(self):
        with tempfile.TemporaryDirectory() as temporary:
            with self.assertRaisesRegex(FileNotFoundError,'EMA-only'):
                TrainingStateStore(temporary).read(settings())
        config=settings();changed=deepcopy(config)
        changed['training'].update(epochs=1670,patience=300)
        validate_resume_config(config,changed)
        changed['training']['learning_rate']=.2
        with self.assertRaisesRegex(ValueError,'only allows'):
            validate_resume_config(config,changed)

    def test_cli_resume_uses_saved_config_and_skips_artifact_stages(self):
        with tempfile.TemporaryDirectory() as temporary:
            output=Path(temporary)
            config=run.load_config(run.ROOT/'config.json')
            config.update(dataset='Beauty',seed=102,device='cpu',method='choicegrow',
                          experiment='neighbor_community_root',partition='native_infomap_leaf_communities_and_catalog')
            write_json(output/'resolved_config.json',config)
            argv=['run.py','--resume','--output',str(output),'--epochs','2000','--patience','300']
            with patch('sys.argv',argv), patch.object(run,'TrainingStateStore') as store, \
                    patch.object(run,'Trainer') as trainer, patch.object(run,'Evaluator') as evaluator, \
                    patch.object(run,'prepare') as prepare,patch.object(run,'build_artifacts') as artifacts, \
                    redirect_stdout(io.StringIO()):
                run.main()
            trainer.return_value.fit.assert_called_once_with(resume=True)
            stored=trainer.call_args.args[1]
            self.assertEqual((stored['dataset'],stored['seed']),('Beauty',102))
            self.assertEqual((stored['training']['epochs'],stored['training']['patience']),(2000,300))
            store.return_value.read.assert_called_once_with(stored)
            evaluator.return_value.test_best.assert_called_once_with(output)
            prepare.assert_not_called();artifacts.assert_not_called()


if __name__=='__main__':
    unittest.main()
