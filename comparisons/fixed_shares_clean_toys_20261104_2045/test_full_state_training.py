"""Check fixed-horizon stopping and numerical continuation after a full snapshot."""
from pathlib import Path
import random
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import experiment as p
from data import loader
from train import EMA
from full_state_training import FixedHorizonTrainer, restore_state, save_full_state


def tiny_trainer(folder):
    model = p.torch.nn.Sequential(p.torch.nn.Linear(2, 1), p.torch.nn.Dropout(.25))
    values = {'seq': p.torch.arange(14, dtype=p.torch.float32).reshape(7, 2) / 10,
              'length': p.torch.ones(7, dtype=p.torch.long),
              'target': p.torch.arange(7, dtype=p.torch.float32) / 10}
    return SimpleNamespace(
        output=folder, device=p.torch.device('cpu'), config={'seed': 100}, model=model, ema=EMA(model, .99),
        optimizer=p.torch.optim.AdamW(model.parameters(), lr=.01),
        process=p.torch.nn.Linear(1, 1),
        batches=loader(values, 3, shuffle=True, seed=100),
        checkpoints=SimpleNamespace(best_path=folder / 'choice_best.pt'))


def tiny_epoch(trainer):
    losses=[]
    for sequence, _, target in trainer.batches:
        perturbation=p.torch.rand(len(target)) + p.np.random.random() + random.random()
        loss=((trainer.model(sequence).squeeze(-1)-target-perturbation)**2).mean()
        trainer.optimizer.zero_grad(set_to_none=True)
        loss.backward();trainer.optimizer.step();trainer.ema.update(trainer.model)
        losses.append(float(loss.detach()))
    return losses


class FullStateTests(unittest.TestCase):
    def test_disk_round_trip_preserves_next_shuffled_epoch_and_updates(self):
        p.torch.manual_seed(100);p.np.random.seed(100);random.seed(100)
        with tempfile.TemporaryDirectory() as temporary:
            folder=Path(temporary)
            original=tiny_trainer(folder)
            tiny_epoch(original)
            p.torch.save({'model': original.ema.model.state_dict(), 'epoch': 1},
                         original.checkpoints.best_path)
            save_full_state(original, 1, {'ndcg@10': .1}, 1, .5, 1.)
            saved=p.torch.load(folder/'training_state.pt', map_location='cpu', weights_only=True)
            expected_losses=tiny_epoch(original)
            restored=tiny_trainer(folder)
            restore_state(restored, saved)
            self.assertEqual(tiny_epoch(restored), expected_losses)
            for key, value in original.model.state_dict().items():
                self.assertTrue(p.torch.equal(value, restored.model.state_dict()[key]))
            for key, value in original.ema.model.state_dict().items():
                self.assertTrue(p.torch.equal(value, restored.ema.model.state_dict()[key]))
            self.assertEqual(original.ema.updates, restored.ema.updates)
            self.assertEqual(saved['best_checkpoint']['epoch'], 1)
            self.assertTrue(saved['epoch_boundary'])

    def test_fixed_horizon_does_not_stop_on_validation_plateau(self):
        with tempfile.TemporaryDirectory() as temporary:
            trainer=object.__new__(FixedHorizonTrainer)
            trainer.output=Path(temporary)
            trainer.settings={'epochs':5,'patience':1,'validation_interval':1,'validation_streams':1}
            trainer.train_epoch=Mock(return_value=1.)
            trainer.ema=SimpleNamespace(model=object(),updates=0)
            trainer.process=SimpleNamespace(generator=SimpleNamespace(rate_policy=SimpleNamespace(is_learned=False)))
            trainer.validation=object()
            trainer.evaluator=SimpleNamespace(
                evaluate=Mock(side_effect=[{'ndcg@10':v} for v in (.2,.1,.1,.1,.1)]),
                rate_statistics=Mock(return_value=None))
            trainer.checkpoints=SimpleNamespace(save=Mock())
            with patch('full_state_training.save_full_state') as save:
                report=trainer.fit()
            self.assertEqual(report['epochs_completed'],5)
            self.assertEqual(report['best_epoch'],1)
            self.assertFalse(report['early_stopping_enabled'])
            self.assertEqual(report['stop_reason'],'fixed_epoch_limit')
            self.assertEqual(save.call_count,5)


if __name__=='__main__':
    p.torch.set_num_threads(2)
    unittest.main()
