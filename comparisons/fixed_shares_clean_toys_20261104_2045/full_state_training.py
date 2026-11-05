"""Full-state training around the frozen numerical Trainer, with optional fixed horizon."""
# Note: ../../docs/decisions/implemented/feature/2026-11-04-fixed-shares.md
from itertools import count
import sys
import time

import experiment as p
from train import append_log, better
sys.path.append(str(p.PROJECT))
from training_state import TrainingStateStore


def restore_state(trainer, state):
    """Restore model, optimizer and RNGs through the main snapshot implementation."""
    TrainingStateStore(trainer.output).restore(trainer, state)


def save_full_state(trainer, epoch, best, best_epoch, best_seconds, elapsed_seconds):
    """Save each completed epoch, including its validation-selection history."""
    TrainingStateStore(trainer.output).save(trainer, epoch, best, best_epoch, best_seconds,
        elapsed_seconds, fixed_horizon=getattr(trainer, 'fixed_horizon', False))


class StatefulTrainer(p.Trainer):
    """Preserve numerical training and early stopping while saving resumable state."""
    fixed_horizon = False

    def fit(self, resume=False):
        settings = self.settings
        limit = settings['epochs']
        if self.fixed_horizon and limit <= 0:
            raise ValueError('A fixed-horizon rerun requires a positive epoch limit')
        store = TrainingStateStore(self.output)
        if resume:
            state = store.read(self.config, fixed_horizon=self.fixed_horizon)
            store.restore(self, state)
            store.restore_outputs(self, state)
            best, best_epoch = state['best_validation'], state['best_epoch']
            best_seconds, elapsed = state['selected_checkpoint_seconds'], state['elapsed_seconds']
            first_epoch = state['epoch'] + 1
        else:
            if any((self.output / name).exists() for name in
                   ('choice_best.pt', 'choice_newst.pt', 'choice_log.jsonl', 'training_state.pt')):
                raise FileExistsError('Training artifacts exist; use an explicit full-state resume')
            best, best_epoch, best_seconds, elapsed, first_epoch = None, 0, 0., 0., 1
        started = time.perf_counter()
        epochs = count(first_epoch) if limit == 0 else range(first_epoch, limit + 1)
        for epoch in epochs:
            mean_loss = self.train_epoch()
            metrics, shares = None, None
            if epoch % settings['validation_interval'] == 0 or epoch == limit:
                metrics = self.evaluator.evaluate(self.ema.model, self.process, self.validation,
                                                   'val', settings['validation_streams'])
                shares = self.evaluator.rate_statistics(self.process, self.validation)
                if better(metrics, best):
                    best, best_epoch = metrics, epoch
                    best_seconds = elapsed + time.perf_counter() - started
                    self.checkpoints.save('choice_best.pt', self.ema.model, self.process,
                                          epoch, metrics, self.ema.updates)
            self.checkpoints.save('choice_newst.pt', self.ema.model, self.process,
                                  epoch, metrics, self.ema.updates)
            append_log(self.output / 'choice_log.jsonl', {
                'stage': 'choice', 'epoch': epoch, 'score_entropy': mean_loss,
                'validation': metrics, 'best_epoch': best_epoch, 'rate_shares': shares})
            save_full_state(self, epoch, best, best_epoch, best_seconds, elapsed + time.perf_counter() - started)
            if not self.fixed_horizon and metrics is not None and epoch - best_epoch >= settings['patience']:
                break
        stop_reason = ('fixed_epoch_limit' if self.fixed_horizon else
                       'early_stopping' if epoch - best_epoch >= settings['patience'] else 'epoch_limit')
        report = {
            'epochs_completed': epoch, 'best_epoch': best_epoch, 'best_validation': best,
            'selection_split': 'val', 'objective': 'score_entropy', 'epoch_limit': limit,
            'stop_reason': stop_reason, 'early_stopping_enabled': not self.fixed_horizon,
            'configured_patience': settings['patience'],
            'selected_checkpoint_seconds': best_seconds, 'elapsed_seconds': elapsed + time.perf_counter() - started,
            'dynamic_rate_gate': self.process.generator.rate_policy.is_learned, 'gate_ema': False,
            'fresh_rerun': not resume, 'full_training_state': 'training_state.pt',
            'full_state_interval': 1, 'resumed_from_epoch': first_epoch - 1 if resume else None,
        }
        p.write_json(self.output / 'training.json', report)
        return report


class FixedHorizonTrainer(StatefulTrainer):
    """Run exactly config.training.epochs while retaining original best selection."""
    fixed_horizon = True
