"""CPU checks for the authorized factors, continuous slots and batch barriers."""
from copy import deepcopy
import unittest

import run_suite as s


class SuiteTests(unittest.TestCase):
    def test_bn_counts_successors_once_without_expanding_training_rows(self):
        sequence = s.p.np.array([[5, 0, 1, 2, 3]])
        target = s.p.np.array([4])
        s.p.np.testing.assert_array_equal(s.background_counts(sequence, target, 5, 'bn'), [0, 1, 1, 1, 1])
        s.p.np.testing.assert_array_equal(s.background_counts(sequence, target, 5, 'bo'), [0, 0, 0, 0, 1])

    def test_clean_neighbors_exclude_final_label_but_keep_history_pairs(self):
        rows = s.p.count_transitions(s.p.np.array([[5, 0, 1, 2, 3]]), s.p.np.array([4]), 5, 'nn_clean')
        self.assertEqual(rows[0][1], 1.)
        self.assertEqual(rows[0][2], .5)
        self.assertEqual(rows[0][3], 1. / 3)
        self.assertFalse(any(4 in row for row in rows.values()))

    def test_slot_fills_without_waiting_for_pair(self):
        jobs = [{'batch': 1, 'state': state, 'id': i} for i, state in
                enumerate(('complete', 'running', 'queued', 'queued'))]
        jobs.append({'batch': 2, 'state': 'queued', 'id': 4})
        self.assertEqual([j['id'] for j in s.eligible_jobs(jobs)], [2, 3])

    def test_evaluation_is_inside_batch_barrier(self):
        jobs = [{'batch': 1, 'state': 'running'}, {'batch': 2, 'state': 'queued'}]
        self.assertEqual(s.eligible_jobs(jobs), [])
        jobs[0]['state'] = 'complete'
        self.assertEqual(s.eligible_jobs(jobs), [jobs[1]])

    def test_protocol_rejects_eta_style_training_changes_or_patience_drift(self):
        job = s.job_spec(1, 'Sports', 100, 'hs')
        config = s.p.read(s.BASELINE / 'resolved_config.json')
        config['diffusion'].update(fixed_shares=[.1, .9],
            level_rates=[config['diffusion']['total_budget'] * x for x in (.1, .9)])
        s.validate_protocol(config, job)
        changed = deepcopy(config); changed['training']['patience'] = 150
        with self.assertRaises(AssertionError): s.validate_protocol(changed, job)
        changed = deepcopy(config); changed['diffusion']['dynamic_rate_gate'] = True
        with self.assertRaises(AssertionError): s.validate_protocol(changed, job)

    def test_sample_sd_uses_training_seeds_and_keeps_seed100(self):
        rows = []
        for seed, value in zip((100, 101, 102), (.1, .2, .3)):
            metrics = {key: value for key in s.p.METRICS}
            tail = {'target_groups': {'tail': metrics}, 'tail_exposure@10': value}
            rows.append({'seed': seed, 'validation': metrics, 'test': metrics,
                         'validation_tail': tail, 'test_tail': tail})
        result = s.mean_sd(rows)['test.ndcg@10']
        self.assertAlmostEqual(result['mean'], .2)
        self.assertAlmostEqual(result['sd'], .1)
        self.assertEqual(result['n_training_seeds'], 3)
        with self.assertRaises(AssertionError): s.mean_sd(rows[1:] + [rows[1]])


if __name__ == '__main__':
    unittest.main()
