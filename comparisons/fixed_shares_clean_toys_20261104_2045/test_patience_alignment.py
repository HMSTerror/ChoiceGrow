"""Check that retrospective stopping cannot use validation improvements after stopping."""
import unittest
from patience_alignment import replay_stopping


class PatienceAlignmentTests(unittest.TestCase):
    def test_first_stop_excludes_later_improvement_and_ties_do_not_reset(self):
        rows = [{'epoch': epoch, 'validation': {'ndcg@10': score}}
                for epoch, score in ((10,.1), (20,.1), (30,.09), (40,.3))]
        result = replay_stopping(rows, 20)
        self.assertEqual(result['stop_epoch'], 30)
        self.assertEqual(result['best_validation'], {'epoch':10, 'ndcg@10':.1})

    def test_stop_is_checked_only_at_validation_not_unvalidated_epochs(self):
        rows = [{'epoch':10, 'validation':{'ndcg@10':.1}}, {'epoch':25, 'validation':None},
                {'epoch':30, 'validation':{'ndcg@10':.09}}]
        self.assertEqual(replay_stopping(rows, 15)['stop_epoch'], 30)
        self.assertFalse(replay_stopping(rows, 30)['stop_reached'])


if __name__ == '__main__':
    unittest.main()
