"""Check prefix-label semantics, padding, repeated events, and the one-history-item case."""
from pathlib import Path
import sys
import unittest

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent / "code"))
from artifacts import prefix_target_counts


class PrefixBackgroundTests(unittest.TestCase):
    def test_each_prefix_contributes_its_next_label_once(self):
        counts = prefix_target_counts(torch.tensor([[0, 1, 2, 3]]), torch.tensor([4]), 5)
        torch.testing.assert_close(counts, torch.tensor([0, 1, 1, 1, 1]), rtol=0, atol=0)

    def test_padding_is_excluded_and_repeated_targets_are_retained(self):
        counts = prefix_target_counts(torch.tensor([[5, 0, 1, 0], [5, 5, 5, 2]]),
                                     torch.tensor([1, 3]), 5)
        torch.testing.assert_close(counts, torch.tensor([1, 2, 0, 1, 0]), rtol=0, atol=0)

    def test_single_history_item_only_contributes_the_final_target(self):
        counts = prefix_target_counts(torch.tensor([[4, 4, 0], [4, 4, 1]]), torch.tensor([2, 2]), 4)
        torch.testing.assert_close(counts, torch.tensor([0, 0, 2, 0]), rtol=0, atol=0)


if __name__ == "__main__":
    unittest.main()
