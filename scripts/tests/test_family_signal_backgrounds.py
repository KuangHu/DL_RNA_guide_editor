import unittest
from collections import Counter
from scripts.family_signal_backgrounds import exact_dinuc_shuffle
from scripts.analyze_family_signal import auc


class FamilyBackgroundTests(unittest.TestCase):
    def test_exact_counts_length_and_endpoints_across_seeds(self):
        for sequence in ['', 'A', 'AAAAAA', 'ATATATAT', 'ACGTACCTAGCGTACGATCGAAATTTCCCGGG'*4]:
            expected=Counter(zip(sequence,sequence[1:]))
            for seed in range(30):
                with self.subTest(sequence=sequence[:12],seed=seed):
                    shuffled=exact_dinuc_shuffle(sequence,seed)
                    self.assertEqual(len(shuffled),len(sequence))
                    self.assertEqual(Counter(shuffled),Counter(sequence))
                    self.assertEqual(Counter(zip(shuffled,shuffled[1:])),expected)
                    if sequence:
                        self.assertEqual(shuffled[0],sequence[0]);self.assertEqual(shuffled[-1],sequence[-1])

    def test_shuffle_is_reproducible_and_changes_nontrivial_sequences(self):
        sequence='ACGTACCTAGCGTACGATCGAAATTTCCCGGG'*4
        self.assertEqual(exact_dinuc_shuffle(sequence,12),exact_dinuc_shuffle(sequence,12))
        self.assertGreater(len({exact_dinuc_shuffle(sequence,i) for i in range(20)}),1)

    def test_auc_direction_and_ties(self):
        self.assertEqual(auc([.9,.8],[.5,.6]),1)
        self.assertEqual(auc([.5,.6],[.9,.8]),0)
        self.assertEqual(auc([.8],[.8]),.5)
        self.assertIsNone(auc([],[.8]))


if __name__=='__main__':unittest.main()
