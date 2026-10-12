import unittest
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from review_family_signal_context import terminal_repeat, junction_direct
from preprocess.alignment import revcomp


class ContextReviewTests(unittest.TestCase):
    def test_inverted_match_coordinates_reproduce_actual_sequence(self):
        end='ACGTCGATCGTAGCATGCTAGCGATCGATGCATGCA'
        insert=end+'N'*400+revcomp(end)
        hit=terminal_repeat(insert,True)
        self.assertEqual(hit['length'],len(end))
        self.assertEqual(insert[hit['left_lo']:hit['left_hi']],revcomp(insert[hit['right_lo']:hit['right_hi']]))
        self.assertEqual(hit['right_hi'],len(insert))

    def test_direct_and_inverted_are_not_interchanged(self):
        end='ACGTCGATCGTAGCATGCTAGCGATCGATGCATGCA'
        insert=end+'N'*400+end
        hit=terminal_repeat(insert)
        self.assertEqual(hit['length'],len(end))
        self.assertEqual(insert[hit['left_lo']:hit['left_hi']],insert[hit['right_lo']:hit['right_hi']])
        self.assertLess(terminal_repeat(insert,True)['length'],len(end))

    def test_exterior_direct_diagnostic_is_exact_and_junction_anchored(self):
        repeat='ACGTAGCTA'; insert='N'*80
        occupied='G'*30+repeat+insert+repeat+'C'*30
        lo=30+len(repeat);hi=lo+len(insert)
        self.assertEqual(junction_direct(occupied,lo,hi),len(repeat))


if __name__=='__main__':unittest.main()
