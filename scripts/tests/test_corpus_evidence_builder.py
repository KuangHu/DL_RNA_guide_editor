"""Regressions for source-derived boundaries, anchored MSA and locus frames."""
import unittest
from pathlib import Path

from scripts.build_v3_corpus_evidence import all_boundaries, md5, star_segment, choose_anchor
from scripts.v5a_framework.background_alignment import same_locus


class CorpusEvidenceTests(unittest.TestCase):
    def test_boundaries_enumerate_entire_domain(self):
        row = dict(inserted_len='1000', decomposition_method='exact',
                   n_equivalent_decompositions='501', inserted_md5=md5('A' * 1000),
                   insert_start_in_long_bp='500')
        intervals, mismatch = all_boundaries('A' * 1500, 'A' * 500, row)
        self.assertEqual((intervals[0], intervals[-1], len(intervals), mismatch),
                         ((0, 1000), (500, 1500), 501, 0))

    def test_upstream_boundary_count_disagreement_is_rejected(self):
        row = dict(inserted_len='1', decomposition_method='exact',
                   n_equivalent_decompositions='1', inserted_md5=md5('A'))
        with self.assertRaisesRegex(ValueError, 'boundary_count_disagreement'):
            all_boundaries('AAA', 'AA', row)

    def test_indel_decomposition_not_silently_hamming_aligned(self):
        with self.assertRaisesRegex(ValueError, 'unsupported_indel'):
            all_boundaries('ACGT', 'AC', dict(inserted_len='1'))

    def test_star_msa_preserves_sequences_and_shared_columns(self):
        sequences = {0: 'ACGTCAGTC', 1: 'ACGTTCAGTC', 2: 'ACGTCAGC'}
        rows, _ = star_segment(sequences, 0)
        self.assertEqual(len({len(s) for s in rows.values()}), 1)
        for idx, row in rows.items():
            self.assertEqual(row.replace('-', ''), sequences[idx])
        self.assertEqual(rows[0].replace('-', '')[:4], rows[1].replace('-', '')[:4])

    def test_paralog_ambiguity_cannot_form_unique_family(self):
        gene = dict(protein='M' + 'ACDEFGHIKLMNPQRSTVWY' * 5, lo=0, hi=300, strand='+')
        with self.assertRaisesRegex(ValueError, 'no_unique_conserved_orf_anchor'):
            choose_anchor({0: dict(genes=[gene, dict(gene)]), 1: dict(genes=[gene])})

    def test_seed_reference_coordinates_never_mixed_with_preevent_genome(self):
        a = dict(pre_event_genome='same', pre_event_contig='same', start=0, end=200,
                 reference_genome='seed_a', reference_contig='chr', reference_start=0, reference_end=200)
        b = dict(a, reference_genome='seed_b')
        self.assertFalse(same_locus(a, b))
        self.assertTrue(same_locus(a, dict(a)))
        self.assertFalse(same_locus(a, dict(b, reference_genome='')))

    def test_cds03597_actual_boundary_counts(self):
        from Bio import SeqIO
        root = Path(__file__).resolve().parents[2] / 'evidence/CDS03597/nc1_L14_fp45_v2'
        for idx, count in ((0, 131), (4, 108), (6, 155), (7, 66)):
            directory = root / f'site_{idx}'
            row = {}
            for line in (directory / 'loci_evidence.tsv').read_text().splitlines()[1:]:
                key, value = line.split('\t', 1)
                row[key] = value
            occupied = str(SeqIO.read(directory / 'occupied.fna', 'fasta').seq)
            empty = str(SeqIO.read(directory / 'empty.fna', 'fasta').seq)
            intervals, _ = all_boundaries(occupied, empty, row)
            self.assertEqual(len(intervals), count)


if __name__ == '__main__':
    unittest.main()
