import sys
import unittest
import csv
import subprocess
import tempfile
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from resolve_copy_carriers import select_copy_row


class CopyJoinTests(unittest.TestCase):
    def setUp(self):
        self.req = dict(site_id='sp.E1.copy2', carrier_assembly='right',
                        source_event='sp.E1', carrier_empty_flank120='ACGT', insert_md5='abc')
        self.row = dict(db_id='sp.E1.copy2', carrier='right', source_event='sp.E1',
                        flank='ACGT', insert_md5='abc', contig='c', elem_start='10', elem_end='20')

    def test_collision_does_not_take_first_wrong_carrier(self):
        match, reason = select_copy_row(self.req, [('old', {**self.row, 'carrier': 'wrong'}),
                                                  ('new', self.row)])
        self.assertEqual(match[0], 'new')
        self.assertFalse(reason)

    def test_wrong_flank_is_unresolved(self):
        match, reason = select_copy_row(self.req, [('file', {**self.row, 'flank': 'AAAA'})])
        self.assertIsNone(match)
        self.assertEqual(reason, 'no_copy_row_matches_request_identity')

    def test_distinct_coordinates_are_ambiguous(self):
        match, reason = select_copy_row(self.req, [('a', self.row),
                                                  ('b', {**self.row, 'contig': 'other'})])
        self.assertIsNone(match)
        self.assertEqual(reason, 'ambiguous_matching_copy_coordinates')

    def test_duplicate_equivalent_coordinates_are_one(self):
        match, reason = select_copy_row(self.req, [('a', self.row), ('b', dict(self.row))])
        self.assertIsNotNone(match)
        self.assertFalse(reason)

    def test_main_exports_requested_carrier_and_unknown_orientation(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / 'copysites').mkdir()
            (root / 'genomes').mkdir()
            req = {**self.req, 'bag_id': 'bag', 'species': 'abaum'}
            manifest = root / 'requests.tsv'
            with manifest.open('w', newline='') as handle:
                writer = csv.DictWriter(handle, fieldnames=list(req), delimiter='\t')
                writer.writeheader()
                writer.writerow(req)
            with (root / 'copysites/abaum_sites.tsv').open('w', newline='') as handle:
                writer = csv.DictWriter(handle, fieldnames=list(self.row), delimiter='\t')
                writer.writeheader()
                writer.writerows([{**self.row, 'carrier': 'wrong'}, self.row])
            (root / 'genomes/right.fna').write_text('>c\n' + 'ACGT' * 100 + '\n')
            subprocess.run([sys.executable, str(Path(__file__).resolve().parents[1] / 'resolve_copy_carriers.py'),
                            '--manifest', str(manifest), '--discovery-root', str(root),
                            '--outdir', str(root / 'out')], check=True, capture_output=True)
            with (root / 'out/copy_carrier_resolved.tsv').open() as handle:
                result = list(csv.DictReader(handle, delimiter='\t'))
            self.assertEqual(len(result), 1)
            self.assertEqual(result[0]['carrier_assembly'], 'right')
            self.assertTrue((root / 'out/copy_carrier_slices.fna').read_text().splitlines()[0].endswith('|unknown'))


if __name__ == '__main__':
    unittest.main()
