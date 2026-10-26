import io
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
from urllib.error import URLError
from zipfile import ZipFile

from src.imdb import IMDB


class IMDBDownloadTests(unittest.TestCase):
    def setUp(self):
        self.tmp_dir = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp_dir.cleanup)
        # Exercise downloading without triggering PyG's processing of tensors.
        self.dataset = IMDB.__new__(IMDB)
        self.dataset.root = self.tmp_dir.name
        self.raw_dir = Path(self.dataset.raw_dir)
        self.raw_dir.mkdir()

    def archive_bytes(self, prefix=''):
        buffer = io.BytesIO()
        with ZipFile(buffer, 'w') as archive:
            for name in self.dataset.raw_file_names:
                archive.writestr(prefix + name, name.encode())
        return buffer.getvalue()

    def assert_raw_files(self):
        for name in self.dataset.raw_file_names:
            self.assertEqual((self.raw_dir / name).read_bytes(), name.encode())

    def test_cached_and_downloaded_html_fall_back_without_poisoning_cache(self):
        cached_zip = self.raw_dir / 'IMDB_processed.zip'
        html = b'<html><title>Dropbox - Link Temporarily Disabled</title></html>'
        cached_zip.write_bytes(html)
        with patch('src.imdb.urlopen', return_value=io.BytesIO(html)), \
                patch.object(self.dataset, '_build_from_source') as rebuild:
            self.dataset.download()
        rebuild.assert_called_once_with()
        self.assertEqual(list(self.raw_dir.iterdir()), [])

    def test_nested_cached_archive_loads_offline(self):
        (self.raw_dir / 'IMDB_processed.zip').write_bytes(
            self.archive_bytes('IMDB_processed/')
        )
        with patch('src.imdb.urlopen') as download:
            self.dataset.download()
        download.assert_not_called()
        self.assert_raw_files()

    def test_valid_download_produces_raw_files(self):
        with patch('src.imdb.urlopen', return_value=io.BytesIO(self.archive_bytes())), \
                patch.object(self.dataset, '_build_from_source') as rebuild:
            self.dataset.download()
        rebuild.assert_not_called()
        self.assert_raw_files()
        self.assertFalse((self.raw_dir / 'IMDB_processed.zip').exists())

    def test_interrupted_download_falls_back(self):
        with patch('src.imdb.urlopen', side_effect=URLError('interrupted')), \
                patch.object(self.dataset, '_build_from_source') as rebuild:
            self.dataset.download()
        rebuild.assert_called_once_with()
        self.assertEqual(list(self.raw_dir.iterdir()), [])

    def test_unrelated_zip_falls_back(self):
        buffer = io.BytesIO()
        with ZipFile(buffer, 'w') as archive:
            archive.writestr('unrelated.txt', b'wrong dataset')
        with patch('src.imdb.urlopen', return_value=io.BytesIO(buffer.getvalue())), \
                patch.object(self.dataset, '_build_from_source') as rebuild:
            self.dataset.download()
        rebuild.assert_called_once_with()
        self.assertEqual(list(self.raw_dir.iterdir()), [])

    def test_corrupt_member_does_not_leave_partial_raw_files(self):
        payload = bytearray(self.archive_bytes())
        with ZipFile(io.BytesIO(payload)) as archive:
            member = archive.getinfo(self.dataset.raw_file_names[-1])
            data_offset = member.header_offset + 30 + len(member.filename.encode())
        payload[data_offset] ^= 0xFF
        with patch('src.imdb.urlopen', return_value=io.BytesIO(payload)), \
                patch.object(self.dataset, '_build_from_source') as rebuild:
            self.dataset.download()
        rebuild.assert_called_once_with()
        self.assertEqual(list(self.raw_dir.iterdir()), [])

    def test_unavailable_sources_explain_manual_recovery(self):
        with patch('src.imdb.urlopen', side_effect=URLError('offline')):
            with self.assertRaisesRegex(RuntimeError, 'Both IMDB download sources') as error:
                self.dataset.download()
        self.assertIn(str(self.raw_dir), str(error.exception))
        self.assertEqual(list(self.raw_dir.iterdir()), [])


if __name__ == '__main__':
    unittest.main()
