import errno
import mmap
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch

from src.checkpoint import save_checkpoint


class CheckpointTests(unittest.TestCase):
    def setUp(self):
        self.tmp_dir = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp_dir.cleanup)
        self.root = Path(self.tmp_dir.name)
        self.destination = self.root / 'best_model.pth'
        self.old_state = {'weight': torch.tensor([1.0])}
        self.new_state = {'weight': torch.tensor([2.0])}

    def assert_saved_state(self, path, expected):
        actual = torch.load(path, weights_only=True)
        torch.testing.assert_close(actual['weight'], expected['weight'])

    def test_save_and_replace_round_trip(self):
        self.destination = self.root / 'nested' / 'best_model.pth'
        first_path = save_checkpoint(self.old_state, self.destination)
        self.assertEqual(first_path, self.destination)
        self.assert_saved_state(first_path, self.old_state)
        second_path = save_checkpoint(self.new_state, first_path)
        self.assertEqual(second_path, self.destination)
        self.assert_saved_state(second_path, self.new_state)
        self.assertEqual(list(second_path.parent.iterdir()), [second_path])

    def test_temporary_lock_is_retried(self):
        save_checkpoint(self.old_state, self.destination)
        real_replace = os.replace
        calls = []

        def replace_after_lock(source, destination):
            calls.append((source, destination))
            if len(calls) == 1:
                raise PermissionError('Checkpoint is temporarily locked')
            real_replace(source, destination)

        with patch('src.checkpoint.os.replace', side_effect=replace_after_lock), \
                patch('src.checkpoint.time.sleep'):
            saved_path = save_checkpoint(self.new_state, self.destination)
        self.assertEqual(len(calls), 2)
        self.assertEqual(saved_path, self.destination)
        self.assert_saved_state(saved_path, self.new_state)
        self.assertEqual(list(self.root.iterdir()), [self.destination])

    def test_persistent_mapping_lock_keeps_new_checkpoint_and_old_file(self):
        save_checkpoint(self.old_state, self.destination)
        original_bytes = self.destination.read_bytes()
        mapped_error = OSError('File has an open mapping')
        mapped_error.winerror = 1224
        with patch('src.checkpoint.os.replace', side_effect=mapped_error), \
                patch('src.checkpoint.time.sleep'):
            saved_path = save_checkpoint(self.new_state, self.destination)
        self.assertNotEqual(saved_path, self.destination)
        self.assertEqual(saved_path.suffix, '.pth')
        self.assertEqual(self.destination.read_bytes(), original_bytes)
        self.assert_saved_state(saved_path, self.new_state)

        # Future epochs must update the returned path, leaving the locked
        # original intact rather than generating a file on every improvement.
        next_path = save_checkpoint(self.old_state, saved_path)
        self.assertEqual(next_path, saved_path)
        self.assert_saved_state(next_path, self.old_state)
        self.assertEqual(self.destination.read_bytes(), original_bytes)
        self.assertEqual(len(list(self.root.iterdir())), 2)

    def test_failed_serialization_preserves_previous_checkpoint(self):
        save_checkpoint(self.old_state, self.destination)
        original_bytes = self.destination.read_bytes()

        def incomplete_save(state, stream):
            stream.write(b'incomplete checkpoint')
            raise RuntimeError('Serialization failed')

        with patch('src.checkpoint.torch.save', side_effect=incomplete_save):
            with self.assertRaisesRegex(RuntimeError, 'Serialization failed'):
                save_checkpoint(self.new_state, self.destination)
        self.assertEqual(self.destination.read_bytes(), original_bytes)
        self.assertEqual(list(self.root.iterdir()), [self.destination])

    def test_repeated_locks_keep_filenames_short(self):
        path = self.destination
        with patch('src.checkpoint.os.replace', side_effect=PermissionError('Locked')), \
                patch('src.checkpoint.time.sleep'):
            for _ in range(12):
                path = save_checkpoint(self.new_state, path)
                self.assertLess(len(path.name), 40)
                self.assert_saved_state(path, self.new_state)

    def test_unrelated_io_errors_are_reported(self):
        save_checkpoint(self.old_state, self.destination)
        with patch('src.checkpoint.os.replace', side_effect=OSError(errno.EIO, 'Disk error')):
            with self.assertRaises(OSError) as error:
                save_checkpoint(self.new_state, self.destination)
        self.assertEqual(error.exception.errno, errno.EIO)
        self.assert_saved_state(self.destination, self.old_state)
        self.assertEqual(list(self.root.iterdir()), [self.destination])

    @unittest.skipUnless(sys.platform == 'win32', 'Windows file mapping regression')
    def test_real_windows_memory_mapped_checkpoint(self):
        save_checkpoint(self.old_state, self.destination)
        with self.destination.open('rb') as handle:
            with mmap.mmap(handle.fileno(), 0, access=mmap.ACCESS_READ) as mapping:
                original_bytes = mapping[:]
                saved_path = save_checkpoint(self.new_state, self.destination)
                self.assertEqual(mapping[:], original_bytes)
                self.assert_saved_state(saved_path, self.new_state)
        self.assert_saved_state(saved_path, self.new_state)


if __name__ == '__main__':
    unittest.main()
