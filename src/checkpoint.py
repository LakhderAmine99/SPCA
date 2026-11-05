"""Save complete PyTorch checkpoints without overwriting files held open on Windows."""

import os
import tempfile
import time
from contextlib import suppress
from pathlib import Path

import torch

def save_checkpoint(state_dict, path):
    """Return the saved path, choosing a new filename if replacement stays locked."""
    destination = Path(path).resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = None
    keep_temporary = False
    try:
        # Passing an open file also keeps PyTorch from opening/truncating the
        # existing checkpoint through its native ZIP writer.
        with tempfile.NamedTemporaryFile(
            mode='wb', dir=destination.parent, prefix='checkpoint.',
            suffix=destination.suffix, delete=False,
        ) as stream:
            temporary_path = Path(stream.name)
            torch.save(state_dict, stream)
            stream.flush()
            os.fsync(stream.fileno())

        for delay in (0, 0.1, 0.2):
            if delay:
                time.sleep(delay)
            try:
                os.replace(temporary_path, destination)
                return destination
            except OSError as exc:
                # Access denied, sharing violation, lock violation, or an open
                # file mapping can prevent replacing a Windows checkpoint.
                if not (
                    isinstance(exc, PermissionError)
                    or getattr(exc, 'winerror', None) in {5, 32, 33, 1224}
                ):
                    raise

        # This file is already fully serialized and closed. Keep it as the
        # checkpoint, so a persistent lock does not discard the best weights.
        keep_temporary = True
        return temporary_path
    finally:
        if temporary_path is not None and not keep_temporary:
            with suppress(OSError):
                temporary_path.unlink(missing_ok=True)
