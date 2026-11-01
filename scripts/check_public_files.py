"""Validate the prospective published tree without removing local artifacts."""
import argparse
from pathlib import Path, PurePosixPath
import subprocess

FORBIDDEN = {'.pt', '.pth', '.ckpt', '.safetensors', '.onnx', '.pkl', '.pickle',
             '.npy', '.npz', '.mat', '.h5', '.hdf5', '.csv', '.tsv', '.dat',
             '.parquet', '.arrow', '.feather', '.bin', '.ipynb', '.pdf', '.png',
             '.jpg', '.jpeg', '.gif', '.svg', '.zip', '.tar', '.gz', '.7z',
             '.mp4', '.xlsx', '.xls', '.jsonl', '.log', '.pyc', '.pyo'}
LOCAL_DIRS = {'data', 'dataset_statistics', 'checkpoint', 'checkpoints', 'results',
              'outputs', 'runs', 'wandb', 'tensorboard', 'local', 'third_party',
              '__pycache__', '.venv', '.pytest_cache', 'notebooks'}


def violation(path, size=0):
    p = PurePosixPath(path)
    if p.suffix.lower() in FORBIDDEN:
        return 'generated artifact or dataset extension'
    if any(part in LOCAL_DIRS for part in p.parts[:-1]):
        return 'local-only directory'
    if p.suffix == '.json' and not (p.parts[0] == 'configs' or p.name == 'PROVENANCE.json'):
        return 'runtime JSON outside source configuration/provenance'
    if p.suffix == '.txt' and not (p.name.startswith('requirements') or p.name in {'LICENSE.txt', 'NOTICE.txt', 'README.txt'}):
        return 'non-source text artifact'
    if size > 2 * 1024 * 1024:
        return 'unexpectedly large source file (>2 MiB)'
    return None


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--tracked-only', action='store_true')
    args = parser.parse_args(argv)
    root = Path(__file__).resolve().parents[1]
    command = ['git', 'ls-files', '-z', '--cached']
    if not args.tracked_only:
        command += ['--others', '--exclude-standard']
    result = subprocess.run(command, cwd=root, capture_output=True, check=True)
    candidates = set(result.stdout.decode('utf-8').split('\0')) - {''}
    errors = []
    for path in sorted(candidates):
        file = root / path
        if not file.is_file():
            continue  # A source deletion in the working tree.
        reason = violation(path, file.stat().st_size)
        if reason:
            errors.append(f'{path}: {reason}')
    ignored = subprocess.run(['git', 'ls-files', '-ci', '--exclude-standard'], cwd=root,
                             text=True, capture_output=True, check=True).stdout.strip()
    if ignored:
        errors.append('Ignored files are still tracked:\n' + ignored)
    if errors:
        print('\n'.join(errors))
        return 1
    print('Publication check passed: source, scripts, configuration, tests and documentation only.')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
