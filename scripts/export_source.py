"""Export the checked source tree without Git history or local artifacts."""
import argparse
from pathlib import Path
import shutil
import subprocess
try:
    from .check_public_files import violation
except ImportError:
    from check_public_files import violation


def export(root, destination):
    root, destination = Path(root).resolve(), Path(destination).resolve()
    if destination == root or root.is_relative_to(destination):
        raise ValueError('The export must not replace the workspace or an ancestor')
    if destination.exists():
        raise FileExistsError(f'Export destination already exists: {destination}')
    result = subprocess.run(['git', 'ls-files', '-z', '--cached', '--others', '--exclude-standard'],
                            cwd=root, capture_output=True, check=True)
    files = []
    for name in sorted(set(result.stdout.decode().split('\0')) - {''}):
        source = root / name
        if not source.is_file():
            continue
        if not source.resolve().is_relative_to(root):
            raise ValueError(f'Source symlink escapes the workspace: {name}')
        reason = violation(name, source.stat().st_size)
        if reason:
            raise ValueError(f'Cannot export {name}: {reason}')
        files.append((name, source))
    # All checks complete before writing the first file. No .git is copied.
    destination.mkdir(parents=True)
    for name, source in files:
        target = destination / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, target)
    return len(files)


def main(argv=None):
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=root / 'local/public-source')
    args = parser.parse_args(argv)
    count = export(root, args.output)
    print(f'Exported {count} source/documentation files to {args.output}; no Git history or runtime artifacts.')


if __name__ == '__main__':
    main()
