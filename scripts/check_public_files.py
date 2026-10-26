"""Reject ignored research artifacts accidentally forced into the Git index."""
from pathlib import Path
import subprocess
import sys


def main():
    root = Path(__file__).resolve().parents[1]
    result = subprocess.run(
        ['git', 'ls-files', '-ci', '--exclude-standard'], cwd=root,
        text=True, capture_output=True, check=True,
    )
    if result.stdout.strip():
        print('Local-only files are tracked; remove them with git rm --cached:')
        print(result.stdout)
        return 1
    print('No ignored artifacts are tracked.')
    return 0


if __name__ == '__main__':
    sys.exit(main())
