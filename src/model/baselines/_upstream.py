"""Run research implementations in their own environment and working directory.

This is a process integration, not a replacement architecture or a shared-split
adapter. Data preparation and evaluation follow the selected upstream checkout.
"""
from dataclasses import dataclass
from pathlib import Path
import argparse
import subprocess
import sys


@dataclass(frozen=True)
class UpstreamBaseline:
    name: str
    source: str
    workdir: str = '.'
    entrypoint: str | None = None

    def command(self, repo, args=(), *, python=sys.executable, entrypoint=None):
        root = Path(repo).expanduser().resolve()
        cwd = (root / self.workdir).resolve()
        entry = entrypoint or self.entrypoint
        if entry is None:
            raise ValueError(f'{self.name}: specify the entrypoint from your verified checkout; {self.source}')
        script = (cwd / entry).resolve()
        if not script.is_relative_to(root):
            raise ValueError('Entrypoint must be inside the upstream checkout')
        if not script.is_file():
            raise FileNotFoundError(f'{self.name}: missing {script}. Obtain the implementation from {self.source}')
        return [str(python), str(script), *map(str, args)], cwd

    def run(self, repo, args=(), *, python=sys.executable, entrypoint=None):
        command, cwd = self.command(repo, args, python=python, entrypoint=entrypoint)
        return subprocess.run(command, cwd=cwd, check=True)


def main(baseline):
    parser = argparse.ArgumentParser(description=f'{baseline.name} upstream integration: {baseline.source}')
    parser.add_argument('--repo', required=True, help='Local upstream checkout (prefer ignored third_party/)')
    parser.add_argument('--python', default=sys.executable, help='Python from the upstream environment')
    parser.add_argument('--entrypoint', help='Script relative to the upstream working directory')
    parser.add_argument('args', nargs=argparse.REMAINDER, help='Arguments after -- passed unchanged')
    options = parser.parse_args()
    args = options.args[1:] if options.args[:1] == ['--'] else options.args
    baseline.run(options.repo, args, python=options.python, entrypoint=options.entrypoint)
