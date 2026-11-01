"""Vendor source-only snapshots from the manuscript's Table 12 (never datasets).

Run explicitly to refresh sources. Existing provenance pins are reused unless
--refresh is supplied. Notebook code cells become .py; outputs are never saved.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path, PurePosixPath
from urllib.parse import quote
from urllib.request import Request, urlopen

ROOT = Path(__file__).resolve().parents[1] / 'src/model/baselines'
REPOS = {
    'gcn': 'tkipf/gcn', 'gat': 'PetarV-/GAT', 'rgcn': 'tkipf/relational-gcn',
    'han': 'Jhy1993/HAN', 'magnn': 'cynricfu/MAGNN', 'hgt': 'acbull/pyHGT',
    'heco': 'liun-online/HeCo', 'gtc': 'PHD-lanyu/GTC',
    'simplehgn': 'THUDM/HGB', 'sehgnn': 'ICT-GIMLab/SeHGNN',
    'bghgnn': 'mao1207/BG-HGNN',
}


def fetch(url):
    with urlopen(Request(url, headers={'User-Agent': 'SPCA-source-vendor'}), timeout=60) as response:
        return response.read()


def selected(path, model):
    p = PurePosixPath(path)
    if model == 'simplehgn' and len(p.parts) > 1:
        if not path.startswith(('NC/benchmark/methods/baseline/', 'NC/benchmark/scripts/')):
            return False
    if any(part.lower() in {'data', 'dataset', 'datasets', 'results', 'checkpoint',
                            '__pycache__', '.git', '.ipynb_checkpoints'} for part in p.parts[:-1]):
        return False
    return (p.suffix in {'.py', '.sh', '.ipynb'} or
            p.name.lower().startswith(('license', 'copying', 'notice', 'readme')) or
            p.name in {'requirements.txt', 'environment.yml', 'setup.cfg'})


def vendor(model, refresh=False):
    repo = REPOS[model]
    folder = ROOT / model / 'upstream'
    folder.mkdir(parents=True, exist_ok=True)
    folder = folder.resolve()
    manifest = folder / 'PROVENANCE.json'
    ref = 'HEAD'
    if manifest.exists() and not refresh:
        ref = json.loads(manifest.read_text())['commit']
    revision = json.loads(fetch(f'https://api.github.com/repos/{repo}/commits/{ref}'))
    commit = revision['sha']
    tree = json.loads(fetch(f'https://api.github.com/repos/{repo}/git/trees/{commit}?recursive=1'))
    if tree.get('truncated'):
        raise RuntimeError(f'Incomplete upstream tree: {repo}')
    entries = [x['path'] for x in tree['tree'] if x['type'] == 'blob' and selected(x['path'], model)]
    def download(path):
        content = fetch(f'https://raw.githubusercontent.com/{repo}/{commit}/{quote(path)}')
        target = PurePosixPath(path)
        converted = target.suffix == '.ipynb'
        if converted:
            notebook = json.loads(content)
            cells = [''.join(cell.get('source', [])) for cell in notebook['cells'] if cell['cell_type'] == 'code']
            # Preserve source as a script, commenting notebook-only magics.
            lines = ('\n\n# %%\n\n'.join(cells)).splitlines()
            content = ('# Source cells extracted from ' + path + '\n' + '\n'.join(
                '# ' + line if line.lstrip().startswith(('%', '!')) else line for line in lines
            ) + '\n').encode()
            target = target.with_suffix('.py')
        destination = folder.joinpath(*target.parts)
        if target.is_absolute() or '..' in target.parts:
            raise ValueError(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(content)
        return {'path': str(target), 'original_path': path,
                'sha256': hashlib.sha256(content).hexdigest(), 'notebook_code_only': converted}
    with ThreadPoolExecutor(max_workers=6) as pool:
        files = list(pool.map(download, entries))
    folder.mkdir(parents=True, exist_ok=True)
    manifest.write_text(json.dumps({'repository': f'https://github.com/{repo}', 'commit': commit, 'tree': tree['sha'],
        'scope': 'Source, preprocessing scripts, notices and upstream documentation only.',
        'files': files}, indent=2) + '\n', encoding='utf-8')
    print(f'{model}: {len(files)} source files at {commit}', flush=True)


def verify_revision(model):
    """Resolve a snapshot revision to a commit and verify its tree identity."""
    manifest = ROOT / model / 'upstream/PROVENANCE.json'
    metadata = json.loads(manifest.read_text())
    revision = json.loads(fetch(f'https://api.github.com/repos/{REPOS[model]}/commits/HEAD'))
    current_tree = revision['commit']['tree']['sha']
    if metadata['commit'] not in {revision['sha'], current_tree}:
        raise ValueError(f'{model}: upstream changed; do not relabel this snapshot')
    metadata.update(commit=revision['sha'], tree=current_tree)
    manifest.write_text(json.dumps(metadata, indent=2) + '\n', encoding='utf-8')
    print(f'{model}: commit {revision["sha"]}, tree {current_tree}', flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--models', nargs='+', choices=REPOS, default=list(REPOS))
    parser.add_argument('--refresh', action='store_true')
    parser.add_argument('--verify-revisions', action='store_true')
    args = parser.parse_args()
    for name in args.models:
        if args.verify_revisions:
            verify_revision(name)
        else:
            vendor(name, args.refresh)
