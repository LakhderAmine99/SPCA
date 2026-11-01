import hashlib
import json
from pathlib import Path
import pytest
from scripts.check_public_files import violation


@pytest.mark.parametrize('path', ['model.pt', 'best.pth', 'anything.csv', 'results/report.json',
    'local/source.py', 'data/raw.dat', 'notebook.ipynb', 'figure.pdf', 'summary.json', 'arbitrary.txt'])
def test_generated_files_cannot_be_published(path):
    assert violation(path)


@pytest.mark.parametrize('path', ['README.md', 'CITATION.cff', 'src/model/spca/layers.py',
    'configs/paper.json', 'src/model/baselines/gcn/upstream/PROVENANCE.json',
    'src/model/baselines/gcn/upstream/LICENSE', 'requirements.txt'])
def test_source_and_provenance_are_publishable(path):
    assert violation(path) is None


def test_all_vendored_files_match_pinned_hashes_and_are_source_only():
    root = Path(__file__).resolve().parents[1]
    manifests = sorted((root / 'src/model/baselines').glob('*/upstream/PROVENANCE.json'))
    assert len(manifests) == 11
    for path in manifests:
        metadata = json.loads(path.read_text())
        assert len(metadata['commit']) == 40
        for file in metadata['files']:
            source = path.parent / file['path']
            assert source.is_file(), source
            assert hashlib.sha256(source.read_bytes()).hexdigest() == file['sha256'], source
            assert violation(source.relative_to(root).as_posix(), source.stat().st_size) is None


def test_every_model_has_documented_architecture_and_layers():
    from src.experiments.config import MODELS
    root = Path(__file__).resolve().parents[1] / 'src/model'
    for name in MODELS:
        folder = root / 'spca' if name == 'spca' else root / 'baselines' / name
        for file in ('__init__.py', 'model.py', 'layers.py', 'README.md'):
            assert (folder / file).is_file(), (name, file)


def test_source_export_omits_history_and_ignored_artifacts(tmp_path, monkeypatch):
    from scripts.export_source import export
    import subprocess
    root = tmp_path / 'repo'
    root.mkdir()
    (root / 'main.py').write_text('print("source")\n')
    (root / '.git').mkdir()
    (root / 'local').mkdir()
    (root / 'local/model.pt').write_bytes(b'local only')
    monkeypatch.setattr(subprocess, 'run', lambda *a, **k: subprocess.CompletedProcess(a, 0, b'main.py\0', b''))
    destination = tmp_path / 'export'
    assert export(root, destination) == 1
    assert (destination / 'main.py').is_file()
    assert not (destination / '.git').exists()
    assert not (destination / 'local').exists()
    assert (root / 'local/model.pt').exists()
    with pytest.raises(FileExistsError):
        export(root, destination)
