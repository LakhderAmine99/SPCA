"""Load MAGNN DBLP, with a labeled raw-data fallback for unavailable archives."""

import io
import json
import tempfile
from pathlib import Path
from urllib.request import urlopen
from zipfile import BadZipFile, ZipFile

import numpy as np
import scipy.sparse as sp
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from torch_geometric.data import HeteroData
from torch_geometric.datasets import DBLP as PyGDBLP


class DBLP(PyGDBLP):
    source_base = (
        'https://raw.githubusercontent.com/cynricfu/MAGNN/'
        'b8557f58ae04a7fe3e7a9fde64ea87c81b331efe/data/raw/DBLP/'
    )

    def download(self):
        cached = Path(self.raw_dir) / 'DBLP_processed.zip'
        try:
            if cached.exists():
                archive_bytes = cached.read_bytes()
            else:
                with urlopen(self.url, timeout=30) as response:
                    archive_bytes = response.read()
            with ZipFile(io.BytesIO(archive_bytes)) as archive:
                members = {Path(name).name: name for name in archive.namelist()}
                if not all(name in members for name in self.raw_file_names):
                    raise BadZipFile('DBLP archive is missing required raw files')
                with tempfile.TemporaryDirectory(dir=self.raw_dir) as directory:
                    for name in self.raw_file_names:
                        (Path(directory) / name).write_bytes(archive.read(members[name]))
                    for name in self.raw_file_names:
                        (Path(directory) / name).replace(Path(self.raw_dir) / name)
            return
        except (OSError, BadZipFile) as error:
            print(f'DBLP archive unavailable ({error}); using MAGNN raw graph and title features')
        self._build_from_source()

    def _build_from_source(self):
        names = ['author_label.txt', 'paper.txt', 'paper_author.txt', 'paper_term.txt', 'paper_conf.txt']
        text = {}
        try:
            for name in names:
                with urlopen(self.source_base + name, timeout=30) as response:
                    text[name] = response.read().decode('cp1252' if name == 'paper.txt' else 'utf-8')
        except OSError as error:
            raise RuntimeError(
                f'Both DBLP sources are unavailable. Place DBLP_processed.zip or '
                f'the seven PyG DBLP raw files in {self.raw_dir} and retry.'
            ) from error
        self._write_raw_graph(text)

    def _write_raw_graph(self, text):
        authors = sorted((int(row.split('\t')[0]), int(row.split('\t')[1]))
                         for row in text['author_label.txt'].splitlines() if row.strip())
        author_ids = {author_id: i for i, (author_id, _) in enumerate(authors)}
        paper_author = [tuple(map(int, row.split())) for row in text['paper_author.txt'].splitlines()
                        if row.strip()]
        paper_author = [(paper, author) for paper, author in paper_author if author in author_ids]
        valid_papers = {paper for paper, _ in paper_author}
        papers = sorted((int(row.split('\t', 1)[0]), row.split('\t', 1)[1])
                        for row in text['paper.txt'].splitlines()
                        if row.strip() and int(row.split('\t', 1)[0]) in valid_papers)
        paper_ids = {paper_id: i for i, (paper_id, _) in enumerate(papers)}
        relations = {'author': paper_author}
        for node_type, name in [('term', 'paper_term.txt'), ('conference', 'paper_conf.txt')]:
            relations[node_type] = [tuple(map(int, row.split())) for row in text[name].splitlines()
                                    if row.strip() and int(row.split()[0]) in paper_ids]
        mappings = {'author': author_ids}
        for node_type in ['term', 'conference']:
            mappings[node_type] = {key: i for i, key in enumerate(sorted({key for _, key in relations[node_type]}))}

        # A reproducible fallback with no external word embeddings or labels in features.
        paper_features = CountVectorizer(
            min_df=2, max_features=2048, stop_words='english', dtype=np.float32,
        ).fit_transform([title for _, title in papers])
        features = {'paper': paper_features}
        incidences = {}
        for node_type, pairs in relations.items():
            rows = [paper_ids[paper] for paper, _ in pairs]
            columns = [mappings[node_type][other] for _, other in pairs]
            incidence = sp.csr_matrix((np.ones(len(rows), dtype=np.float32), (rows, columns)),
                                      shape=(len(papers), len(mappings[node_type])))
            incidence.data[:] = 1
            incidences[node_type] = incidence
            degree = np.asarray(incidence.sum(axis=0)).ravel()
            features[node_type] = sp.diags(1 / np.maximum(degree, 1)) @ incidence.T @ paper_features

        node_types = ['author', 'paper', 'term', 'conference']
        sizes = [len(authors), len(papers), len(mappings['term']), len(mappings['conference'])]
        # Raw terms retain their original identities rather than applying the
        # archive's WordNet lemmatization, so their count differs from PyG's.
        if [sizes[0], sizes[1], sizes[3]] != [4057, 14328, 20] or sizes[2] < 1:
            raise RuntimeError(f'MAGNN DBLP source has unexpected node counts: {sizes}')
        offsets = dict(zip(node_types, np.cumsum([0] + sizes[:-1])))
        rows, columns = [], []
        for node_type, incidence in incidences.items():
            paper, other = incidence.nonzero()
            rows.extend((paper + offsets['paper']).tolist())
            columns.extend((other + offsets[node_type]).tolist())
        adjacency = sp.csr_matrix(
            (np.ones(2 * len(rows), dtype=np.float32), (rows + columns, columns + rows)),
            shape=(sum(sizes), sum(sizes)),
        )
        labels = np.array([label for _, label in authors], dtype=np.int64)
        remaining, val = train_test_split(np.arange(len(authors)), test_size=400, random_state=1566911444)
        train, test = train_test_split(remaining, test_size=3257, random_state=1566911444)
        with tempfile.TemporaryDirectory(dir=self.raw_dir) as directory:
            root = Path(directory)
            sp.save_npz(root / 'adjM.npz', adjacency)
            sp.save_npz(root / 'features_0.npz', features['author'].tocsr())
            sp.save_npz(root / 'features_1.npz', paper_features)
            np.save(root / 'features_2.npy', features['term'].toarray().astype(np.float32))
            np.save(root / 'labels.npy', labels)
            np.save(root / 'node_types.npy', np.repeat(np.arange(4), sizes))
            np.savez(root / 'train_val_test_idx.npz', train_idx=np.sort(train),
                     val_idx=np.sort(val), test_idx=np.sort(test))
            for name in self.raw_file_names:
                (root / name).replace(Path(self.raw_dir) / name)
        (Path(self.raw_dir) / 'feature_source.json').write_text(json.dumps({
            'feature_source': 'MAGNN raw graph (unlemmatized terms): title bag of words (max 2048) and neighbor means',
        }), encoding='utf-8')

    def process(self):
        super().process()
        self.load(self.processed_paths[0], data_cls=HeteroData)
        data = self.get(0)
        provenance = Path(self.raw_dir) / 'feature_source.json'
        data.feature_source = (json.loads(provenance.read_text(encoding='utf-8'))['feature_source']
                               if provenance.exists() else 'Original PyG/MAGNN features')
        data.preprocessing = 'raw-title-v1' if provenance.exists() else 'original'
        self.save([data], self.processed_paths[0])
