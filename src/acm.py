"""The labeled ACM paper/author/subject graph from DGL's HAN example.

Preprocessing reference:
https://github.com/dmlc/dgl/blob/master/examples/pytorch/han/utils.py
The five selected conferences define three paper classes. DGL is not required.
"""

import shutil
import tempfile
from pathlib import Path
from urllib.request import urlopen

import numpy as np
import scipy.sparse as sp
import torch
from scipy.io import loadmat
from sklearn.model_selection import train_test_split
from torch_geometric.data import HeteroData, InMemoryDataset


class ACM(InMemoryDataset):
    url = 'https://data.dgl.ai/dataset/ACM.mat'

    def __init__(self, root, transform=None, pre_transform=None, force_reload=False):
        super().__init__(root, transform, pre_transform, force_reload=force_reload)
        self.load(self.processed_paths[0], data_cls=HeteroData)

    @property
    def raw_file_names(self):
        return ['ACM.mat']

    @property
    def processed_file_names(self):
        return ['data.pt']

    def download(self):
        destination = Path(self.raw_paths[0])
        with tempfile.TemporaryDirectory(dir=self.raw_dir) as directory:
            temporary = Path(directory) / 'ACM.mat'
            try:
                with urlopen(self.url, timeout=30) as source, temporary.open('wb') as target:
                    shutil.copyfileobj(source, target)
                raw = loadmat(temporary)
                if not all(key in raw for key in ['PvsA', 'PvsL', 'PvsT', 'PvsC']):
                    raise ValueError('ACM.mat is missing required matrices')
            except (OSError, ValueError) as error:
                raise RuntimeError(
                    f'Could not download ACM. Place ACM.mat in {self.raw_dir} and retry.'
                ) from error
            temporary.replace(destination)

    def process(self):
        self.save([self.build_graph(loadmat(self.raw_paths[0]))], self.processed_paths[0])

    def build_graph(self, raw):
        conferences = [0, 1, 9, 10, 13]
        class_ids = [0, 1, 2, 2, 1]
        paper_conference = raw['PvsC'].tocsr()
        selected = np.flatnonzero(np.asarray(paper_conference[:, conferences].sum(axis=1)).ravel())
        paper_conference = paper_conference[selected]
        paper_features = raw['PvsT'].tocsr()[selected].astype(np.float32)
        labels = np.full(len(selected), -1, dtype=np.int64)
        for conference, class_id in zip(conferences, class_ids):
            rows = paper_conference[:, conference].nonzero()[0]
            if np.any((labels[rows] != -1) & (labels[rows] != class_id)):
                raise ValueError('ACM papers have conflicting class labels')
            labels[rows] = class_id

        data = HeteroData()
        data.feature_source = 'DGL HAN ACM raw: paper words; author/subject neighbor means'
        data.preprocessing = 'han-raw-v1'
        data['paper'].x = torch.from_numpy(paper_features.toarray())
        data['paper'].y = torch.from_numpy(labels)
        for node_type, matrix_name in [('author', 'PvsA'), ('subject', 'PvsL')]:
            incidence = raw[matrix_name].tocsr()[selected].astype(np.float32)
            incidence = incidence[:, np.asarray(incidence.getnnz(axis=0)).ravel() > 0]
            incidence.data[:] = 1
            rows, columns = incidence.nonzero()
            edges = torch.from_numpy(np.stack([rows, columns]).astype(np.int64))
            data['paper', 'to', node_type].edge_index = edges
            data[node_type, 'to', 'paper'].edge_index = edges.flip(0)
            # Auxiliary features use only graph structure and paper words.
            degree = np.asarray(incidence.sum(axis=0)).ravel()
            mean = sp.diags(1 / np.maximum(degree, 1)) @ incidence.T @ paper_features
            data[node_type].x = torch.from_numpy(mean.toarray().astype(np.float32))

        indices = np.arange(len(labels))
        train_indices, held_out = train_test_split(
            indices, train_size=round(len(labels) * 0.20), random_state=42, stratify=labels,
        )
        val_indices, test_indices = train_test_split(
            held_out, train_size=round(len(labels) * 0.10), random_state=42,
            stratify=labels[held_out],
        )
        for split, indices in [('train', train_indices), ('val', val_indices), ('test', test_indices)]:
            mask = torch.zeros(len(labels), dtype=torch.bool)
            mask[torch.from_numpy(indices)] = True
            data['paper'][f'{split}_mask'] = mask
        if self.pre_transform is not None:
            data = self.pre_transform(data)
        return data
