"""Load PyG's IMDB data even when the upstream Dropbox archive is unavailable."""

import csv
import io
import shutil
import tempfile
from pathlib import Path
from urllib.request import urlopen
from zipfile import BadZipFile, ZipFile

import numpy as np
import scipy.sparse as sp
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from torch_geometric.datasets import IMDB as PyGIMDB


class IMDB(PyGIMDB):
    # Follow the original authors' preprocessing, with a pinned source revision:
    # https://github.com/cynricfu/MAGNN/blob/b8557f58ae04a7fe3e7a9fde64ea87c81b331efe/preprocess_IMDB.ipynb
    source_url = (
        'https://raw.githubusercontent.com/cynricfu/MAGNN/'
        'b8557f58ae04a7fe3e7a9fde64ea87c81b331efe/'
        'data/raw/IMDB/movie_metadata.csv'
    )

    def _extract_raw_files(self, path):
        # Archives may contain an enclosing IMDB_processed/ directory. Only
        # extract the six files PyG needs, directly into raw_dir.
        with ZipFile(path) as archive:
            members = {
                Path(member.filename).name: member
                for member in archive.infolist() if not member.is_dir()
            }
            if not all(name in members for name in self.raw_file_names):
                raise BadZipFile('The archive is missing required IMDB files')
            # Finish reading every member (including its CRC check) before
            # publishing files that PyG could mistake for a complete cache.
            with tempfile.TemporaryDirectory(dir=self.raw_dir) as tmp_dir:
                for name in self.raw_file_names:
                    with archive.open(members[name]) as source:
                        with (Path(tmp_dir) / name).open('wb') as target:
                            shutil.copyfileobj(source, target)
                for name in self.raw_file_names:
                    (Path(tmp_dir) / name).replace(Path(self.raw_dir) / name)

    def download(self):
        raw_dir = Path(self.raw_dir)
        cached_zip = raw_dir / 'IMDB_processed.zip'
        if cached_zip.exists():
            try:
                self._extract_raw_files(cached_zip)
                return
            except BadZipFile:
                print(f'Ignoring invalid IMDB archive: {cached_zip}')
                cached_zip.unlink()

        # Download into a temporary location so an HTML error response or an
        # interrupted transfer cannot become a cached dataset archive.
        try:
            with tempfile.TemporaryDirectory(dir=raw_dir) as tmp_dir:
                archive_path = Path(tmp_dir) / cached_zip.name
                print('Downloading IMDB archive')
                with urlopen(self.url, timeout=30) as response:
                    with archive_path.open('wb') as target:
                        shutil.copyfileobj(response, target)
                self._extract_raw_files(archive_path)
                return
        except (BadZipFile, OSError) as exc:
            print(f'IMDB archive unavailable ({exc}); rebuilding from MAGNN source')

        self._build_from_source()

    def _build_from_source(self):
        raw_dir = Path(self.raw_dir)
        source_path = raw_dir / 'movie_metadata.csv'
        if source_path.exists():
            source_text = source_path.read_text(encoding='utf-8')
        else:
            try:
                with urlopen(self.source_url, timeout=30) as response:
                    source_text = response.read().decode('utf-8')
            except OSError as exc:
                raise RuntimeError(
                    'Both IMDB download sources are unavailable. Place the six '
                    f'IMDB raw files or a valid IMDB_processed.zip in {raw_dir} '
                    'and rerun.'
                ) from exc

        reader = csv.DictReader(io.StringIO(source_text))
        required_columns = {
            'director_name', 'actor_1_name', 'actor_2_name', 'actor_3_name',
            'genres', 'plot_keywords',
        }
        if not required_columns.issubset(reader.fieldnames or []):
            raise RuntimeError(f'Invalid MAGNN source CSV: {source_path}')

        movies, labels = [], []
        genre_labels = {'Action': 0, 'Comedy': 1, 'Drama': 2}
        for movie in reader:
            if not movie['actor_1_name'] or not movie['director_name']:
                continue
            for genre in movie['genres'].split('|'):
                if genre in genre_labels:
                    movies.append(movie)
                    labels.append(genre_labels[genre])
                    break

        directors = sorted({movie['director_name'] for movie in movies})
        actors = sorted({
            movie[f'actor_{i}_name'] for movie in movies for i in range(1, 4)
            if movie[f'actor_{i}_name']
        })
        director_ids = {name: i for i, name in enumerate(directors)}
        actor_ids = {name: i for i, name in enumerate(actors)}
        num_movies, num_directors = len(movies), len(directors)

        rows, cols = [], []
        for i, movie in enumerate(movies):
            rows.append(i)
            cols.append(num_movies + director_ids[movie['director_name']])
            for j in range(1, 4):
                actor = movie[f'actor_{j}_name']
                if actor:
                    rows.append(i)
                    cols.append(num_movies + num_directors + actor_ids[actor])

        num_nodes = num_movies + num_directors + len(actors)
        adjacency = sp.csr_matrix(
            (np.ones(2 * len(rows), dtype=np.int64),
             (rows + cols, cols + rows)), shape=(num_nodes, num_nodes),
        )
        adjacency.data[:] = 1  # Match the original binary adjacency matrix.
        movie_features = CountVectorizer(min_df=2).fit_transform(
            [movie['plot_keywords'] for movie in movies]
        )
        if (num_movies, num_directors, len(actors), movie_features.shape[1]) != (
            4278, 2081, 5257, 3066
        ):
            raise RuntimeError('MAGNN source does not match the expected IMDB dataset')

        incidence = adjacency[num_movies:, :num_movies]
        normalized = sp.diags(1.0 / np.asarray(incidence.sum(axis=1)).ravel()) @ incidence
        person_features = normalized @ movie_features
        train_idx, val_idx = train_test_split(
            np.arange(num_movies), test_size=400, random_state=1566911444,
        )
        train_idx, test_idx = train_test_split(
            train_idx, test_size=3478, random_state=1566911444,
        )

        source_path.write_text(source_text, encoding='utf-8')
        sp.save_npz(raw_dir / 'adjM.npz', adjacency)
        sp.save_npz(raw_dir / 'features_0.npz', movie_features)
        sp.save_npz(raw_dir / 'features_1.npz', person_features[:num_directors])
        sp.save_npz(raw_dir / 'features_2.npz', person_features[num_directors:])
        np.save(raw_dir / 'labels.npy', np.asarray(labels, dtype=np.int64))
        np.savez(
            raw_dir / 'train_val_test_idx.npz',
            train_idx=np.sort(train_idx), val_idx=np.sort(val_idx),
            test_idx=np.sort(test_idx),
        )
