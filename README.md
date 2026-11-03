# SPCA

Source code for **SPCA: SubPath Semantic Encoding for Context Aware Heterogeneous Graph Neural Networks**, by Amine Lakhder, Fahd Kalloubi, Lamiae Azizi, Imad Kissami, and Abdelhak Boulaalam.

SPCA combines attention over metapath endpoint neighborhoods with learned embeddings of the internal node-type sequence. The implementation follows Sections 4 and 7.2 of the manuscript: typed feature projection, dot-product neighbor attention, structural LayerNorm, shared type embeddings and a dedicated two-layer MLP per distinct subpath, additive context fusion followed by LayerNorm, and concatenation with a final linear projection.

This repository contains source, scripts, tests, configuration and documentation. Datasets, trained parameters, experiment outputs, CSVs and figures remain in ignored local directories.

## Setup

Use Python 3.11 or newer and a PyTorch installation appropriate for your CPU/CUDA environment.

```bash
python -m venv .venv
# Linux/macOS: source .venv/bin/activate
# PowerShell: .venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python -m pytest -q
```

The models use PyTorch and PyG. Tests use synthetic graphs and mocked downloads.

## Run commands

Inspect the complete configuration without loading a dataset, importing PyTorch, or training:

```bash
python run_dblp.py --dry-run
python run_imdb.py --dry-run
python run_acm.py --dry-run
python run_lastfm.py --dry-run
python run_movielens.py --dry-run
```

When you are ready to run experiments, remove `--dry-run`. Each script includes SPCA and all 12 baselines, performs the applicable Cartesian grid search over three tuning seeds, then evaluates five reporting seeds. Defaults are 500 epochs, patience 100, Adam, and 128-dimensional final embeddings. NC checkpoints maximize validation Macro-F1, with lower validation loss breaking ties. Configurations are ranked by mean validation Macro-F1 across tuning seeds. Test metrics are computed once per reporting run at its selected checkpoint. Reports include mean and **sample** standard deviation.

```bash
# Complete main-protocol comparison: 20%, 40%, 60% of each class's training pool
python run_dblp.py
python run_imdb.py

# Low-label protocol: fractions of all labeled nodes per class, outside holdouts
python run_dblp.py --protocol low-label --train-ratios 5% 10%

# Select models, seeds, data location and fixed hyperparameters
python run_imdb.py --models spca han --train-ratios 20% 60% --seeds 42 43 44 45 46 --tuning-seeds 0 1 2 --data-root data/imdb --set learning_rate=0.005 --set spca.aggregator=mean

# Tune first, then report using the saved winning configurations
python run_dblp.py --phase tune --output local/dblp_tuning
python run_dblp.py --phase report --selected-config local/dblp_tuning/selected.json

# Run a fixed configuration
python run_dblp.py --phase report --models spca --set hidden_channels=128 --set dropout=0.5

# K-means on the selected NC embeddings, reporting NMI and ARI
python run_dblp.py --task clustering

# LP: LastFM 70/10/20; MovieLens-1M 80/10/10; one negative per positive
python run_lastfm.py
python run_movielens.py
```

Use `--help` for all settings. `--set MODEL.KEY=VALUE` accepts model-specific options, including SPCA's context positions, normalization and encoder; MAGNN's instance encoder and sample cap; GTC's hop count and Transformer depth; HetTree's training-label propagation; and embedding/head widths. Values accept JSON (`true`, `false`, `null`, numbers) or plain strings. `--config configs/paper.json` supplies protocol defaults; CLI flags override them. `--grid configs/grid.json` supplies search arrays; fixed `--set` values remove their axes from the search. Per-head widths are searched only for architectures with a separate head-width parameter. See [experiment settings](docs/paper_protocol.md).

## Data and reproducibility

The exact Table 3 metapaths are defined in [config.py](src/experiments/config.py). All methods use these selected paths. Methods without native metapaths receive their induced endpoint graphs; dynamic path expansion is disabled. MAGNN also retains ordered path instances. LP removes validation/test interactions and their reverses **before** preparing paths, propagated features, positives or other encoder inputs. Negative pairs exclude every observed interaction.

NC retains the loader's fixed validation/test sets by default. Main ratios sample within its original training pool; low-label ratios count all labeled nodes per class. `--val-count`, `--test-count`/`--test-ratio`, and `--split-seed` explicitly replace those holdouts. `--graph path/to/graph.pt` accepts a trusted local `HeteroData` graph with typed features, edges, labels and split masks. `--metapaths path/to/paths.json` accepts a target-type-to-path-list mapping.

Use `--graph` to select a prepared local dataset:

```bash
python run_acm.py --graph local/acm3025_graph.pt
```

Run manifests record graph/feature/split fingerprints, seeds, effective settings and software versions locally. See [paper_protocol.md](docs/paper_protocol.md) for dataset preparation and experiment settings.

## Models and layout

```text
src/model/spca/             SPCA architecture, reusable PyG layers and encoder variants
src/model/baselines/<name>/ Model architecture, layers, preprocessing, README
src/experiments/            Configuration, input adapters, splits, training and CLI
src/graph.py               Exact binary metapath composition and instance sampling
run_<dataset>.py           Five dataset entrypoints
configs/                  Protocol defaults and the manuscript grid
tests/                    Offline unit/regression tests on synthetic data
local/, data/             Ignored runtime artifacts and dataset caches
```

| Model | Documentation and original source |
| --- | --- |
| SPCA | [Model and equation mapping](src/model/spca/README.md) |
| GCN | [Model](src/model/baselines/gcn/README.md) · [tkipf/gcn](https://github.com/tkipf/gcn) |
| GAT | [Model](src/model/baselines/gat/README.md) · [PetarV-/GAT](https://github.com/PetarV-/GAT) |
| RGCN | [Model](src/model/baselines/rgcn/README.md) · [tkipf/relational-gcn](https://github.com/tkipf/relational-gcn) |
| HAN | [Model](src/model/baselines/han/README.md) · [Jhy1993/HAN](https://github.com/Jhy1993/HAN) |
| MAGNN | [Model](src/model/baselines/magnn/README.md) · [cynricfu/MAGNN](https://github.com/cynricfu/MAGNN) |
| HGT | [Model](src/model/baselines/hgt/README.md) · [acbull/pyHGT](https://github.com/acbull/pyHGT) |
| HeCo | [Model](src/model/baselines/heco/README.md) · [liun-online/HeCo](https://github.com/liun-online/HeCo) |
| GTC | [Model](src/model/baselines/gtc/README.md) · [PHD-lanyu/GTC](https://github.com/PHD-lanyu/GTC) |
| Simple-HGN / HGB | [Model](src/model/baselines/simplehgn/README.md) · [THUDM/HGB](https://github.com/THUDM/HGB) |
| SeHGNN | [Model](src/model/baselines/sehgnn/README.md) · [ICT-GIMLab/SeHGNN](https://github.com/ICT-GIMLab/SeHGNN) |
| BG-HGNN | [Model](src/model/baselines/bghgnn/README.md) · [mao1207/BG-HGNN](https://github.com/mao1207/BG-HGNN) |
| HetTree | [Model and paper](src/model/baselines/hettree/README.md) |

Each model folder provides its runnable architecture, reusable layers and model-specific preprocessing. Use `run_<dataset>.py` for the manuscript protocol.

## Citation

```bibtex
@misc{lakhder2026spca,
  title  = {{SPCA}: SubPath Semantic Encoding for Context Aware Heterogeneous Graph Neural Networks},
  author = {Lakhder, Amine and Kalloubi, Fahd and Azizi, Lamiae and Kissami, Imad and Boulaalam, Abdelhak},
  year   = {2026},
  note   = {Preprint},
  url    = {https://github.com/LakhderAmine99/SPCA}
}
```

Machine-readable citation: [CITATION.cff](CITATION.cff). The project's own code uses the [MIT license](LICENSE). Baseline sources retain their authors' licenses and notices. Cite the relevant baseline papers when using their implementations.
