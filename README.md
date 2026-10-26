# SPCA: Subpath Semantic Encoding

Context-aware heterogeneous graph neural networks, controlled SPCA variants,
and a module for each baseline in the paper.

## Repository layout

```text
src/NC/                 Node-classification model adapters
src/LP/                 Link-prediction adapters and edge decoders
src/model/spca/          SPCA classifier, encoders, and ablations
src/model/baselines/     One module per baseline and lazy registry
src/model/model.py      Compatibility import for SPCA
src/{imdb,dblp,acm}.py   Dataset loaders
src/datasets.py         Shared dataset/metapath configuration
src/splits.py           Reproducible train/validation/test splits
src/checkpoint.py       Local checkpoint persistence
index.py                Existing experiment CLI
tests/                 Unit and regression tests
```

Notebooks, trained weights, datasets, generated reports, archives, editor files,
and historical exploratory scripts are **local-only**. Their working copies are
preserved, ignored, and removed from the Git index. Put new scratch work under
`local/` and external code under `third_party/`. Both directories are ignored.
Do not force-add ignored artifacts. Removing files from the index affects the
next commit; it does not erase old commits or change the remote until published.

## Task adapters: NC and LP

Both task packages reuse `src/model/`; architecture code is not duplicated.
Factories support SPCA (including its variants), GCN, GAT, HAN, RGCN, HGT,
Simple-HGN/HGB, SeHGNN, and HetTree. Model-specific constructor arguments and
forward inputs remain the same as in the shared modules.

```python
from src.NC import build_model as build_nc
from src.LP import build_model as build_lp

nc = build_nc("gcn", num_classes=3,
              in_channels=32, hidden_channels=64, num_layers=2)
node_logits = nc(x, train_edge_index)

lp = build_lp("gcn", embedding_channels=64, decoder="mlp",
              in_channels=32, hidden_channels=64, num_layers=2)
link_logits = lp(x, train_edge_index, edge_label_index=query_edges)
```

NC returns node logits for cross entropy on the training mask. `NodeClassifier`
can also add a classification head to a custom embedding encoder. The NC factory
uses each existing model's classification head directly.

LP sets the shared model's output width to `embedding_channels` and trains that
output projection as an embedding head. Existing model output transforms are
retained, including Simple-HGN normalization. It adds either an ordered-endpoint
MLP (`decoder="mlp"`, default) or symmetric dot product (`decoder="dot"`). Outputs
are raw link logits for `binary_cross_entropy_with_logits`; apply sigmoid only
when probabilities are needed. This is a task adaptation, not a claim to reproduce
the original paper's LP setup.

For cross-type links, use a separate encoder for each endpoint type:

```python
from src.LP import LinkPredictor

lp = LinkPredictor(source_encoder, embedding_channels=64,
                   destination_encoder=destination_encoder)
link_logits = lp(
    *source_args, edge_label_index=query_edges,
    destination_inputs=(destination_args, destination_kwargs),
    **source_kwargs,
)
```

Each encoder must return a matrix with 64 columns in this example. Query edges
have shape `[2, num_links]`, dtype `torch.long`, and index rows of the source and
destination output matrices respectively. For target-only architectures, configure
one model per endpoint target type, including that target's metapaths/features.

LP graph splits and negative sampling belong to the experiment: exclude held-out
positive edges and their reverse relations before message passing or cached
metapath/feature preparation. Rebuild SeHGNN, HetTree, and Simple-HGN caches from
that training graph. Query edges are passed only to the decoder; the adapter does
not remove them from supplied graph inputs. HetTree LP rejects NC label-propagation
inputs. Supply positive and sampled negative query edges with binary labels.

MAGNN, HeCo, GTC, and BG-HGNN remain external integrations. Both factories reject
their process launchers explicitly; they require upstream task-specific encoders
before they can use these adapters. The existing `index.py` NC CLI remains intact;
these packages add Python APIs, not an LP dataset/training CLI.

## SPCA and variants

```python
from src.model.spca import SPCA

model = SPCA(
    in_channels_dict, hidden_channels=128, num_classes=3,
    metadata=metadata, metapaths=metapaths, subpaths_dict=subpaths_dict,
    variant="base", context_mode="subpath", aggregator="concat-mlp",
)
logits = model(x_dict, edge_index_dict)
```

Variants: `base`, `regularized`, and `stacked`. Context ablations: `subpath`,
`no-context`, `metapath-bias`, and `metapath-scalar-bias`. Subpath aggregators:
`concat-mlp`, `mean`, `lstm`, and `transformer`. Historical encoders are exported
under their existing names; imports from `index` still work. State-dict names
inside the existing encoders have been preserved.

## Baselines

| Model | Module under `src.model.baselines` | Implementation / input | Pipeline |
| --- | --- | --- | --- |
| GCN | `gcn` | PyG GCN; homogeneous features and edges | E2E |
| GAT | `gat` | PyG GAT; homogeneous features and edges | E2E |
| HAN | `han` | PyG HANConv classifier; precomputed metapath graphs | Pre |
| MAGNN | `magnn` | Official-code process integration | Pre |
| RGCN | `rgcn` | PyG RGCNConv; packed features, edges, relation IDs | E2E |
| HGT | `hgt` | PyG HGTConv; original typed graph | E2E |
| GTC | `gtc` | Official-code process integration | Pre |
| HeCo | `heco` | Official-code process integration | Pre |
| HGB / Simple-HGN | `simplehgn` / `hgb` | Existing local adaptation | Input-dependent |
| SeHGNN | `sehgnn` | Existing feature-only adaptation | Pre |
| BG-HGNN | `bghgnn` | External contract; verified code still required | Pre |
| HetTree | `hettree` | Existing semantic-tree implementation | Pre |

E2E means no separate algorithmic preprocessing/postprocessing stage; normal
input packing is excluded. Pre identifies a separate feature/path preparation
stage. These labels describe the documented input pipeline, not every possible
implementation. Simple-HGN on the original graph is E2E; the current experiment
runner uses metapath projection and therefore requires Pre. Contrastive methods
also require their own representation-learning and downstream evaluation stages.

```python
from src.model.baselines.registry import BASELINES, get_baseline
GCN = get_baseline("gcn")
model = GCN(in_channels=32, hidden_channels=64, num_layers=2, out_channels=3)
logits = model(x, edge_index)
```

Model classes follow the input contracts in their modules. Homogeneous inputs
must use one common feature width and globally offset node IDs; discard relation
labels for GCN/GAT. RGCN uses fixed relation IDs. HGT receives `x_dict` and
`edge_index_dict`. HAN receives target features and a dictionary of already
composed target-to-target metapath graphs. No loader or download runs on import.

MAGNN, HeCo, and GTC retain the authors' preprocessing, objectives, and evaluation
through process integrations. Obtain the official code in an ignored local
checkout and install its dependencies in a separate environment:

- [MAGNN](https://github.com/cynricfu/MAGNN)
- [HeCo](https://github.com/liun-online/HeCo)
- [GTC](https://github.com/PHD-lanyu/GTC)
- [BG-HGNN paper](https://arxiv.org/abs/2403.08207): no verified upstream checkout
  is bundled; specify your verified implementation and its entrypoint explicitly.

```powershell
python -m src.model.baselines.magnn --repo third_party/MAGNN --entrypoint run_DBLP.py -- --help
python -m src.model.baselines.heco --repo third_party/HeCo -- acm --gpu=0
python -m src.model.baselines.gtc --repo third_party/GTC -- dblp
python -m src.model.baselines.bghgnn --repo third_party/BG-HGNN --entrypoint path/to/verified_script.py -- --help
```

Use `--python path/to/upstream/python` to select the upstream environment.
Launchers fail explicitly when code is absent; they never substitute another
architecture. Record the upstream commit and prepare identical data/splits before
comparing results. The launchers do **not** translate this project's shared masks
into upstream formats, and are not `nn.Module` classes. No new baseline scores or
paper-parity claims are implied by this organization.

## Existing experiments and validation

The existing CLI still supports its SPCA variants, SeHGNN, Simple-HGN, and HetTree;
new baseline modules are separate Python APIs or upstream launchers, not additional
`index.py --model` choices. Full prior experiment instructions and adaptation
limitations are preserved in [docs/experiments.md](docs/experiments.md).

```powershell
python index.py --dataset imdb --model subpath --train-ratio "10%"
python -m pytest -q
python scripts/check_public_files.py
```

The checked local environment is Python with PyTorch 2.12.1 and PyG 2.8.0.
Install `requirements.txt` in an environment appropriate for your CPU/CUDA setup.
External integrations use their upstream environments instead.
