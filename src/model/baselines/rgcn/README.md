# RGCN

Paper: [Modeling Relational Data with Graph Convolutional Networks](https://arxiv.org/abs/1703.06103). Original source: [https://github.com/tkipf/relational-gcn](https://github.com/tkipf/relational-gcn).

`model.py` contains the runnable architecture, `layers.py` the reusable layers/kernel imports, and `preprocessing.py` where model-specific preparation is needed. Shared metapath connectivity is implemented in `src/graph.py` and input binding in `src/experiments/models.py`.

One relation ID per selected metapath; stacked PyG RGCNConv layers with optional basis decomposition. The common runner supplies an endpoint graph rather than the original heterogeneous adjacency.

```python
from src.model.baselines.rgcn import RGCN
# Construct with dimensions/options documented in model.py, then:
model(x, edge_index, edge_type)
```

The shared runner obtains a 128-dimensional embedding and applies a linear NC head or the paper's LP dot-product decoder. `python run_dblp.py --models rgcn --dry-run` inspects the configuration without loading data. Constructor settings, applicable grid axes, defaults and all common protocol adaptations are described in [paper_protocol.md](../../../../docs/paper_protocol.md). Synthetic forward/backward and task checks cover this model in `tests/test_paper_models.py`.

## Upstream snapshot

Pinned commit: `4bec1341dd46b72bf482f7ed26c2dca4533577f6`. [PROVENANCE.json](upstream/PROVENANCE.json) records original paths and SHA-256 hashes. The snapshot contains source, preprocessing scripts and available original documentation/notices; data, parameters, result files and notebook outputs are excluded. Notebook code cells are extracted into Python scripts where present. Upstream imports are not executed by ordinary package discovery.

The adapter is a documented translation/adaptation, not a claim of bit-for-bit reproduction or verified benchmark scores. Retain upstream attribution and license notices; the repository's MIT license does not relicense the snapshot. Where upstream supplies no license file, no substitute license is asserted here.
