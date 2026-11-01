# GAT

Paper: [Graph Attention Networks](https://arxiv.org/abs/1710.10903). Original source: [https://github.com/PetarV-/GAT](https://github.com/PetarV-/GAT).

`model.py` contains the runnable architecture, `layers.py` the reusable layers/kernel imports, and `preprocessing.py` where model-specific preparation is needed. Shared metapath connectivity is implemented in `src/graph.py` and input binding in `src/experiments/models.py`.

The metapath union feeds local GATConv layers. Input projection width and channels per attention head are independently configurable; the final output has the common embedding width.

```python
from src.model.baselines.gat import GAT
# Construct with dimensions/options documented in model.py, then:
model(x, edge_index)
```

The shared runner obtains a 128-dimensional embedding and applies a linear NC head or the paper's LP dot-product decoder. `python run_dblp.py --models gat --dry-run` inspects the configuration without loading data. Constructor settings, applicable grid axes, defaults and all common protocol adaptations are described in [paper_protocol.md](../../../../docs/paper_protocol.md). Synthetic forward/backward and task checks cover this model in `tests/test_paper_models.py`.

## Upstream snapshot

Pinned commit: `5af87e7fce2b90ae1cbd621cd58059036a3c7436`. [PROVENANCE.json](upstream/PROVENANCE.json) records original paths and SHA-256 hashes. The snapshot contains source, preprocessing scripts and available original documentation/notices; data, parameters, result files and notebook outputs are excluded. Notebook code cells are extracted into Python scripts where present. Upstream imports are not executed by ordinary package discovery.

The adapter is a documented translation/adaptation, not a claim of bit-for-bit reproduction or verified benchmark scores. Retain upstream attribution and license notices; the repository's MIT license does not relicense the snapshot. Where upstream supplies no license file, no substitute license is asserted here.
