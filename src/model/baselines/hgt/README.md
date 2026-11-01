# HGT

Paper: [Heterogeneous Graph Transformer](https://arxiv.org/abs/2003.01332). Original source: [https://github.com/acbull/pyHGT](https://github.com/acbull/pyHGT).

`model.py` contains the runnable architecture, `layers.py` the reusable layers/kernel imports, and `preprocessing.py` where model-specific preparation is needed. Shared metapath connectivity is implemented in `src/graph.py` and input binding in `src/experiments/models.py`.

Typed projections followed by PyG HGTConv and a task projection. In the paper runner each selected metapath becomes a relation over the receiving endpoint type; full typed graphs remain supported by the Python model.

```python
from src.model.baselines.hgt import HGT
# Construct with dimensions/options documented in model.py, then:
model(x_dict, edge_index_dict)
```

The shared runner obtains a 128-dimensional embedding and applies a linear NC head or the paper's LP dot-product decoder. `python run_dblp.py --models hgt --dry-run` inspects the configuration without loading data. Constructor settings, applicable grid axes, defaults and all common protocol adaptations are described in [paper_protocol.md](../../../../docs/paper_protocol.md). Synthetic forward/backward and task checks cover this model in `tests/test_paper_models.py`.

## Upstream snapshot

Pinned commit: `85eaccd482bc1d1af56c2de297b6e3a88b96d5cd`. [PROVENANCE.json](upstream/PROVENANCE.json) records original paths and SHA-256 hashes. The snapshot contains source, preprocessing scripts and available original documentation/notices; data, parameters, result files and notebook outputs are excluded. Notebook code cells are extracted into Python scripts where present. Upstream imports are not executed by ordinary package discovery.

The adapter is a documented translation/adaptation, not a claim of bit-for-bit reproduction or verified benchmark scores. Retain upstream attribution and license notices; the repository's MIT license does not relicense the snapshot. Where upstream supplies no license file, no substitute license is asserted here.
