# BG-HGNN

Paper: [BG-HGNN: Toward Efficient Learning for Complex Heterogeneous Graphs](https://arxiv.org/abs/2403.08207). Original source: [https://github.com/mao1207/BG-HGNN](https://github.com/mao1207/BG-HGNN).

`model.py` contains the runnable architecture, `layers.py` the reusable layers/kernel imports, and `preprocessing.py` where model-specific preparation is needed. Shared metapath connectivity is implemented in `src/graph.py` and input binding in `src/experiments/models.py`.

Random node-type/relation encodings and low-rank multiplicative fusion follow `NC/model.py`, then a shared homogeneous GAT processes the metapath union. Feature construction uses the configured random seed and device.

```python
from src.model.baselines.bghgnn import BGHGNN
# Construct with dimensions/options documented in model.py, then:
model(x, edge_index, edge_type)
```

The shared runner obtains a 128-dimensional embedding and applies a linear NC head or the paper's LP dot-product decoder. `python run_dblp.py --models bghgnn --dry-run` inspects the configuration without loading data. Constructor settings, applicable grid axes, defaults and shared experiment settings are described in [paper_protocol.md](../../../../docs/paper_protocol.md). Synthetic forward/backward and task checks cover this model in `tests/test_paper_models.py`.

## Upstream snapshot

Pinned commit: `87d1eb796563f46355eca1d721df32c4c53a45d3`. [PROVENANCE.json](upstream/PROVENANCE.json) records original paths and SHA-256 hashes. The snapshot contains source, preprocessing scripts and available original documentation/notices; data, parameters, result files and notebook outputs are excluded. Notebook code cells are extracted into Python scripts where present. Upstream imports are not executed by ordinary package discovery.

Upstream sources retain their authors' attribution and license notices.
