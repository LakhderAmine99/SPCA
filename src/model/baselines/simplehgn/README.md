# Simple-HGN / HGB

Paper: [Are We Really Making Much Progress? Revisiting, Benchmarking and Refining Heterogeneous Graph Neural Networks](https://arxiv.org/abs/2112.14936). Original source: [https://github.com/THUDM/HGB](https://github.com/THUDM/HGB).

`model.py` contains the runnable architecture, `layers.py` the reusable layers/kernel imports, and `preprocessing.py` where model-specific preparation is needed. Shared metapath connectivity is implemented in `src/graph.py` and input binding in `src/experiments/models.py`.

Edge-type embeddings, incoming-edge attention, typed self-loops, node/attention residuals and final L2 normalization. The shared runner supplies one endpoint relation per selected metapath. Hidden input width and per-head width can be varied independently.

```python
from src.model.baselines.simplehgn import SimpleHGN
# Construct with dimensions/options documented in model.py, then:
model(x_dict)
```

The shared runner obtains a 128-dimensional embedding and applies a linear NC head or the paper's LP dot-product decoder. `python run_dblp.py --models simplehgn --dry-run` inspects the configuration without loading data. Constructor settings, applicable grid axes, defaults and shared experiment settings are described in [paper_protocol.md](../../../../docs/paper_protocol.md). Synthetic forward/backward and task checks cover this model in `tests/test_paper_models.py`.

## Upstream snapshot

Pinned commit: `ca6fd5bb0c1ca32e63b132c8bfe8f11a4a6629fe`. [PROVENANCE.json](upstream/PROVENANCE.json) records original paths and SHA-256 hashes. The snapshot contains source, preprocessing scripts and available original documentation/notices; data, parameters, result files and notebook outputs are excluded. Notebook code cells are extracted into Python scripts where present. Upstream imports are not executed by ordinary package discovery.

Upstream sources retain their authors' attribution and license notices.
