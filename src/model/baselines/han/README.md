# HAN

Paper: [Heterogeneous Graph Attention Network](https://arxiv.org/abs/1903.07293). Original source: [https://github.com/Jhy1993/HAN](https://github.com/Jhy1993/HAN).

`model.py` contains the runnable architecture, `layers.py` the reusable layers/kernel imports, and `preprocessing.py` where model-specific preparation is needed. Shared metapath connectivity is implemented in `src/graph.py` and input binding in `src/experiments/models.py`.

PyG HANConv performs node-level and semantic attention on precomputed target-to-target metapath graphs. Hidden projection and per-head width are independently configurable. The task projection returns the requested output width.

```python
from src.model.baselines.han import HAN
# Construct with dimensions/options documented in model.py, then:
model(x, metapath_edge_index_dict)
```

The shared runner obtains a 128-dimensional embedding and applies a linear NC head or the paper's LP dot-product decoder. `python run_dblp.py --models han --dry-run` inspects the configuration without loading data. Constructor settings, applicable grid axes, defaults and shared experiment settings are described in [paper_protocol.md](../../../../docs/paper_protocol.md). Synthetic forward/backward and task checks cover this model in `tests/test_paper_models.py`.
