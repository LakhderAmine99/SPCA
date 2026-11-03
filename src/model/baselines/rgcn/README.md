# RGCN

Paper: [Modeling Relational Data with Graph Convolutional Networks](https://arxiv.org/abs/1703.06103). Original source: [https://github.com/tkipf/relational-gcn](https://github.com/tkipf/relational-gcn).

`model.py` contains the runnable architecture, `layers.py` the reusable layers/kernel imports, and `preprocessing.py` where model-specific preparation is needed. Shared metapath connectivity is implemented in `src/graph.py` and input binding in `src/experiments/models.py`.

Stacked PyG RGCNConv layers with optional basis decomposition. The shared runner supplies an endpoint graph with one relation ID per selected metapath.

```python
from src.model.baselines.rgcn import RGCN
# Construct with dimensions/options documented in model.py, then:
model(x, edge_index, edge_type)
```

The shared runner obtains a 128-dimensional embedding and applies a linear NC head or the paper's LP dot-product decoder. `python run_dblp.py --models rgcn --dry-run` inspects the configuration without loading data. Constructor settings, applicable grid axes, defaults and shared experiment settings are described in [paper_protocol.md](../../../../docs/paper_protocol.md). Synthetic forward/backward and task checks cover this model in `tests/test_paper_models.py`.
