# HGT

Paper: [Heterogeneous Graph Transformer](https://arxiv.org/abs/2003.01332). Original source: [https://github.com/acbull/pyHGT](https://github.com/acbull/pyHGT).

`model.py` contains the runnable architecture, `layers.py` the reusable layers/kernel imports, and `preprocessing.py` where model-specific preparation is needed. Shared metapath connectivity is implemented in `src/graph.py` and input binding in `src/experiments/models.py`.

Typed projections followed by PyG HGTConv and a task projection. In the paper runner each selected metapath becomes a relation over the receiving endpoint type; full typed graphs remain supported by the Python model.

```python
from src.model.baselines.hgt import HGT
# Construct with dimensions/options documented in model.py, then:
model(x_dict, edge_index_dict)
```

The shared runner obtains a 128-dimensional embedding and applies a linear NC head or the paper's LP dot-product decoder. `python run_dblp.py --models hgt --dry-run` inspects the configuration without loading data. Constructor settings, applicable grid axes, defaults and shared experiment settings are described in [paper_protocol.md](../../../../docs/paper_protocol.md). Synthetic forward/backward and task checks cover this model in `tests/test_paper_models.py`.
