# GTC

Paper: [GTC: GNN-Transformer Co-contrastive Learning for Self-supervised Heterogeneous Graph Representation](https://doi.org/10.1016/j.neunet.2024.106645). Original source: [https://github.com/PHD-lanyu/GTC](https://github.com/PHD-lanyu/GTC).

`model.py` contains the runnable architecture, `layers.py` the reusable layers/kernel imports, and `preprocessing.py` where model-specific preparation is needed. Shared metapath connectivity is implemented in `src/graph.py` and input binding in `src/experiments/models.py`.

Hop2Token propagation, a Transformer per selected metapath, semantic fusion, a relation-wise GNN view and co-contrastive learning. `transformer.py` provides the Transformer layers with configurable head count and depth. NC uses a detached linear evaluation head alongside the contrastive branch for validation checkpoint selection.

```python
from src.model.baselines.gtc import GTC
# Construct with dimensions/options documented in model.py, then:
model(x, metapath_graphs, hop_tokens); model.contrastive_loss(x, metapath_graphs, hop_tokens, positive_edges)
```

The shared runner obtains a 128-dimensional embedding and applies a linear NC head or the paper's LP dot-product decoder. `python run_dblp.py --models gtc --dry-run` inspects the configuration without loading data. Constructor settings, applicable grid axes, defaults and shared experiment settings are described in [paper_protocol.md](../../../../docs/paper_protocol.md). Synthetic forward/backward and task checks cover this model in `tests/test_paper_models.py`.
