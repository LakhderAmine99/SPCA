# GTC

Paper: [GTC: GNN-Transformer Co-contrastive Learning for Self-supervised Heterogeneous Graph Representation](https://doi.org/10.1016/j.neunet.2024.106645). Original source: [https://github.com/PHD-lanyu/GTC](https://github.com/PHD-lanyu/GTC).

`model.py` contains the runnable architecture, `layers.py` the reusable layers/kernel imports, and `preprocessing.py` where model-specific preparation is needed. Shared metapath connectivity is implemented in `src/graph.py` and input binding in `src/experiments/models.py`.

Hop2Token propagation, a Transformer per selected metapath, semantic fusion, a relation-wise GNN view and co-contrastive learning. transformer.py preserves original Transformer layer code; the adapter corrects exchanged head/layer arguments and singleton batch squeezing. NC evaluation uses the common detached-head cadence documented in the protocol.

```python
from src.model.baselines.gtc import GTC
# Construct with dimensions/options documented in model.py, then:
model(x, metapath_graphs, hop_tokens); model.contrastive_loss(x, metapath_graphs, hop_tokens, positive_edges)
```

The shared runner obtains a 128-dimensional embedding and applies a linear NC head or the paper's LP dot-product decoder. `python run_dblp.py --models gtc --dry-run` inspects the configuration without loading data. Constructor settings, applicable grid axes, defaults and all common protocol adaptations are described in [paper_protocol.md](../../../../docs/paper_protocol.md). Synthetic forward/backward and task checks cover this model in `tests/test_paper_models.py`.

## Upstream snapshot

Pinned commit: `1c4b3b979a2688d4a2b6da8bf48b9aea34b80fbb`. [PROVENANCE.json](upstream/PROVENANCE.json) records original paths and SHA-256 hashes. The snapshot contains source, preprocessing scripts and available original documentation/notices; data, parameters, result files and notebook outputs are excluded. Notebook code cells are extracted into Python scripts where present. Upstream imports are not executed by ordinary package discovery.

The adapter is a documented translation/adaptation, not a claim of bit-for-bit reproduction or verified benchmark scores. Retain upstream attribution and license notices; the repository's MIT license does not relicense the snapshot. Where upstream supplies no license file, no substitute license is asserted here.
