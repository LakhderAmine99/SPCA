# HeCo

Paper: [Self-supervised Heterogeneous Graph Neural Network with Co-contrastive Learning](https://arxiv.org/abs/2105.09111). Original source: [https://github.com/liun-online/HeCo](https://github.com/liun-online/HeCo).

`model.py` contains the runnable architecture, `layers.py` the reusable layers/kernel imports, and `preprocessing.py` where model-specific preparation is needed. Shared metapath connectivity is implemented in `src/graph.py` and input binding in `src/experiments/models.py`.

Metapath GCN and schema attention views retain a bidirectional multi-positive contrastive objective. For common inputs the schema view uses fixed metapath relations, eight heads, and all neighbors instead of upstream CUDA-only type-neighbor sampling. NC uses a detached representation plus trainable output/evaluation projections, advanced alongside the contrastive branch for validation checkpoint selection.

```python
from src.model.baselines.heco import HeCo
# Construct with dimensions/options documented in model.py, then:
model(x, metapath_graphs); model.contrastive_loss(x, metapath_graphs, positive_edges)
```

The shared runner obtains a 128-dimensional embedding and applies a linear NC head or the paper's LP dot-product decoder. `python run_dblp.py --models heco --dry-run` inspects the configuration without loading data. Constructor settings, applicable grid axes, defaults and all common protocol adaptations are described in [paper_protocol.md](../../../../docs/paper_protocol.md). Synthetic forward/backward and task checks cover this model in `tests/test_paper_models.py`.

## Upstream snapshot

Pinned commit: `20e8ca2042838877b2c815279920218b6bbea1db`. [PROVENANCE.json](upstream/PROVENANCE.json) records original paths and SHA-256 hashes. The snapshot contains source, preprocessing scripts and available original documentation/notices; data, parameters, result files and notebook outputs are excluded. Notebook code cells are extracted into Python scripts where present. Upstream imports are not executed by ordinary package discovery.

The adapter is a documented translation/adaptation, not a claim of bit-for-bit reproduction or verified benchmark scores. Retain upstream attribution and license notices; the repository's MIT license does not relicense the snapshot. Where upstream supplies no license file, no substitute license is asserted here.
