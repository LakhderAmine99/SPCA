# SeHGNN

Paper: [Simple and Efficient Heterogeneous Graph Neural Network](https://arxiv.org/abs/2207.02547). Original source: [https://github.com/ICT-GIMLab/SeHGNN](https://github.com/ICT-GIMLab/SeHGNN).

`model.py` contains the runnable architecture, `layers.py` the reusable layers/kernel imports, and `preprocessing.py` where model-specific preparation is needed. Shared metapath connectivity is implemented in `src/graph.py` and input binding in `src/experiments/models.py`.

Preserves the existing feature-only adaptation: fixed mean-propagated metapath features, per-path MLPs, gated Transformer semantic fusion and residual task layers. Selected paths and the raw root are retained. Label propagation and multi-stage pseudo-label training are disabled, so this is not the complete upstream multi-stage protocol.

```python
from src.model.baselines.sehgnn import SeHGNN
# Construct with dimensions/options documented in model.py, then:
model()  # prepared features are nonpersistent model buffers
```

The shared runner obtains a 128-dimensional embedding and applies a linear NC head or the paper's LP dot-product decoder. `python run_dblp.py --models sehgnn --dry-run` inspects the configuration without loading data. Constructor settings, applicable grid axes, defaults and all common protocol adaptations are described in [paper_protocol.md](../../../../docs/paper_protocol.md). Synthetic forward/backward and task checks cover this model in `tests/test_paper_models.py`.

## Upstream snapshot

Pinned commit: `e92bd37d0b803457339555684f139b4c8f3e160d`. [PROVENANCE.json](upstream/PROVENANCE.json) records original paths and SHA-256 hashes. The snapshot contains source, preprocessing scripts and available original documentation/notices; data, parameters, result files and notebook outputs are excluded. Notebook code cells are extracted into Python scripts where present. Upstream imports are not executed by ordinary package discovery.

The adapter is a documented translation/adaptation, not a claim of bit-for-bit reproduction or verified benchmark scores. Retain upstream attribution and license notices; the repository's MIT license does not relicense the snapshot. Where upstream supplies no license file, no substitute license is asserted here.
