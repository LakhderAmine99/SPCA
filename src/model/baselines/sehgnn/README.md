# SeHGNN

Paper: [Simple and Efficient Heterogeneous Graph Neural Network](https://arxiv.org/abs/2207.02547). Original source: [https://github.com/ICT-GIMLab/SeHGNN](https://github.com/ICT-GIMLab/SeHGNN).

`model.py` contains the runnable architecture, `layers.py` the reusable layers/kernel imports, and `preprocessing.py` where model-specific preparation is needed. Shared metapath connectivity is implemented in `src/graph.py` and input binding in `src/experiments/models.py`.

Fixed mean-propagated metapath features, per-path MLPs, gated Transformer semantic fusion and residual task layers. The input channels contain the selected paths and raw target features.

```python
from src.model.baselines.sehgnn import SeHGNN
# Construct with dimensions/options documented in model.py, then:
model()  # prepared features are nonpersistent model buffers
```

The shared runner obtains a 128-dimensional embedding and applies a linear NC head or the paper's LP dot-product decoder. `python run_dblp.py --models sehgnn --dry-run` inspects the configuration without loading data. Constructor settings, applicable grid axes, defaults and shared experiment settings are described in [paper_protocol.md](../../../../docs/paper_protocol.md). Synthetic forward/backward and task checks cover this model in `tests/test_paper_models.py`.
