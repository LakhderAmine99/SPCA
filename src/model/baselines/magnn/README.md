# MAGNN

Paper: [MAGNN: Metapath Aggregated Graph Neural Network for Heterogeneous Graph Embedding](https://arxiv.org/abs/2002.01680). Original source: [https://github.com/cynricfu/MAGNN](https://github.com/cynricfu/MAGNN).

`model.py` contains the runnable architecture, `layers.py` the reusable layers/kernel imports, and `preprocessing.py` where model-specific preparation is needed. Shared metapath connectivity is implemented in `src/graph.py` and input binding in `src/experiments/models.py`.

Type-specific projections, ordered metapath-instance encoders, destination-normalized instance attention and semantic attention. RotatE0 is the default; mean, linear, GRU and LSTM are available. Reverse relations share conjugate rotations. Prefix-count sampling caps complete instances per receiver at 100 by default. Set `max_instances=null` for exhaustive instances.

```python
from src.model.baselines.magnn import MAGNN
# Construct with dimensions/options documented in model.py, then:
model(x_dict, instances_by_metapath)
```

The shared runner obtains a 128-dimensional embedding and applies a linear NC head or the paper's LP dot-product decoder. `python run_dblp.py --models magnn --dry-run` inspects the configuration without loading data. Constructor settings, applicable grid axes, defaults and shared experiment settings are described in [paper_protocol.md](../../../../docs/paper_protocol.md). Synthetic forward/backward and task checks cover this model in `tests/test_paper_models.py`.
