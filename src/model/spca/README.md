# SPCA

Paper: *SPCA: SubPath Semantic Encoding for Context Aware Heterogeneous Graph Neural Networks* (Lakhder et al., 2026 manuscript). [Project source](https://github.com/LakhderAmine99/SPCA), [citation](../../../CITATION.cff).

`model.py` supplies the classifier. `layers.py` contains the current method:

- `DotProductConv(MessagePassing)` implements Eqs. 1–2 over metapath endpoint neighbors.
- `SubpathContext` shares node-type embeddings and reuses one encoder per distinct ordered type sequence (Eqs. 3–4).
- `SPCALayer` projects typed inputs, normalizes structural aggregates, adds context and normalizes again, then concatenates/project views (Eqs. 5–7).

```python
from src.model.spca import SPCA, SPCALayer

model = SPCA(in_channels_dict, 128, num_classes, metadata, metapaths,
             dropout=0.5, context_positions='intermediate')
logits = model(x_dict, edge_index_dict)
# For repeated epochs, precompute connectivity once with compose_metapaths():
logits = model(x_dict, metapath_edge_index_dict=prepared_metapath_edges)
```

PyG edge indices use `[source, destination]` rows and `torch.long` IDs. Prepared graph keys are full node-type tuples. One layer returns embeddings for one receiving type. LP uses one encoder per endpoint and the final dot-product decoder. Context subpaths must be nonempty; both even- and odd-hop metapaths are accepted.

The default is endpoint-neighbor aggregation, intermediate-only Concat+MLP context, and both LayerNorm stages. `heads` on the classifier is retained for historical constructors; the paper's dot-product score has no independent attention heads. For ablations set `aggregator`, `context_mode`, `context_positions`, or `normalization`, or pass explicit `subpaths_dict` to the layer. `reset_parameters()` resets the current architecture. Attention batching preserves exact neighborhood normalization.

`encoders.py` preserves historical hop-wise and regularized/stacked APIs for compatibility. Select `variant='legacy'`, `'regularized'`, or `'stacked'` explicitly; these are not the default manuscript method. See [protocol notes](../../../docs/paper_protocol.md) for settings and tests.
