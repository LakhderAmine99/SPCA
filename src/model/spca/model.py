"""Classification interface for the SPCA model and controlled variants."""
from .encoders import SubpathClassifier, SPCAConv2, RegularizedSPCAConv, StackedSPCA
from .layers import SPCALayer


class SPCA(SubpathClassifier):
    def __init__(self, in_channels_dict, hidden_channels, num_classes, metadata,
                 metapaths, subpaths_dict=None, *, variant='base', context_mode='subpath',
                 aggregator='concat-mlp', heads=1, dropout=0.5, num_layers=2,
                 context_positions='intermediate', normalization=True, node_chunk_size=256):
        encoders = {'base': SPCALayer, 'legacy': SPCAConv2, 'regularized': RegularizedSPCAConv,
                    'stacked': StackedSPCA}
        if variant not in encoders:
            raise ValueError(f'Unknown SPCA variant: {variant}')
        options = dict(heads=heads, dropout=dropout, aggregator=aggregator,
                       context_mode=context_mode)
        if variant == 'base':
            options.pop('heads')  # Eq. (1) has one full-vector dot product, no head parameters.
            options.update(context_positions=context_positions, normalization=normalization,
                           node_chunk_size=node_chunk_size)
        elif subpaths_dict is None:
            subpaths_dict = {tuple(p): [tuple(p)[1:-1]] for p in metapaths}
        if variant == 'stacked':
            if num_layers < 1:
                raise ValueError('num_layers must be positive')
            options['num_layers'] = num_layers
        encoder = encoders[variant](in_channels_dict, hidden_channels, metadata,
                                    metapaths, subpaths_dict, **options)
        super().__init__(encoder, hidden_channels, num_classes)

    def encode(self, x_dict, edge_index_dict=None, **kwargs):
        return self.encoder(x_dict, edge_index_dict, **kwargs)

    def forward(self, x_dict, edge_index_dict=None, **kwargs):
        return self.classifier(self.encode(x_dict, edge_index_dict, **kwargs))

    def reset_parameters(self):
        if hasattr(self.encoder, 'reset_parameters'):
            self.encoder.reset_parameters()
        self.classifier.reset_parameters()
