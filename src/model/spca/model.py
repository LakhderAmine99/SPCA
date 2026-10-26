"""Classification interface for the SPCA model and controlled variants."""
from .encoders import SubpathClassifier, SubpathHANConv2, RegularizedSubpathHANConv, StackedSubpathHAN


class SPCA(SubpathClassifier):
    def __init__(self, in_channels_dict, hidden_channels, num_classes, metadata,
                 metapaths, subpaths_dict, *, variant='base', context_mode='subpath',
                 aggregator='concat-mlp', heads=1, dropout=0.0, num_layers=2):
        encoders = {'base': SubpathHANConv2, 'regularized': RegularizedSubpathHANConv,
                    'stacked': StackedSubpathHAN}
        if variant not in encoders:
            raise ValueError(f'Unknown SPCA variant: {variant}')
        options = dict(heads=heads, dropout=dropout, aggregator=aggregator,
                       context_mode=context_mode)
        if variant == 'stacked':
            if num_layers < 1:
                raise ValueError('num_layers must be positive')
            options['num_layers'] = num_layers
        encoder = encoders[variant](in_channels_dict, hidden_channels, metadata,
                                    metapaths, subpaths_dict, **options)
        super().__init__(encoder, hidden_channels, num_classes)
