"""Construct shared architectures with a task-selected output width."""
from .baselines.registry import ALIASES, BASELINES, get_baseline
from .spca import SPCA


def build_backbone(name, output_channels, **kwargs):
    if not isinstance(output_channels, int) or output_channels < 1:
        raise ValueError('output_channels must be a positive integer')
    key = ALIASES.get(name.lower(), name.lower())
    if key == 'spca':
        cls = SPCA
    else:
        cls = get_baseline(key)
        if BASELINES[key].integration in {'upstream', 'external-required'}:
            raise NotImplementedError(
                f'{name} requires an upstream task implementation; its process launcher '
                'cannot be used as a PyTorch encoder.'
            )
    width_key = 'num_classes' if key in {'spca', 'sehgnn', 'simplehgn', 'hettree'} else 'out_channels'
    if 'out_channels' in kwargs or 'num_classes' in kwargs:
        raise ValueError('Specify the output width through the task factory only')
    return cls(**kwargs, **{width_key: output_channels})
