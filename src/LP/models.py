"""LP encoders plus edge decoders; graph inputs and query edges stay separate."""
from torch import nn
from src.model.tasks import build_backbone
from .decoders import DotProductDecoder, MLPDecoder


class LinkPredictor(nn.Module):
    """Use one encoder for same-type links or two for different endpoint types.

    Each encoder returns a [num_endpoint_nodes, embedding_channels] matrix.
    destination_inputs is an (args_tuple, kwargs_dict) pair for a second encoder.
    Query edge indices use the local row numbering of these two matrices.
    """

    def __init__(self, encoder, embedding_channels, *, decoder='mlp', destination_encoder=None):
        super().__init__()
        self.encoder = encoder
        self.destination_encoder = destination_encoder
        if decoder not in {'mlp', 'dot'}:
            raise ValueError('decoder must be mlp or dot')
        if embedding_channels < 1:
            raise ValueError('embedding_channels must be positive')
        self.embedding_channels = embedding_channels
        self.decoder = MLPDecoder(embedding_channels) if decoder == 'mlp' else DotProductDecoder()

    def forward(self, *args, edge_label_index, destination_inputs=None, **kwargs):
        if (self.destination_encoder is None) != (destination_inputs is None):
            raise ValueError('A destination encoder and destination_inputs must be supplied together')
        source = self.encoder(*args, **kwargs)
        if self.destination_encoder is None:
            destination = source
        else:
            destination_args, destination_kwargs = destination_inputs
            destination = self.destination_encoder(*destination_args, **destination_kwargs)
        if source.ndim != 2 or destination.ndim != 2 or (
            source.shape[1] != self.embedding_channels or destination.shape[1] != self.embedding_channels
        ):
            raise ValueError('Encoders must return matrices with the configured embedding width')
        return self.decoder(source, destination, edge_label_index)


def build_model(name, embedding_channels, *, decoder='mlp', destination_encoder=None, **model_kwargs):
    """Repurpose a native model's output projection as an LP embedding head.

    No classification loss is applied. Existing architectural output transforms
    (e.g. Simple-HGN normalization) are retained. Cached features and graphs must
    be built from the training graph; HetTree label propagation is disabled.
    """
    if name.lower() == 'hettree' and model_kwargs.get('label_features') is not None:
        raise ValueError('LP HetTree must use feature-only inputs, without NC label propagation')
    encoder = build_backbone(name, embedding_channels, **model_kwargs)
    return LinkPredictor(encoder, embedding_channels, decoder=decoder,
                         destination_encoder=destination_encoder)
