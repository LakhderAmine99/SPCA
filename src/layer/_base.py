
# class SubpathHANConv2(nn.Module):
#     def __init__(
#         self,
#         in_channels_dict,
#         out_channels,
#         metadata,
#         metapaths,
#         subpaths_dict,
#         heads=1,
#         dropout=0.0,
#         aggregator='concat-mlp',
#     ):
#         super().__init__()
#         self.out_channels = out_channels
#         self.heads = heads
#         self.metadata = metadata
#         self.metapaths = metapaths
#         self.subpaths_dict = subpaths_dict
#         self.dropout = dropout
#         self.aggregator = aggregator

#         self.type_emb = nn.ParameterDict()
#         for node_type in metadata[0]:
#             self.type_emb[node_type] = nn.Parameter(torch.randn(out_channels))

#         self.proj = nn.ModuleDict()
#         for node_type, in_channels in in_channels_dict.items():
#             self.proj[node_type] = Linear(in_channels, out_channels)

#         self.subpath_mlps = nn.ModuleDict()
#         if aggregator == 'concat-mlp':
#             for metapath in metapaths:
#                 for subpath in subpaths_dict[metapath]:
#                     key = '_'.join(subpath)
#                     self.subpath_mlps[key] = nn.Sequential(
#                         nn.Linear(len(subpath) * out_channels, out_channels),
#                         nn.ReLU(),
#                         nn.Linear(out_channels, out_channels),
#                         nn.LayerNorm(out_channels)
#                     )

#         self.final_lin = Linear(len(metapaths) * out_channels, out_channels)
#         self.norm = nn.LayerNorm(out_channels)

#     def _message_pass(self, x_src, x_dst, edge_index):
#         alpha = (x_src[edge_index[0]] * x_dst[edge_index[1]]).sum(dim=-1, keepdim=True)
#         alpha = F.leaky_relu(alpha)
#         alpha = softmax(alpha, edge_index[1], num_nodes=x_dst.size(0))
#         alpha = F.dropout(alpha, p=self.dropout, training=self.training)
#         h = x_src[edge_index[0]] * alpha
#         out = torch.zeros_like(x_dst)
#         out.index_add_(0, edge_index[1], h)
#         return out

#     def forward(self, x_dict, edge_index_dict):
#         x_proj = {k: self.proj[k](v) for k, v in x_dict.items()}

#         metapath_embs = []
#         for metapath in self.metapaths:
#             assert len(metapath) % 2 == 1, "Metapath must be odd-length"
#             num_hops = len(metapath) // 2
#             src_type = metapath[0]
#             dst_type = metapath[-1]

#             h = x_proj[src_type]
#             for i in range(num_hops):
#                 edge1 = (metapath[i * 2], 'to', metapath[i * 2 + 1])
#                 edge2 = (metapath[i * 2 + 1], 'to', metapath[i * 2 + 2])
#                 h = self._message_pass(h, x_proj[metapath[i * 2 + 1]], edge_index_dict[edge1])
#                 h = self._message_pass(h, x_proj[metapath[i * 2 + 2]], edge_index_dict[edge2])
#             h = self.norm(h)
#             within_emb = h

#             sub_embs = []
#             for subpath in self.subpaths_dict[metapath]:
#                 type_embs = [self.type_emb[t] for t in subpath]
#                 if self.aggregator == 'concat-mlp':
#                     key = '_'.join(subpath)
#                     emb = torch.cat(type_embs, dim=-1)
#                     emb = self.subpath_mlps[key](emb)
#                 elif self.aggregator == 'mean':
#                     emb = torch.stack(type_embs).mean(dim=0)
#                 emb = emb.unsqueeze(0).expand(within_emb.size(0), -1)
#                 sub_embs.append(emb)
#             subpath_emb = torch.stack(sub_embs, dim=0).mean(dim=0)

#             metapath_embs.append(within_emb + subpath_emb)

#         out = torch.cat(metapath_embs, dim=-1)
#         out = self.final_lin(out)
#         return out

from typing import Any, List, Union, Dict, Tuple
import torch.nn as nn

MetaData = List[List[str]]

MetaPath = Tuple[str, str, str]

Agg = Union[]

class EntityTypeEncoder(nn.Module):
    def __init__(self, *args, **kwargs) -> None: # type: ignore
        super().__init__(*args, **kwargs) # type: ignore
        
    def forward(self):
        ...

class SubPathEncoder(nn.Module):
    pass

class SubPathContextAwareConv(nn.Module):
    """SubPath Conv Module
    """
    def __init__(
        self,
        in_channels_dict:Dict[str, int],
        out_channels: int,
        metadata: MetaData,
        metapaths: List[MetaPath],
        dropout: float = 0.5,
        **kwargs: Any
    ) -> None:
        pass