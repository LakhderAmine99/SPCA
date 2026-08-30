import torch
from torch_geometric.datasets import LastFM, DBLP

dataset = LastFM(root='./data/lastfm')

if __name__ == '__main__':
    data = dataset[0]
    print(data)