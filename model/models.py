import torch
import torch.nn as nn
import torch.nn.functional as F
from model import layers

class DiGCN(nn.Module):
    def __init__(self, n_feat, n_hid, n_class, enable_bias, K, droprate):
        super(DiGCN, self).__init__()
        self.graph_convs = nn.ModuleList()
        self.K = K
        self.graph_convs.append(layers.GraphConv(in_features=n_feat, out_features=n_hid, bias=enable_bias))
        for k in range(1, K-1):
            self.graph_convs.append(layers.GraphConv(in_features=n_hid, out_features=n_hid, bias=enable_bias))
        self.graph_convs.append(layers.GraphConv(in_features=n_hid, out_features=n_class, bias=enable_bias))
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=droprate)
        self.log_softmax = nn.LogSoftmax(dim=1)

    def forward(self, x, filter):
        for k in range(self.K-1):
            x = self.graph_convs[k](x, filter)
            x = self.relu(x)
        x = self.dropout(x)
        x = self.graph_convs[-1](x, filter)
        x = self.log_softmax(x)

        return x