import math
import os

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.nn.parameter import Parameter


def masked_softmax(X, valid_len):
    """
    masked softmax for attention scores
    args:
        X: 3-D tensor, valid_len: 1-D or 2-D tensor
    """
    if valid_len is None:
        return nn.functional.softmax(X, dim=-1)
    else:
        shape = X.shape
        if valid_len.dim() == 1:
            valid_len = torch.repeat_interleave(
                valid_len, repeats=shape[1], dim=0)
        else:
            valid_len = valid_len.reshape(-1)
        # Fill masked elements with a large negative, whose exp is 0
        X = X.reshape(-1, shape[-1])
        for count, row in enumerate(X):
            row[int(valid_len[count]):] = -1e6
        return nn.functional.softmax(X.reshape(shape), dim=-1)


class SelfAttentionLayer(nn.Module):
    """
    Self-attention layer. no scale_factor d_k
    """

    def __init__(self, in_channels, global_graph_width):
        super(SelfAttentionLayer, self).__init__()
        self.in_channels = in_channels
        
        self.q_lin = nn.Linear(in_channels, global_graph_width)
        self.k_lin = nn.Linear(in_channels, global_graph_width)
        self.v_lin = nn.Linear(in_channels, global_graph_width)

        self._norm_fact = 1 / math.sqrt(in_channels)


    def forward(self, x):
        query = self.q_lin(x)
        key = self.k_lin(x)
        value = self.v_lin(x)
        scores = torch.bmm(query, key.transpose(1, 2)) * self._norm_fact
        scores = nn.functional.softmax(scores, dim=-1)
        
        return torch.bmm(scores,value)
