import math
import torch
from torch import nn

import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
import d2l

print(d2l.masked_softmax(torch.rand(2, 2, 4), torch.tensor([2, 3])))
print(d2l.masked_softmax(torch.rand(2, 2, 4), torch.tensor([[1, 3], [2, 4]])))

queries, keys = torch.normal(0, 1, (2, 1, 20)), torch.ones((2, 10, 2))
values = torch.arange(40, dtype=torch.float32).reshape(1, 10, 4).repeat(2, 1, 1)
valid_lens = torch.tensor([2, 6])
attention = d2l.AdditiveAttention(key_size=2, query_size=20, num_hiddens=8, dropout=0.1)
attention.eval()
print(attention(queries, keys, values, valid_lens))
d2l.show_heatmaps(attention.attention_weights.reshape((1, 1, 2, 10)),
                  xlabel='Keys', ylabel='Queries')

queries = torch.normal(0, 1, (2, 1, 2))
attention = d2l.DotProductAttention(dropout=0.5)
attention.eval()
print(attention(queries, keys, values, valid_lens))
