import math
import torch
from torch import nn

import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
import d2l

num_hiddens, num_heads = 100, 5
attention = d2l.MultiHeadAttention(num_hiddens, num_hiddens, num_hiddens, num_hiddens, num_heads, 0.5)
attention.eval()
print(attention)

batch_size, num_queries, valid_lens = 2, 4, torch.tensor([3, 2])
X = torch.ones((batch_size, num_queries, num_hiddens))
print(attention(X, X, X, valid_lens).shape)

encoding_dim, num_steps = 32, 60
pos_encoding = d2l.PositionalEncoding(encoding_dim, 0)
pos_encoding.eval()
X = pos_encoding(torch.zeros((1, num_steps, encoding_dim)))
P = pos_encoding.P[:, :X.shape[1], :]
d2l.plot(torch.arange(num_steps), P[0:, :, 6:10].T, xlabel='Row (position)',
         figsize=(6, 2.5), legend=["Col %d" %d for d in torch.arange(6, 10)])
