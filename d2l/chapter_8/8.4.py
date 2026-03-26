import os
import sys

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
import d2l

X, W_xh = torch.normal(0, 1, (3, 1)), torch.normal(0, 1, (1, 4))
H, W_hh = torch.normal(0, 1, (3, 4)), torch.normal(0, 1, (4, 4))
print(torch.matmul(X, W_xh) + torch.matmul(H, W_hh))
print(torch.matmul(torch.cat((X, H), 1), torch.cat((W_xh, W_hh), 0)))
