import os
import sys
import torch

sys.path.insert(0, os.path.join(os.path.dirname('__file__'), '..'))
import d2l

attention_weights = torch.eye(10).reshape((1, 1, 10, 10))
d2l.show_heatmaps(attention_weights, xlabel='Keys', ylabel='Queries')
