import math
import pandas as pd
import torch
from torch import nn

import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
import d2l

ffn = d2l.PositionWiseFFN(4, 4, 8)
ffn.eval()
print(ffn(torch.ones((2, 3, 4)))[0])

add_norm = d2l.AddNorm([3, 4], 0.5)
add_norm.eval()
print(add_norm(torch.ones((2, 3, 4)), torch.ones((2, 3, 4))).shape)
