import os
import sys

import torch
from torch import nn

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
import d2l

corpus, vocab = d2l.load_corpus_time_machine()
print(len(corpus), len(vocab))
for i in range(10):
    print(corpus[i])
