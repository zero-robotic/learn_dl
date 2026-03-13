import os
import sys

import collections
import re

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
import d2l

lines = d2l.read_time_machine()
print(f'# w文本总行数: {len(lines)}')
print(lines[0])
print(lines[10])

tokens = d2l.tokenize(lines)
for i in range(11):
    print(tokens[i])


vocab = d2l.Vocab(tokens)
print(list(vocab.token_to_idx.items())[:10])

for i in [0, 10]:
    print('文本:', tokens[i])
    print('索引:', vocab[tokens[i]])
