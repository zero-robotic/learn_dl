import os
import sys

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
import d2l

my_seq = list(range(35))
for X, Y in d2l.seq_data_iter_random(my_seq, batch_size=2, num_steps=5):
    print('X: ', X, '\nY:', Y)

print('----------------------------')

for X, Y in d2l.seq_data_iter_sequential(my_seq, batch_size=2, num_steps=5):
    print('X: ', X, '\nY:', Y)

print('=============================')
