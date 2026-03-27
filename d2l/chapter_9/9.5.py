import os
import sys
import torch
from torch import nn

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
import d2l

raw_text = d2l.read_data_nmt()

text = d2l.preprocess_nmt(raw_text)
source, target = d2l.tokenize_nmt(text)

# d2l.show_list_len_pair_hist(['source', 'target'], '# tokens per sequence', 'count', source, target)
src_vocab = d2l.Vocab(source, min_freq=2,
                      reserved_tokens=['<pad>', '<bos>', '<eos>'])
print(len(src_vocab))
print(d2l.truncate_pad(src_vocab[source[0]], 10, src_vocab['<pad>']))

train_iter, src_vocab, tgt_vocab = d2l.load_data_nmt(batch_size=2, num_steps=8)
for X, X_valid_len, Y, Y_valid_len in train_iter:
    print('X:', X.type(torch.int32))
    print('X的有效长度:', X_valid_len)
    print('Y:', Y.type(torch.int32))
    print('Y的有效长度:', Y_valid_len)
    break
