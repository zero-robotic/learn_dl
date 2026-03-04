import os
import sys

import torch
from torch import nn

# 使用项目根目录下的本地 d2l.py，而不是系统里安装的 d2l 包
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
import d2l

net = nn.Sequential(
    nn.Conv2d(1, 6, kernel_size=5, padding=2), nn.Sigmoid(),
    nn.AvgPool2d(kernel_size=2, stride=2),
    nn.Conv2d(6, 16, kernel_size=5),  nn.Sigmoid(),
    nn.AvgPool2d(kernel_size=2, stride=2),
    nn.Flatten(),
    nn.Linear(16 * 5 * 5, 120), nn.Sigmoid(),
    nn.Linear(120, 84), nn.Sigmoid(),
    nn.Linear(84, 10))

X = torch.rand(size=(1, 1, 28, 28), dtype=torch.float32)
for layer in net:
    X = layer(X)
    print(layer.__class__.__name__, 'output shape: \t', X.shape)


batch_size = 256
#train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size=batch_size)
train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size = batch_size)

lr, num_epochs = 0.9, 10
d2l.train_ch6(net, train_iter, test_iter, num_epochs, lr, d2l.try_gpu())
