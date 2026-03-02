import torch
import torchvision
from torch.utils import data
from torchvision import transforms
import d2l

trans = transforms.ToTensor()
mnist_train = torchvision.datasets.FashionMNIST(
    root="../data", train=True, transform=trans, download=True)
mnist_test = torchvision.datasets.FashionMNIST(
    root="../data", train=False, transform=trans, download=True)
print(len(mnist_train), len(mnist_test))
print(mnist_train[0][0].shape)

# X, y = next(iter(data.DataLoader(mnist_train, batch_size=18)))
# d2l.show_images(X.reshape(18, 28, 28), 2, 9, titles=d2l.get_fashion_mnist_labels(y))

batch_size = 256
train_iter = data.DataLoader(mnist_train, batch_size, shuffle=True)

timer = d2l.Timer()
for X, y in train_iter:
    continue
print(f'{timer.stop():.2f} sec')
