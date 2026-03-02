import d2l

train_iter, test_iter = d2l.load_data_fashion_mnist(23, resize=64)
for X, y in train_iter:
    print(X.shape, X.dtype, y.shape, y.dtype)
    break
