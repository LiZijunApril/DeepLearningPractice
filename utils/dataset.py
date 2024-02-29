# %matplotlib inline
import random
import torchvision
import torch

def load_array(data_arrays, batch_szie, is_train=True):
    """构造一个PyTorch数据迭代器"""
    dataset = torch.utils.data.TensorDataset(*data_arrays)
    return torch.utils.data.DataLoader(dataset, batch_szie, shuffle=is_train)

def load_data_fashion_mnist(batch_size, resize=None, root='~/Datasets/FashionMNIST'):
    """Download the fashion mnist dataset and then load into memory."""
    trans = [torchvision.transforms.ToTensor()]
    if resize:
        trans.insert(0, torchvision.transforms.Resize(size=resize))
    trans = torchvision.transforms.Compose(trans)

    mnist_train = torchvision.datasets.FashionMNIST(root=root, train=True, download=True, transform=trans)
    mnist_test = torchvision.datasets.FashionMNIST(root=root, train=False, download=True, transform=trans)

    train_iter = torch.utils.data.DataLoader(mnist_train, batch_size=batch_size, shuffle=True, num_workers=4)
    test_iter = torch.utils.data.DataLoader(mnist_test, batch_size=batch_size, shuffle=False, num_workers=4)

    return train_iter, test_iter
