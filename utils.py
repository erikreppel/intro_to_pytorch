'''Contains classes and functions I want to re-use between notebooks'''
import torch
import torch.utils.data as data
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
from torchvision.datasets import MNIST

import numpy as np
import gzip
import os
import PIL


class FashionMNIST(data.Dataset):
    '''Implement Dataset with FashionMnist'''

    def __init__(self, fashionmnist_dir, kind='train'):
        if kind == 'test':
            kind = 't10k'

        labels_path = os.path.join(
            fashionmnist_dir, '%s-labels-idx1-ubyte.gz' % kind)
        images_path = os.path.join(
            fashionmnist_dir, '%s-images-idx3-ubyte.gz' % kind)

        with gzip.open(labels_path, 'rb') as lbpath:
            labels = np.frombuffer(lbpath.read(), dtype=np.uint8, offset=8)

        with gzip.open(images_path, 'rb') as imgpath:
            images = np.frombuffer(
                imgpath.read(), dtype=np.uint8, offset=16).reshape(len(labels), 784)

        self.labels = torch.from_numpy(labels).long()
        self.images = torch.from_numpy(images)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, i):
        return self.images[i], self.labels[i]


class SimpleClassifier(nn.Module):
    def __init__(self, in_dim, h_dim, out_dim):
        super(SimpleClassifier, self).__init__()
        self.l1 = nn.Linear(in_dim, h_dim)
        self.l2 = nn.Linear(h_dim, out_dim)

    def forward(self, x):
        x = self.l1(x)
        x = F.sigmoid(x)

        x = F.softmax(self.l2(x))
        return x


class FashionMNIST2D(MNIST):
    '''Implement Dataset with FashionMnist'''
    urls = [
        'http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/train-images-idx3-ubyte.gz',
        'http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/train-labels-idx1-ubyte.gz',
        'http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/t10k-images-idx3-ubyte.gz',
        'http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/t10k-labels-idx1-ubyte.gz',
    ]

class FashionModel(nn.Module):
    def __init__(self):
        super(FashionModel, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)

        self.conv2_drop = nn.Dropout2d()

        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, 10)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))

        x = x.view(-1, 320)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return F.log_softmax(x)