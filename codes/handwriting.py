#!/usr/bin/env python3

import numpy
import torch
import torchvision
from torch import nn, optim
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

batch_size = 64
learning_rate = 1e-3
num_epoches = 5

# 定义三层全连接神经网络


class simpleNet(nn.Module):
    def __init__(self, in_dim, n_hidden_1, n_hidden_2, out_dim):
        '''
        param in_dim: 输入的维度
        param n_hidden_1: 第一层网络的神经元个数
        param n_hidden_2: 第二层网络的神经元个数
        param out_dim: 输出层神经元的个数
        '''
        super(simpleNet, self).__init__()
        self.layer1 = nn.Linear(in_dim, n_hidden_1)
        self.layer2 = nn.Linear(n_hidden_1, n_hidden_2)
        self.layer3 = nn.Linear(n_hidden_2, out_dim)

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)

        return x


class Activation_Net(nn.Module):
    '''
    只需在每层网络的输出部分添加激活函数即可，使用nn.Sequential()将网络层组合到一起
    注意：最后一层是输出层，不能添加激活函数
    '''

    def __init__(self, in_dim, n_hidden_1, n_hidden_2, out_dim):
        super(Activation_Net, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Linear(in_dim, n_hidden_1), nn.ReLU(True))
        self.layer2 = nn.Sequential(
            nn.Linear(n_hidden_1, n_hidden_2), nn.ReLU(True))
        self.layer3 = nn.Sequential(nn.Linear(n_hidden_2, out_dim))

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)

        return x


class Batch_Net(nn.Module):
    '''
    同样使用nn.Sequential()将nn.BatchNormld()组合到网络层
    注意：皮标准化一般放在全连接层的后面，非线性层（激活函数）的前面
    '''

    def __init__(self, in_dim, n_hidden_1, n_hidden_2, out_dim):
        super(Batch_Net, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Linear(in_dim, n_hidden_1), nn.BatchNorm1d(n_hidden_1), nn.ReLU(True))
        self.layer1 = nn.Sequential(
            nn.Linear(n_hidden_1, n_hidden_2), nn.BatchNorm1d(n_hidden_2), nn.ReLU(True))
        self.layer3 = nn.Sequential(nn.Linear(n_hidden_2, out_dim))

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)

        return x


class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()

        self.layer1 = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3), nn.BatchNorm2d(16), nn.ReLU(inplace=True))
        self.layer2 = nn.Sequential(nn.Conv2d(16, 32, kernel_size=3), nn.BatchNorm2d(
            32), nn.ReLU(inplace=True), nn.MaxPool2d(kernel_size=2, stride=2))
        self.layer3 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3), nn.BatchNorm2d(64), nn.ReLU(inplace=True))
        self.layer4 = nn.Sequential(nn.Conv2d(64, 128, kernel_size=3), nn.BatchNorm2d(
            128), nn.ReLU(inplace=True), nn.MaxPool2d(kernel_size=2, stride=2))
        self.fc = nn.Sequential(nn.Linear(128*4*4, 1024), nn.ReLU(inplace=True),
                                nn.Linear(1024, 128), nn.ReLU(inplace=True), nn.Linear(128, 10))

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x


data_tf = transforms.Compose(
    [transforms.ToTensor(), transforms.Normalize([0.5], [0.5])])

train_dataset = datasets.MNIST(
    root='./data', train=True, transform=data_tf, download=True)

test_dataset = datasets.MNIST(root='./data', train=False, transform=data_tf)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# model = simpleNet(28*28, 300, 100, 10)
model = CNN()
if torch.cuda.is_available():
    model = model.cuda()

# 损失函数
critertion = nn.CrossEntropyLoss()

# 优化
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

for epoch in range(num_epoches):
    print('Epoch {}/{} train ok'.format(epoch, num_epoches))
    i = 0
    loss_total = 0
    for data in train_loader:
        x_train, y_train = data

        outputs = model(x_train)

        optimizer.zero_grad()
        loss = critertion(outputs, y_train)

        loss.backward()
        optimizer.step()

        loss_total += loss.item()

        if i % 1000 == 0:
            print('Epoch {}/{},loss:{:.4f}'.format(epoch, num_epoches, loss.item()))

    # print('Epoch {}/{} train ok,loss:{}'.format(epoch, num_epoches,loss_total/len(train_loader)))
    print('train ok')
    test_correct = 0
    for data in test_loader:
        x_test, y_test = data

        outputs = model(x_test)

        _, pred = torch.max(outputs, 1)
        test_correct += torch.sum(pred == y_test.data)

    print('Test Accuracy is:{:.4f}'.format(test_correct/len(test_loader)))
