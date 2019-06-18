#!/usr/bin/env python3

import numpy
import torch
from torch import nn
from torch.autograd import Variable
from matplotlib import pyplot as plt


class LinerRegression(nn.Module):
    def __init__(self):
        super(LinerRegression, self).__init__()
        self.linear = nn.Linear(1, 1)

    def forward(self, x):
        out = self.linear(x)
        return out


def one():
    x_train = numpy.array([[3.3], [4.4], [5.5], [6.71], [6.39], [4.168], [9.779], [6.182], [
                          7.59], [2.167], [7.042], [10.791], [5.313], [7.997], [3.1]], dtype=numpy.float32)
    y_train = numpy.array([[1.7], [2.76], [2.09], [3.19], [1.694], [1.573], [3.366], [2.596], [
                          2.53], [1.221], [2.827], [3.465], [1.65], [2.904], [1.3]], dtype=numpy.float32)

    x_train = torch.from_numpy(x_train)
    y_train = torch.from_numpy(y_train)

    # print(x_train)
    # print(y_train)

    if torch.cuda.is_available():
        model = LinerRegression().cuda()
    else:
        model = LinerRegression()

    criterion = nn.MSELoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)

    num_epochs = 1000
    for epoch in range(num_epochs):
        if torch.cuda.is_available():
            inputs = Variable(x_train).cuda()
            target = Variable(y_train).cuda()
        else:
            inputs = Variable(x_train)
            target = Variable(y_train)

        out = model(inputs)
        loss = criterion(out, target)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (epoch+1) % 20 == 0:
            print('Epoch[{}/{}],loss:{:.6f}'.format(epoch +
                                                    1, num_epochs, loss.item()))

    model.eval()
    predict = model(Variable(x_train))
    predict = predict.data.numpy()

    plt.plot(x_train.numpy(), y_train.numpy(), 'ro', label='Original data')
    plt.plot(x_train.numpy(), predict, label='Fitting Lint')
    plt.show()


# polynomial regression 多项式回归
def make_features(x):
    '''
    build features i.e a matrix with columns [x,x^2,x^3]
    '''
    # unsqueeze(arg):表示在第arg维度增加一个维度为1的维度
    # squeeze(arg):表示第arg维的维度值为1，则去掉该维度，否则不变
    x = x.unsqueeze(1)
    return torch.cat([x ** i for i in range(1, 4)], 1)


W_target = torch.FloatTensor([0.5, 3, 2.4]).unsqueeze(1)
b_target = torch.FloatTensor([0.9])


def f(x):
    '''
    approximated function
    '''
    return x.mm(W_target)+b_target[0]


def get_batch(batch_size=32):
    '''
    build a batch ie. (x,f(x)) pair
    '''
    random = torch.randn(batch_size)
    x = make_features(random)
    y = f(x)
    if torch.cuda.is_available():
        return Variable(x).cuda(), Variable(y).cuda()
    else:
        return Variable(x), Variable(y)


class poly_model(nn.Module):
    def __init__(self):
        super(poly_model, self).__init__()
        self.poly = nn.Linear(3, 1)

    def forward(self, x):
        out = self.poly(x)
        return out


def multi():

    if torch.cuda.is_available():
        model = poly_model().cuda()
    else:
        model = poly_model()

    criterion = nn.MSELoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)

    epoch = 0

    while True:
        batch_x, batch_y = get_batch()
        output = model(batch_x)
        loss = criterion(output, batch_y)
        print_loss = loss.item()

        optimizer.zero_grad()

        loss.backward()

        optimizer.step()
        epoch += 1

        # print('loss:{} after {} batches'.format(print_loss,epoch))

        if print_loss < 1e-3:
            break

    print('Actual function: y = {} + {}*x + {}*x^2 + {}*x^3'.format(0.9, 0.5, 3, 2.4))
    # print('Learned function: y = {} + {}*x + {}*x^2 + {}*x^3'.format(0.9,0.5, 3, 2.4))
    W_learned = model.state_dict()['poly.weight']
    b_learned = model.state_dict()['poly.bias']
    print(W_learned, b_learned)


if __name__ == "__main__":
    # one()
    multi()
