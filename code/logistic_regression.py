#!/usr/bin/env python3

import numpy
import torch
from torch import nn
from torch.autograd import Variable
from matplotlib import pyplot as plt

with open('data.txt','r') as f:
    data_list = f.readlines()
    data_list = [i.split('\n')[0] for i in data_list]
    data_list = [i.split(',') for i in data_list]
    data = [(float(i[0]),float(i[1]),float(i[2])) for i in data_list]
    # print(data)

# 使用matplotlib将数据画出来
x0 = list(filter(lambda x: x[-1] == 0.0,data))
x1 = list(filter(lambda x: x[-1] == 1.0,data))
# plot_x0_0 = [i[0] for i in x0]
# plot_x0_1 = [i[1] for i in x0]
# plot_x1_0 = [i[0] for i in x1]
# plot_x1_1 = [i[1] for i in x1]

# plt.plot(plot_x0_0,plot_x0_1,'ro',label='x_0')
# plt.plot(plot_x1_0,plot_x1_1,'bo',label='x_1')
# # plt.legend('best')
# plt.show()

class LogisticRegression(nn.Module):
    def __init__(self):
        super(LogisticRegression,self).__init__()
        self.lr = nn.Linear(2,1)
        self.sm = nn.Sigmoid()

    def forward(self,x):
        x = self.lr(x)
        x = self.sm(x)
        return x

if torch.cuda.is_available():
    model = LogisticRegression().cuda()
else:
    model = LogisticRegression()

criterion = nn.BCELoss()
optimizer = torch.optim.SGD(model.parameters(),lr=1e-3,momentum=0.9)


x_data = torch.FloatTensor([(i[0],i[1]) for i in data])
y_data = torch.FloatTensor([(i[2],) for i in data])
# print(x_data)
# print(y_data)

epochs = 10000
for epoch in range(epochs):
    inputs = Variable(x_data)
    target = Variable(y_data)

    out = model(inputs)

    loss = criterion(out,target)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if (epoch+1)%20 == 0:
        print('Epoch[{}/{}],loss:{:.6f}'.format(epoch+1,epochs,loss.item()))

print(model.state_dict())

model.eval()
predict = model(Variable(x_data))
predict = predict.data.numpy()

plt.plot(Variable(x_data).data.numpy(),predict,'ro')
plt.show()