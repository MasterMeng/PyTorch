#!/usr/bin/env python3

import numpy as np

# # N是批量大小，D_in 是输入维度，H 是隐藏维度， D_out 是输出维度
# N, D_in, H, D_out = 64, 1000, 100, 10

# # 随机创建输入输出数据
# x = np.random.randn(N, D_in)
# y = np.random.randn(N, D_out)

# # 随机初始化权重
# w1 = np.random.randn(D_in, H)
# w2 = np.random.randn(H, D_out)

# learning_rate = 1e-6
# for i in range(20000):

#     # 前向传递：计算预测y
#     h = x.dot(w1)
#     h_relu = np.maximum(h, 0)
#     y_pred = h_relu.dot(w2)

#     # 计算并输出 loss
#     loss = np.square(y_pred-y).sum()
#     print(i, loss)

#     # 反向传递，计算相对于loss的w1和w2的梯度
#     grad_y_pred = 2.0*(y_pred-y)
#     grad_w2 = h_relu.T.dot(grad_y_pred)
#     grad_h_relu = grad_y_pred.dot(w2.T)
#     grad_h = grad_h_relu.copy()
#     grad_h[h < 0] = 0
#     grad_w1 = x.T.dot(grad_h)

#     # 更新权重
#     w1 -= learning_rate*grad_w1
#     w2 -= learning_rate*grad_w2


# import torch

# class MyReLU(torch.autograd.Function):

#     @staticmethod
#     def forward(ctx,input):
#         '''
#         在正向传递中，我们收到一个包含输入和返回的张量
#         包含输出的张量。ctx是一个可以使用的上下文对象
#         隐藏信息以备向后计算。在向后传递过程中你可以
#         使用ctx.save_for_backward方法任意缓存使用的对象。
#         '''
#         ctx.save_for_backward(input)
#         return input.clamp(min=0)

#     @staticmethod
#     def backward(ctx,grad_output):
#         '''
#         在后向传递中，我们得到一个关于输出的包含损失梯度的张量，我们需要计算关于输入的损失梯度。
#         '''
#         input, = ctx.saved_tensors
#         grad_input = grad_output.clone()
#         grad_input[input < 0] = 0
#         return grad_input

# dtype = torch.float
# device = torch.device('cpu')
# # device = torch.device('cuda:0') # 若在GPU上执行，取消注释

# # N是批量大小，D_in 是输入维度，H 是隐藏维度， D_out 是输出维度
# N, D_in, H, D_out = 64, 1000, 100, 10

# # 随机创建输入输出数据
# x = torch.randn(N, D_in, device=device, dtype=dtype)
# y = torch.randn(N, D_out, device=device, dtype=dtype)

# # 随机初始化权重
# w1 = torch.randn(D_in, H, device=device, dtype=dtype, requires_grad=True)
# w2 = torch.randn(H, D_out, device=device, dtype=dtype, requires_grad=True)

# learning_rate = 1e-6
# for i in range(500):
#     # 使用我们自定义的函数
#     relu = MyReLU.apply

#     # 前向传递：使用Tensor的操作来计算预测y，这些操作跟使用Tensor计算前向传递的操作很相似，
#     # 但是我们不需要保留中间变量的引用，因为我们不用手动执行后向传递。
#     y_pred = relu(x.mm(w1)).mm(w2)

#     # 计算并输出 loss
#     loss = (y_pred-y).pow(2).sum()
#     print(i, loss.item())

#     # 反向传递，计算相对于loss的w1和w2的梯度
#     loss.backward()

#     # 更新权重
#     with torch.no_grad():
#         w1 -= learning_rate * w1.grad
#         w2 -= learning_rate * w2.grad

#         w1.grad.zero_()
#         w2.grad.zero_()

# import tensorflow as tf
# import numpy as np

# # 首先定义计算图

# # N是批量大小，D_in 是输入维度，H 是隐藏维度， D_out 是输出维度
# N, D_in, H, D_out = 64, 1000, 100, 10

# # 为输入和目标数据创建占位符;它们在执行时会被真实数据填充
# x = tf.placeholder(tf.float32, shape=(None, D_in))
# y = tf.placeholder(tf.float32, shape=(None, D_out))

# # 随机初始化权重
# w1 = tf.Variable(tf.random_normal((D_in, H)))
# w2 = tf.Variable(tf.random_normal((H, D_out)))

# # 前向传递：通过TensorFlow张量上的操作计算预测的y值
# # 注意，这段代码并不执行任何数值操作，它仅仅是设置计算图以便我们之后执行
# h = tf.matmul(x, w1)
# h_relu = tf.maximum(h, tf.zeros(1))
# y_pred = tf.matmul(h_relu, w2)

# # 通过TensorFlow张量上的操作计算损失
# loss = tf.reduce_sum((y-y_pred)**2.0)

# # 计算loss关于w1和w2的梯度
# grad_w1, grad_w2 = tf.gradients(loss, [w1, w2])

# # 使用梯度更新权重。实际上在更新权重时，我们需要在执行图的过程中评估新的权重new_w1和new_w2。
# # 注意，在TensorFlow中更新权重值得操作是计算图的一部分，而在PyTorch中，这些操作发生在计算图之外。
# learning_rate = 1e-6
# new_w1 = w1.assign(w1 - learning_rate * grad_w1)
# new_w2 = w2.assign(w2 - learning_rate * grad_w2)

# # 现在，我们已经构建玩计算图，那么我们输入一个TensorFlow回话来实际执行图
# with tf.Session() as sess:
#     # 执行图之前需要先初始化变量w1和w2
#     sess.run(tf.global_variables_initializer())

#     # 创建numpy数组来存储输入x和目标y的真实数据
#     x_value = np.random.randn(N, D_in)
#     y_value = np.random.randn(N, D_out)

#     for _ in range(500):
#         # 重复执行多次。它每次执行时，我们使用参数`feed_dict`将x_value赋值给x，y_value赋值给y。
#         # 我们每次执行计算图时，我们需要计算loss、new_w1和new_w2的值，这些值得张量将以numpy数组的形式返回
#         loss_value, _, _ = sess.run([loss, new_w1, new_w2], feed_dict={
#                                     x: x_value, y: y_value})
#         print(loss_value)

# import torch

# # N是批量大小，D_in 是输入维度，H 是隐藏维度， D_out 是输出维度
# N, D_in, H, D_out = 64, 1000, 100, 10

# # 随机创建输入输出数据
# x = torch.randn(N, D_in)
# y = torch.randn(N, D_out)

# # 使用nn包将我们的模型定义为一系列的层。nn.Sequential是一个模型，它包含其它模型并按顺序使用它们来生成输出。
# # 每次模型使用线性函数从输入中计算输出，并保留权重和偏置的张量。
# model = torch.nn.Sequential(
#     torch.nn.Linear(D_in,H),
#     torch.nn.ReLU(),
#     torch.nn.Linear(H,D_out),
# )

# # nn包中也包含了常用的损失函数定义，这里我们使用Mean Squared Error(MSE，均方误差)作为我们的损失函数
# loss_fn = torch.nn.MSELoss(reduction='sum')

# # 学习率和优化算法
# learning_rate = 1e-4
# optimizer = torch.optim.Adam(model.parameters(),lr=learning_rate)

# for i in range(500):
#     # 前向传递：使用模型计算关于x的预测y
#     y_pred = model(x)

#     # 计算损失
#     loss = loss_fn(y_pred,y)
#     print(i,loss.item())

#     # 反向传递前零化梯度
#     optimizer.zero_grad()

#     # 反向传递
#     loss.backward()

#     # 调用优化算法的setp()函数来更新参数
#     optimizer.step()


# import torch

# class TwoLayerNet(torch.nn.Module):
#     def __init__(self,D_in,H,D_out):
#         super(TwoLayerNet,self).__init__()
#         self.linear1 = torch.nn.Linear(D_in,H)
#         self.linear2 = torch.nn.Linear(H,D_out)

#     def forward(self,x):
#         h_relu = self.linear1(x).clamp(min=0)
#         y_pred = self.linear2(h_relu)
#         return y_pred

# # N是批量大小，D_in 是输入维度，H 是隐藏维度， D_out 是输出维度
# N, D_in, H, D_out = 64, 1000, 100, 10

# # 随机创建输入输出数据
# x = torch.randn(N, D_in)
# y = torch.randn(N, D_out)

# # 实例化自定义模型
# model = TwoLayerNet(D_in,H,D_out)

# # 损失函数和优化函数
# criterion = torch.nn.MSELoss(reduction='sum')
# optimizer = torch.optim.SGD(model.parameters(),lr=1e-4)

# for i in range(500):
#     # 前向传递：使用模型计算关于x的预测y
#     y_pred = model(x)

#     # 计算损失
#     loss = criterion(y_pred,y)
#     print(i,loss.item())

#     # 反向传递前零化梯度
#     optimizer.zero_grad()

#     # 反向传递
#     loss.backward()

#     # 调用优化算法的setp()函数来更新参数
#     optimizer.step()

import random
import torch


class DynamicNet(torch.nn.Module):
    def __init__(self, D_in, H, D_out):
        super(DynamicNet, self).__init__()
        self.input_linear = torch.nn.Linear(D_in, H)
        self.middle_linear = torch.nn.Linear(H, H)
        self.output_linear = torch.nn.Linear(H, D_out)

    def forward(self, x):
        h_relu = self.input_linear(x).clamp(min=0)
        for _ in range(random.randint(0, 3)):
            h_relu = self.middle_linear(h_relu).clamp(min=0)
        y_pred = self.output_linear(h_relu)
        return y_pred


# N是批量大小，D_in 是输入维度，H 是隐藏维度， D_out 是输出维度
N, D_in, H, D_out = 64, 1000, 100, 10

# 随机创建输入输出数据
x = torch.randn(N, D_in)
y = torch.randn(N, D_out)

# 实例化自定义模型
model = DynamicNet(D_in, H, D_out)

# 损失函数和优化函数
criterion = torch.nn.MSELoss(reduction='sum')
optimizer = torch.optim.SGD(model.parameters(), lr=1e-4, momentum=0.9)

for i in range(500):
    # 前向传递：使用模型计算关于x的预测y
    y_pred = model(x)

    # 计算损失
    loss = criterion(y_pred, y)
    print(i, loss.item())

    # 反向传递前零化梯度
    optimizer.zero_grad()

    # 反向传递
    loss.backward()

    # 调用优化算法的setp()函数来更新参数
    optimizer.step()
