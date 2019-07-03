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


import torch

dtype = torch.float
device = torch.device('cpu')
# device = torch.device('cuda:0') # 若在GPU上执行，取消注释

# N是批量大小，D_in 是输入维度，H 是隐藏维度， D_out 是输出维度
N, D_in, H, D_out = 64, 1000, 100, 10

# 随机创建输入输出数据
x = torch.randn(N, D_in, device=device, dtype=dtype)
y = torch.randn(N, D_out, device=device, dtype=dtype)

# 随机初始化权重
w1 = torch.randn(D_in, H, device=device, dtype=dtype)
w2 = torch.randn(H, D_out, device=device, dtype=dtype)

learning_rate = 1e-6
for i in range(500):
    # 前向传递：计算预测y
    h = x.mm(w1)
    h_relu = h.clamp(min=0)
    y_pred = h_relu.mm(w2)

    # 计算并输出 loss
    loss = (y_pred-y).pow(2).sum().item()
    print(i, loss)

    # 反向传递，计算相对于loss的w1和w2的梯度
    grad_y_pred = 2.0*(y_pred-y)
    grad_w2 = h_relu.t().mm(grad_y_pred)
    grad_h_relu = grad_y_pred.mm(w2.t())
    grad_h = grad_h_relu.clone()
    grad_h[h < 0] = 0
    grad_w1 = x.t().mm(grad_h)

    # 更新权重
    w1 -= learning_rate*grad_w1
    w2 -= learning_rate*grad_w2
