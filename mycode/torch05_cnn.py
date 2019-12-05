#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 27 16:45:09 2019

@author: james
"""
#############
# 5.1
import torch
from torch import nn

print(torch.__version__)


# 2d cnn
def corr2d(X, K):
    h, w = K.shape
    X, K = X.float(), K.float()
    Y = torch.zeros((X.shape[0] - h + 1, X.shape[0] - w + 1))
    for i in range(Y.shape[0]):
        for j in range(Y.shape[1]):
            Y[i, j] = (X[i: i + h, j: j + w] * K).sum()
    return Y
    
X = torch.tensor([[0, 1, 2], [3, 4, 5], [6, 7,8]])
K = torch.tensor([[0, 1], [2, 3]])
corr2d(X, K)


# 2d cnn layer
class Conv2D(nn.Module):
    def __init__(self, kernel_size):
        super(Conv2D, self).__init__()
        self.weight = nn.Parameter(torch.randn(kernel_size))
        self.bias = nn.Parameter(torch.randn(1))
        
    def forward(self, x):
        return corr2d(x, self.weight) + self.bias
    

X = torch.ones(6, 8)
X[:, 2:6] =  0
X

K = torch.tensor([[1, -1]])
print(K)
Y = corr2d(X, K)
Y


# build a (1, 2) 2d cnn layer
conv2d = Conv2D(kernel_size=(1,2))

step = 20
lr= 0.01
for i in range(step):
    Y_hat = conv2d(X)
    l = ((Y_hat - Y) ** 2).sum()
    l.backward()
    
    # GD
    conv2d.weight.data -= lr * conv2d.weight.grad
    conv2d.bias.data -= lr * conv2d.bias.grad
    # GD zero
    conv2d.weight.grad.fill_(0)
    conv2d.bias.grad.fill_(0)
    
    if (i + 1) % 5 == 0:
        print('Step %d, loss %.3f' % (i + 1, l.item()))
        
print("weight: ", conv2d.weight.data)
print("bias: ", conv2d.bias.data)

#############
# 5.2
import torch
from torch import nn
print(torch.__version__)

# padding
# 定义一个函数来计算卷积层。它对输入和输出做相应的升维和降维
def comp_conv2d(conv2d, X):
    # (1, 1)代表批量大小和通道数（“多输入通道和多输出通道”一节将介绍）均为1
    X = X.view((1, 1) + X.shape)
    Y = conv2d(X)
    return Y.view(Y.shape[2:])  # 排除不关心的前两维：批量和通道

# 注意这里是两侧分别填充1行或列，所以在两侧一共填充2行或列
conv2d = nn.Conv2d(in_channels=1, out_channels=1, kernel_size=3, padding=1)

X = torch.rand(8, 8)
comp_conv2d(conv2d, X).shape


# 使用高为5、宽为3的卷积核。在高和宽两侧的填充数分别为2和1
conv2d = nn.Conv2d(in_channels=1, out_channels=1, kernel_size=(5, 3), padding=(2, 1))
comp_conv2d(conv2d, X).shape

# stride
conv2d = nn.Conv2d(1, 1, kernel_size=3, padding=1, stride=2)
comp_conv2d(conv2d, X).shape

conv2d = nn.Conv2d(1, 1, kernel_size=(3, 5), padding=(0, 1), stride=(3, 4))
comp_conv2d(conv2d, X).shape


#############
# 5.3.1
import torch
from torch import nn
import sys
sys.path.append('..')
import d2lzh_pytorch as d2l

def corr2d_multi_in(X, K):
    res = d2l.corr2d(X[0, :, :], K[0, :, :])
    for i in range(1, X.shape[0]):
        res += d2l.corr2d(X[i, :, :], K[i, :, :])
    return res

X = torch.tensor([[[0,1,2],[3,4,5],[6,7,8]],
                  [[1,2,3],[4,5,6],[7,8,9]]])
K = torch.tensor([[[0,1],[2,3]],[[1,2],[3,4]]])

corr2d_multi_in(X, K)

# 5.3.2
def corr2d_multi_in_out(X, K):
    return torch.stack([corr2d_multi_in(X,k) for k in K])

K = torch.stack([K, K + 1, K + 2])
K.shape # torch.Size([3,2,2,2])

corr2d_multi_in_out(X, K)

# 5.3.3
def corr2d_multi_in_out_1x1(X, K):
    c_i, h, w = X.shape
    c_o = K.shape[0]
    X = X.view(c_i, h * w)
    K = K.view(c_o, c_i)
    Y = torch.mm(K, X)
    return Y.view(c_o, h, w)

X = torch.rand(3, 3, 3)
K = torch.rand(2, 3, 1, 1)

Y1 = corr2d_multi_in_out_1x1(X, K)
Y2 = corr2d_multi_in_out(X, K)

(Y1 - Y2).norm().item() < 1e-6

#############
# 5.4.1
import torch
from torch import nn

def pool2d(X, pool_size, mode='max'):
    X = X.float()
    p_h, p_w = pool_size
    Y = torch.zeros(X.shape[0] - p_h + 1, X.shape[1] - p_w + 1)
    for i in range(Y.shape[0]):
        for j in range(Y.shape[1]):
            if mode == 'max':
                Y[i, j] = X[i: i + p_h, j: j + p_w].max()
            elif mode == 'avg':
                Y[i, j] = X[i: i + p_h, j: j + p_w].mean()       
    return Y

X = torch.tensor([[0, 1, 2], [3, 4, 5], [6, 7, 8]])
pool2d(X, (2, 2))

pool2d(X, (2, 2), 'avg')

# 5.4.2
X = torch.arange(16, dtype=torch.float).view((1, 1, 4, 4))
X

pool2d = nn.MaxPool2d(3)
pool2d(X) 

pool2d = nn.MaxPool2d(3, padding=1, stride=2)
pool2d(X)

pool2d = nn.MaxPool2d((2, 4), padding=(1, 2), stride=(2, 3))
pool2d(X)

# 5.4.3
X = torch.cat((X, X + 1), dim=1)
X

pool2d = nn.MaxPool2d(3, padding=1, stride=2)
pool2d(X)

































