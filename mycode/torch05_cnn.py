#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 27 16:45:09 2019

@author: james
"""

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
class Conv2D(nn.Moudle):
    def __init__(self, kernel_size):
        super(Conv2D, self).__init__()
        self.weight = nn.Parameter(torch.randn(kernel_size))
        self.bais = nn.Parameter(torch.randn(1))
        
    def forward(self, x):
        return corr2d(x, self.weight) + self.bias
    
