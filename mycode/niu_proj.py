# -*- coding: utf-8 -*-
"""
Created on Wed Nov  6 21:09:47 2019

@author: bejin
"""

import torch
import torchvision
import torch.utils.data as Data
import os
from numpy import genfromtxt
import matplotlib.pyplot as plt
import numpy as np


BATCH_SIZE = 1


## - functions
def f_plot_data(txt_path):
    #txt_data01 = 'Walabot_Tue Aug 28 14_00_30 2018_.txt'    
    my_data = genfromtxt(txt_path, delimiter=' ')
        
    x = my_data[:, 0]
    y = my_data[:, 1]           
    
    plt.figure(1)
    plt.plot(x, y)
    plt.show()


class Loadfile():
    def __init__(self, data_path):
        self.data_path = data_path
        self.entries = os.listdir(data_path)
        self.data = [[], []]
        self.extract_data()
        
    def extract_data(self):
        if self.entries == None:
            return None
            
        for e in self.entries:
            my_data = genfromtxt(self.data_path + e, delimiter=' ')
            x = my_data[:, 0]
            y = my_data[:, 1]
            self.data[0].append(x)
            self.data[1].append(y)
            
        data_x = np.array(self.data[0])
        data_y = np.array(self.data[1])
        data_x_mean = np.mean(data_x, axis = 0)
        data_y_mean = np.mean(data_y, axis = 0)
        self.data = np.array([data_x_mean, data_y_mean])
###

data_path = 'E:/data/Field_28_08/walabot_20_4/'
entries = os.listdir(data_path)


#[f_plot_data(data_path + d) for d in entries]


loadfile = Loadfile(data_path)
(data_x, data_y) = loadfile.data

plt.figure(1, figsize = (8,6))
plt.plot(data_x, data_y)
plt.show()

















