#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov  5 13:59:42 2019

@author: james

1. computational graph

2. activation

3. quickly build a network

4. Data Loader

5. Optimizer

"""


# 1. computational graph
import torch
from torch.autograd import Variable

tensor = torch.FloatTensor([[1,2], [3,4]])
variable = Variable(tensor, requires_grad = True)

print(tensor)

print(variable)

t_out = torch.mean(tensor*tensor)
v_out = torch.mean(variable*variable)  # v_out = 1/4 * sum(variable*variable)

print(t_out)
print(v_out)


print(v_out.grad)
print(variable.grad)


v_out = torch.mean(variable*variable)
v_out.backward()

print(variable.grad)


print(variable)
print(variable.data)

print(variable.data.numpy())



# 2. activation
import torch
import torch.nn.functional as F
from torch.autograd import Variable

x = torch.linspace(-5, 5, 200) # x data (tensor), shape=(100, 1)
x = Variable(x)

x_np = x.data.numpy()   

# some activation methods
y_relu = F.relu(x).data.numpy()
y_sigmoid = F.sigmoid(x).data.numpy()
y_tanh = F.tanh(x).data.numpy()
y_softplus = F.softplus(x).data.numpy()
y_softmax = F.softmax(x)  #softmax 比较特殊, 不能直接显示, 不过他是关于概率的, 用于分类



import matplotlib.pyplot as plt

plt.figure(1, figsize=(8, 6))
plt.subplot(221)
plt.plot(x_np, y_relu, c='red', label='relu')
plt.ylim((-1, 5))
plt.legend(loc='best')

plt.subplot(222)
plt.plot(x_np, y_sigmoid, c='red', label='sigmoid')
plt.ylim((-0.2, 1.2))
plt.legend(loc = 'best')


plt.subplot(223)
plt.plot(x_np, y_tanh, c='red', label='tanh')
plt.ylim((-1.2, 1.2))
plt.legend(loc='best')

plt.subplot(224)
plt.plot(x_np, y_softplus, c='red', label='softplus')
plt.ylim((-0.2, 6))
plt.legend(loc='best')

plt.show()



# 3. quickly build a network
import torch
import torch.nn.functional as F

class Net(torch.nn.Module):
    def __init__(self, n_feature, n_hidden, n_output):
        super(Net, self).__init__()
        self.hidden = torch.nn.Linear(n_feature, n_hidden)
        self.predict = torch.nn.Linear(n_hidden, n_output)
        
    def forward(self, x):
        x = F.relu(self.hidden(x))
        x = self.predict(x)
        return x
    
net1 = Net(1, 10, 1)
net2 = torch.nn.Sequential(
        torch.nn.Linear(1, 10),
        torch.nn.ReLU(),
        torch.nn.Linear(10,1)
)

print(net1)
print(net2)



# 4. Data Loader
import torch
import torch.utils.data as Data
torch.manual_seed(1) 

BATCH_SIZE = 5

x = torch.linspace(1, 10, 10)
y = torch.linspace(10, 1, 10)

# torch dataset
torch_dataset = Data.TensorDataset (x , y)

# 把 dataset 放入 DataLoader
loader = Data.DataLoader(
    dataset=torch_dataset,      # torch TensorDataset format
    batch_size=BATCH_SIZE,      # mini batch size
    shuffle=True,               # 要不要打乱数据 (打乱比较好)
    num_workers=2,              # 多线程来读数据
)

for epoch in range(3):
    for step, (batch_x, batch_y) in enumerate(loader):
        ...
        print('Epoch:', epoch, '| Step: ', step, '| batch x: ',
              batch_x.numpy(), '| batch y: ', batch_y.numpy())



BATCH_SIZE = 8      # 批训练的数据个数
loader = Data.DataLoader(
    dataset=torch_dataset,      # torch TensorDataset format
    batch_size=BATCH_SIZE,      # mini batch size
    shuffle=True,               # 要不要打乱数据 (打乱比较好)
    num_workers=2,              # 多线程来读数据
)

for epoch in range(3):
    for step, (batch_x, batch_y) in enumerate(loader):
        ...
        print('Epoch:', epoch, '| Step: ', step, '| batch x: ',
              batch_x.numpy(), '| batch y: ', batch_y.numpy())



# 6. Optimizer
import torch 
import torch.utils.data as Data
import torch.nn.functional as F
from torch.autograd import Variable
import matplotlib.pyplot as plt

torch.manual_seed(1)

LR = 0.01
BATCH_SIZE = 32
EPOCH = 12


x = torch.unsqueeze(torch.linspace(-1, 1, 1000), dim = 1)
y = x.pow(2) + 0.1 * torch.normal(torch.zeros(x.size()))


# plot dataset
plt.scatter(x.numpy(), y.numpy())
plt.show()

torch_data = Data.TensorDataset(x, y)
loader = Data.DataLoader(dataset=torch_data, 
                         batch_size=BATCH_SIZE, 
                         shuffle = True, 
                         num_workers=2,
)


class Net(torch.nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.hidden = torch.nn.Linear(1, 20)
        self.predict = torch.nn.Linear(20, 1)
        
    def forward(self, x):
        x = F.relu(self.hidden(x))
        x = self.predict(x)
        return x
    

# 为每个优化器创建一个 net
net_SGD         = Net()
net_Momentum    = Net()
net_RMSprop     = Net()
net_Adam        = Net()
nets = [net_SGD, net_Momentum, net_RMSprop, net_Adam]





# different optimizers
opt_SGD         = torch.optim.SGD(net_SGD.parameters(), lr=LR)
opt_Momentum    = torch.optim.SGD(net_Momentum.parameters(), lr=LR, momentum=0.8)
opt_RMSprop     = torch.optim.RMSprop(net_RMSprop.parameters(), lr=LR, alpha=0.9)
opt_Adam        = torch.optim.Adam(net_Adam.parameters(), lr=LR, betas=(0.9, 0.99))
optimizers = [opt_SGD, opt_Momentum, opt_RMSprop, opt_Adam]
 
loss_func = torch.nn.MSELoss()
losses_his = [[], [], [], []]   # 记录 training 时不同神经网络的 loss



	
for epoch in range(EPOCH):
    print('Epoch: ', epoch)
    for step, (batch_x, batch_y) in enumerate(loader):
        #print(len(batch_x))
        b_x = Variable(batch_x)  # 务必要用 Variable 包一下
        b_y = Variable(batch_y)
        # 对每个优化器, 优化属于他的神经网络
        for net, opt, l_his in zip(nets, optimizers, losses_his):            
            output = net(b_x)              # get output for every net            
            loss = loss_func(output, b_y)  # compute loss for every net           
            opt.zero_grad()                # clear gradients for next train            
            loss.backward()                # backpropagation, compute gradients            
            opt.step()                     # apply gradients            
            l_his.append(loss.data)     # loss recoder
            
            
labels = ['SGD', 'Momentum', 'RMSprop', 'Adam']
for i, l_his in enumerate(losses_his):
    plt.plot(l_his, label=labels[i])
plt.legend(loc='best')
plt.xlabel('Steps')
plt.ylabel('Loss')
plt.ylim((0, 0.2))
plt.show()































