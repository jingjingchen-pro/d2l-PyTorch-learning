# 241117
import torch
import numpy as np
from torch.utils import data
from d2l import torch as d2l
from torch import nn


# 构建数据迭代器读取数据
def load_array(data_arrays, batch_size, is_train=True):
    dataset = data.TensorDataset(*(data_arrays))  # torch.utils.data.TensorDataset()用于将输入数据和相应的标签封装成一个数据集，为了之后方便批量处理
    return data.DataLoader(dataset, batch_size, shuffle=is_train)  # train的时候要打乱，test的时候不要打乱

# Define model
net = nn.Sequential(nn.Linear(2, 1))

# Define loss function
loss = nn.MSELoss()

# Define optimization algorithm
trainer = torch.optim.SGD(net.parameters(), lr=0.03)

"""Training

"""
# 创建真实数据集
true_w = torch.tensor([2, -3.4])
true_b = 4.2
features, labels = d2l.synthetic_data(true_w, true_b, 1000)

# 定义超参数
num_epochs = 3
batch_size = 10

# data_iter
data_iter = load_array((features, labels), batch_size)

# Initialize params
net[0].weight.data.normal_(0, 0.01)
net[0].bias.data.fill_(0)

for epoch in range(num_epochs):
    for X, y in data_iter:  # mini-batch 中的每个样本
        l = loss(net(X), y)
        trainer.zero_grad()
        l.backward()
        trainer.step()
    l = loss(net(features), labels)  # 整个样本集，默认mean损失
    print(f'epoch {epoch + 1}, loss {l:f}')

w = net[0].weight.data
print('w的估计误差：', true_w - w.reshape(true_w.shape))
b = net[0].bias.data
print('b的估计误差：', true_b - b)