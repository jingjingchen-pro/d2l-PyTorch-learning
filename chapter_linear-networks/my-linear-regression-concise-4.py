import numpy as np
import torch
from torch.utils import data
from d2l import torch as d2l
from torch import nn

# 生成数据集
true_w = torch.tensor([2, -3.4])
true_b = 4.2
features, labels = d2l.synthetic_data(true_w, true_b, 1000)

# 读取数据集
def load_array(data_arrays, batch_size, is_train=True):
    """构造一个PyTorch数据迭代器"""
    dataset = data.TensorDataset(*data_arrays)
    return data.DataLoader(dataset, batch_size, shuffle=is_train)

batch_size = 10
data_iter = load_array((features, labels), batch_size)

# define model
net = nn.Sequential(nn.Linear(2, 1))  # Squential类将多个层串联在一起；全连接层在Linear类中定义，两个参数分别是输入特征形状和输出特征形状

# 初始化模型参数
net[0].weight.data.normal_(0, 0.01)  # 访问网络第一层的权重参数，.data直接访问权重参数的底层数据部分（Tensor数据），.normal_()对权重进行初始化
net[0].bias.data.fill_(0)

# 定义损失函数
loss = nn.MSELoss()

# 定义优化算法
trainer = torch.optim.SGD(net.parameters(), lr=0.03)  # net.parameters()是一种方法，返回的是网络中所有可学习参数的迭代器。意味着每次调用时，会动态生成当前网络的参数列表
                                                      # 当我们实例化一个SGD实例时，我们要指优化的参数以及优化算法所需要的超参数字典 

# 训练
num_epochs = 3
for epoch in range(num_epochs):
    for X, y in data_iter:
        l = loss(net(X), y)
        trainer.zero_grad()
        l.backward()
        trainer.step()
    l = loss(net(features), labels)
    print(f'epoch {epoch + 1}, loss {l:f}')

w = net[0].weight.data
print('w的估计误差：', true_w - w.reshape(true_w.shape))
b = net[0].bias.data
print('b的估计误差', true_b - b)