import numpy as np
import torch
from torch.utils import data
from d2l import torch as d2l
from torch import nn

# 生成数据集
true_w = torch.tensor([2, -3.4])
true_b = 4.2
features, labels = d2l.synthetic_data(true_w, true_b, 1000)

# 读取数据集（小批量）
batch_size = 10
dataset = data.TensorDataset(features, labels)
data_iter = data.DataLoader(dataset, batch_size, shuffle=True)

# 定义模型
model = nn.Sequential(nn.Linear(2, 1))

# 定义损失
loss = nn.MSELoss()

# 定义优化算法
trainer = torch.optim.SGD(model.parameters(), lr=0.03)

# 初始化模型参数
model[0].weight.data.normal_(0, 0.01)
model[0].bias.data.fill_(0)

# 训练
num_epochs = 3
for epoch in range(num_epochs):
    for X, y in data_iter:
        output = model(X)
        l = loss(output, y)
        trainer.zero_grad()
        l.backward()
        trainer.step()
    l = loss(model(features), labels)
    print(f'epoch {epoch + 1}, loss {l:f}')

# 查看 w 和 b 的误差
w = model[0].weight.data
print('w 的估计误差：', true_w - w.reshape(true_w.shape))
b = model[0].bias.data
print('w 的估计误差：', true_b - b)