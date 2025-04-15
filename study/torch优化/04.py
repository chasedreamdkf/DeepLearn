import torch
from torch import nn, optim
from torch.autograd import Variable
import numpy as np


# Linear Regression Model
class LinearRegression(nn.Module):
    def __init__(self):
        super(LinearRegression, self).__init__()
        self.linear = nn.Linear(2, 2)  # input and output is 2 dimension

    def forward(self, x):
        out = self.linear(x)
        return out


x_train = Variable(torch.from_numpy(np.array([[1, 2], [3, 4]], dtype=np.float32)))
y_train = Variable(torch.from_numpy(np.array([[1, 5], [2, 8]], dtype=np.float32)))

model_Adam1 = LinearRegression()
model_Adam2 = LinearRegression()
models = [model_Adam1, model_Adam2]
# /********** Begin *********/

# 声明一个Adam优化器 optimizer1， 设置 lr为0.2，betas为(0.9,0.9)
opt_Adam1 = optim.Adam(model_Adam1.parameters(), lr=0.2, betas=(0.9, 0.9))
# 声明一个Adam优化器 optimizer2， 设置 lr为0.001，betas为(0.9,0.9)
opt_Adam2 = optim.Adam(model_Adam2.parameters(), lr=0.001, betas=(0.9, 0.9))

optimizers = [opt_Adam1, opt_Adam2]
losses_his = [[], []]
loss_func = nn.MSELoss()

for epoch in range(10):
    # 对每个优化器, 优化属于他的神经网络
    for model, opt, l_his in zip(models, optimizers, losses_his):
        output = model(x_train)
        loss = loss_func(output, y_train)
        opt.zero_grad()
        loss.backward()
        opt.step()
        l_his.append(loss.item())
loss1 = sum(losses_his[0])
loss2 = sum(losses_his[1])

# 利用 if-else 结构判断 loss1、loss2的大小
# 若loss1小于loss2，输出“opt_Adam1 is better than opt_Adam2”；
# 否则输出“opt_Adam2 is better than opt_Adam1”。
if loss1 < loss2:
    print("opt_Adam1 is better than opt_Adam2")
elif loss1 > loss2:
    print("opt_Adam2 is better than opt_Adam1")
# /********** End *********/

