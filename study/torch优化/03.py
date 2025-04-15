import torch
from torch import nn, optim
from torch.autograd import Variable
import numpy as np


# Linear Regression Model
class LinearRegression(nn.Module):
    def __init__(self):
        super(LinearRegression, self).__init__()
        self.linear = nn.Linear(1, 1)  # input and output is 1 dimension

    def forward(self, x):
        out = self.linear(x)
        return out


x_train = Variable(torch.randn(1, 1))
y_train = Variable(torch.randn(1, 1))
criterion = nn.MSELoss()

model = LinearRegression()

# /********** Begin *********/

# 声明一个 RMSprop 优化器 optimizer， 按要求设置 lr,alpha 的值
optimizer = optim.RMSprop(model.parameters(), lr=0.1, alpha=0.9)
# 清空梯度
optimizer.zero_grad()
# 计算Loss
loss = criterion(model(x_train), y_train)
# 反向传播
loss.backward()
# 更新参数
optimizer.step()
# 按照格式输出optimizer的lr
# print(optimizer.param_groups)
print("optimizer's lr:" ,optimizer.param_groups[0]['lr'])
##按照格式输出optimizer的alpha
print("optimizer's alpha:" ,optimizer.param_groups[0]['alpha'])
# /********** End *********/
