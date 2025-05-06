import torch.nn as nn
import torch.optim
import torch
from torch.autograd import Variable


# Linear Regression Model
class LinearRegression(nn.Module):
    def __init__(self):
        super(LinearRegression, self).__init__()
        self.linear = nn.Linear(2, 2)  # input and output is 2 dimension

    def forward(self, x):
        out = self.linear(x)
        return out


model = LinearRegression()

# /********** Begin *********/

# 声明一个 SGD 优化器 optimizer，传入参数
optimizer = torch.optim.SGD(model.parameters(), lr=0.01, weight_decay=0)
# 利用optimizer.param_groups查看优化器的各项参数并输出lr的值。
canshu = optimizer.param_groups
print(canshu[0]['lr'])
# /********** End *********/

