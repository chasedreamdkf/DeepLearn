import torch.nn as nn
import torch.optim
import torch
from torch.autograd import Variable


# Linear Regression Model
class LinearRegression(nn.Module):
    def __init__(self):
        super(LinearRegression, self).__init__()
        self.linear = nn.Linear(2, 2)  # input and output is 1 dimension
        self.linear2 = nn.Linear(2, 2)

    def forward(self, x):
        out = self.linear(x)
        out = self.linear2(out)
        return out


model = LinearRegression()

# /********** Begin *********/


# 声明一个 SGD优化器 optimizer， 按要求设置 lr 的值
optimizer = torch.optim.SGD([
    {'params': model.linear.parameters()},
    {'params': model.linear2.parameters(), 'lr': 0.01}
], lr=1e-5, momentum=0.9)
# 按照格式输出optimizer.param_groups的长度
# print(optimizer.param_groups)
print('The len of param_groups list:', len(optimizer.param_groups))
# 按照格式输出linear层的lr
print("linear's lr:", optimizer.param_groups[0]['lr'])
# 按照格式输出linear2层的lr
print("linear2's lr:", optimizer.param_groups[1]['lr'])
# /********** End *********/
