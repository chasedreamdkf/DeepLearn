import torch
import torch.nn as nn
from torch.autograd import Variable

input = Variable(torch.Tensor([[1,2,3,4],[5,6,7,8]]))

#/********** Begin *********/
# 创建一个4维的 带有学习参数的正则化量 m
m = nn.BatchNorm1d(4)

#输出weight和bias
print(m.weight)
print(m.bias)

#在 input 上应用该正则化并输出
print(m(input))

#/********** End *********/
