import torch
import torch.nn as nn
from torch.autograd import Variable

input = Variable(torch.Tensor([2.3,-1.4,0.54]))

#/********** Begin *********/
#创建 Tanh 模型 m
m = nn.Tanh()
#输出经 m 变化后的 input 值
print(m(input))
#/********** End *********/
