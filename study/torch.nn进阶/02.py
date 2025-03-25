import torch
import torch.nn as nn
from torch.autograd import Variable

input = Variable(torch.Tensor([1.1,2.2,2.93,3.8]))
target = Variable(torch.Tensor([1,2,3,4]))

#/********** Begin *********/

#创建名为 loss 的 L1Loss损失函数
loss = nn.L1Loss()
#对 input 和 target应用 loss 并输出
output = loss(input, target)
print(output)

#/********** End *********/