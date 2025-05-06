import torch
import torch.nn as nn
from torch.autograd import Variable

input = Variable(torch.randn(10, 16, 40))

#/********** Begin *********/

#创建一个in_channels=16, out_channels=24, kernel_size=4, stride=3的Conv1d变量conv
conv = nn.Conv1d(16, 24, 4, stride=3)

#对input应用卷积操作并赋值给变量 output
output = conv(input)

#输出 output 的大小，要求输出不换行
print(output.size(), end='')

#/********** End *********/
