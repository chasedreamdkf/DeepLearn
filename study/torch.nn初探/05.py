import torch
import torch.nn as nn
from torch.autograd import Variable


x = Variable(torch.randn(10, 3, 28, 28))

#/********** Begin *********/

#创建一个in_channels=3, out_channels=32, kernel_size=(3, 3), stride=1, padding=1, bias=True的Conv2d变量conv
conv = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=(3, 3), stride=1, padding=1, bias=True)

#创建一个kernel_size=(2, 2), stride=2的MaxPool2d变量pool
pool = nn.MaxPool2d(kernel_size=(2, 2), stride=2)

#对x应用卷积和最大池化操作并赋值给变量outpout_pool
outpout_pool = pool(conv(x))

#输出 outpout_pool 的大小,要求输出打印不换行
print(outpout_pool.size())
#/********** End *********/
