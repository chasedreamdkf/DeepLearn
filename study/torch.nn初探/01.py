import torch
import torch.nn as nn
from torch.autograd import Variable


#/********** Begin *********/
#声明一个in_features=2,out_features=3的线性模型 l并输出
l = nn.Linear(2, 3)
print(l)
#变量 net 由三个l 序列构成，并输出 net
net = nn.Sequential(l, l, l)
print(net)

#/********** End *********/
