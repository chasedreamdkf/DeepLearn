import torch
import torch.nn as nn
from torch.autograd import Variable

#/********** Begin *********/
# 创建in_features=3, out_features=2线性层变量 linear
linear = nn.Linear(3, 2)
#输出linear
print(linear)
#输出linear的 type 属性
print(linear.type)
#/********** End *********/
