import torch
import torch.nn as nn
from torch.autograd import Variable

input = Variable(torch.Tensor([[1.1,2.2],[2.93,3.8]]))
target = Variable(torch.Tensor([[1,2],[3,4]]))

#/********** Begin *********/
#创建一范数的变量pdist
pdlist = nn.PairwiseDistance(p=1)
#对 input 、 target应用该范数并输出
output = pdlist(input, target)
print(output)
#/********** End *********/
