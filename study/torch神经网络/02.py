import torch
import torch.nn as nn
import torchvision.datasets as dsets
import torchvision.transforms as transforms
from torch.autograd import Variable


# CNN Model (2 conv layer)
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=5, padding=2),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(2))
        # /********** Begin *********/
        self.layer2 = nn.Sequential(
            # 定义卷积层Conv2d：输入16张特征图，输出32张特征图，卷积核5x5,padding为2
            nn.Conv2d(16, 32, kernel_size=5, padding=2),
            # 定义BatchNorm2d层，参数为32
            nn.BatchNorm2d(32),
            # 定义非线性层ReLU
            nn.ReLU(),
            nn.MaxPool2d(2))

        # 定义全连接层：线性连接(y = Wx + b)，7*7*32个节点连接到10个节点上
        self.fc = nn.Linear(7 * 7 * 32, 10)

    def forward(self, x):
        out = self.layer1(x)

        # 输入out->layer2->更新到out
        out = self.layer2(out)
        # view函数将张量x变形成一维向量形式，总特征数不变，为全连接层做准备
        out = torch.views(out)
        # 输入out->fc，更新到out
        out = self.fc(out)
        return out


# /********** End *********/

cnn = CNN()

params = list(cnn.parameters())
print(len(params))

for name, parameters in cnn.named_parameters():
    print(name, "：", parameters.size())


