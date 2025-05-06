import torch
from torch import nn
class LeNet(nn.Module):
    def __init__(self):
        super(LeNet, self).__init__()
        '''
        这里搭建卷积层，需要按顺序定义卷积层、
        激活函数、最大池化层、卷积层、激活函数、最大池化层，
        具体形状见测试说明
        '''

        self.conv = nn.Sequential(
            ########## Begin ##########
            nn.Conv2d(1, 6, 5),  # 卷积层
            nn.Sigmoid(),  # 激活函数
            nn.MaxPool2d(2, 2, padding=0, dilation=1, ceil_mode=1),  # 最大池化层
            nn.Conv2d(6, 16, 5),
            nn.Sigmoid(),
            nn.MaxPool2d(2, 2, padding=0, dilation=1, ceil_mode=1),
            ########## End ##########
        )

        '''
        这里搭建全连接层，需要按顺序定义全连接层、
        激活函数、全连接层、激活函数、全连接层，
        具体形状见测试说明
        '''
        self.fc = nn.Sequential(
            ########## Begin ##########
            nn.Linear(14 * 14, 120),  # 全连接层
            nn.Sigmoid(),  # 激活函数
            nn.Linear(120, 84),
            nn.Sigmoid(),
            nn.Linear(84, 10)

            ########## End ##########
        )

    def forward(self, img):
        """
        这里需要定义前向计算
        """
        ########## Begin ##########
        out = self.conv(img)
        out = self.fc(out.view(img.shape[0], -1))
        return out
        ########## End ##########

l = LeNet()
print(l)