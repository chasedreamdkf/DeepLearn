import torch
from torch import nn


class AlexNet(nn.Module):
    def __init__(self):
        super(AlexNet, self).__init__()

        '''
        这里搭建卷积层，需要按顺序定义卷积层、
        激活函数、最大池化层、卷积层、激活函数、
        最大池化层、卷积层、激活函数、卷积层、
        激活函数、卷积层、激活函数、最大池化层，
        具体形状见测试说明
        '''
        self.conv = nn.Sequential(
            ########## Begin ##########
            nn.Conv2d(1, 96, kernel_size=11, stride=4),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=0, dilation=1, ceil_mode=False),
            nn.Conv2d(96, 256, 5, 1, 2),
            nn.ReLU(),
            nn.MaxPool2d(3, 2, 0, 1, False),
            nn.Conv2d(256, 384, 3, 1, 1),
            nn.ReLU(),
            nn.Conv2d(384, 384, 3, 1, 1),
            nn.ReLU(),
            nn.Conv2d(384, 256, 3, 1, 1),
            nn.ReLU(),
            nn.MaxPool2d(3, 2, 0, 1, False),
            ########## End ##########
        )


        '''
        这里搭建全连接层，需要按顺序定义
        全连接层、激活函数、丢弃法、
        全连接层、激活函数、丢弃法、全连接层，
        具体形状见测试说明
        '''
        self.fc = nn.Sequential(
            ########## Begin ##########
            nn.Linear(6400, 4096, True),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(4096, 4096),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(4096, 10)
            ########## End ##########
        )


    def forward(self, img):
        """
        这里需要定义前向计算
        """
        ########## Begin ##########
        feature = self.conv(img)
        out = self.fc(feature.view(img.shape[0], -1))
        return out
        ########## End ##########