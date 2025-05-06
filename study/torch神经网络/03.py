import torch
import torch.nn as nn
import torchvision.datasets as dsets
import torchvision.transforms as transforms
from torch.autograd import Variable


# Hyper Parameters
batch_size = 100
learning_rate = 0.001
num_epochs = 1
# MNIST Dataset
train_dataset = dsets.MNIST(root='./data/',
                            train=True,
                            transform=transforms.ToTensor(),
                            download=True)
train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                           batch_size=batch_size,
                                           shuffle=True)

# CNN Model (2 conv layer)
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.layer1 = nn.Sequential(
                                    nn.Conv2d(1, 16, kernel_size=5, padding=2),
                                    nn.BatchNorm2d(16),
                                    nn.ReLU(),
                                    nn.MaxPool2d(2))
        self.layer2 = nn.Sequential(
                                    nn.Conv2d(16, 32, kernel_size=5, padding=2),
                                    nn.BatchNorm2d(32),
                                    nn.ReLU(),
                                    nn.MaxPool2d(2))
        self.fc = nn.Linear(7*7*32, 10)
    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        return out
cnnmodel = CNN()
#创建输出文件 output.txt
f = open('./output.txt', 'w')
f.seek(0)
f.truncate()   #清空文件
#/********** Begin *********/
# 声明一个为交叉熵损失函数的变量criterion
criterion = nn.CrossEntropyLoss()
# 声明一个为Adam优化函数的变量optimizer，传入 cnnmodel的参数，并使学习率lr为0.001
optimizer = torch.optim.Adam(cnnmodel.parameters(), lr=learning_rate)
# 训练模型
for i, (images, labels) in enumerate(train_loader):
    # 将images，labels数据转换为Variable类型
    images = Variable(images)
    labels = Variable(labels)
    # optimizer梯度归零
    optimizer.zero_grad()
    # 对 images 应用 cnnmodel 模型并赋值给变量 outputs
    outputs = cnnmodel(images)
    # 计算损失
    loss = criterion(outputs, labels)
    # Backward
    loss.backward()
    # Optimize
    optimizer.step()
    # 共训练60次，分别100次输出一回loss信息，并将输出信息存到文件中
    if (i+1) % 10 == 0:
        f.writelines('Epoch [%d/%d], Iter [%d/%d] Loss: %.4f \n'
                     % (1, num_epochs, i+1, len(train_dataset) // 1000, loss.item()))
        print('Epoch [%d/%d], Iter [%d/%d] Loss: %.4f'
              % (1, num_epochs, i+1, len(train_dataset) // 1000, loss.item()))
    if i > 60:
        break
f.close()
#/********** End *********