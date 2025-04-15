import torch
import torch.nn as nn
import torchvision.datasets as dsets
import torchvision.transforms as transforms
from torch.autograd import Variable
import warnings
warnings.filterwarnings('ignore')

import os,sys
path = os.path.split(os.path.abspath(os.path.realpath(sys.argv[0])))[0] + os.path.sep
rootpath = path[:-10]

#print("validation path:" ,root)


# MNIST Dataset
test_dataset = dsets.MNIST(
    root='./data/',
    train=False,
    transform=transforms.ToTensor(),
    download=False
)

test_loader = torch.utils.data.DataLoader(
    dataset=test_dataset,
    batch_size=100,
    shuffle=True
)

# CNN Model (2 conv layer)
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=5, padding=2),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        self.layer2 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=5, padding=2),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        self.fc = nn.Linear(7*7*32, 10)

def forward(self, x):
    out = self.layer1(x)
    out = self.layer2(out)
    out = out.view(out.size(0), -1)
    out = self.fc(out)
    return out

cnnmodel = CNN()
# cnnmodel = torch.load( rootpath + 'src/step3/cnnModel.pkl')

#/********** Begin *********/
# 将模型转为测试模式
cnnmodel.eval()
correct = 0
total = 0
i  = 0
for images, labels in test_loader:
    images = Variable(images)
    #对images 应用cnn模型，将结果赋值给 outputs
    outputs = cnnmodel(images)
    _, predicted = torch.max(outputs.data, 1)
    total += labels.size(0)
    correct += (predicted == labels).sum()
    i += 1
    # 为了节约时间, 我们测试时只测试前10个
    if  i> 10 :
        break

#按格式输出正确率correct/total 的百分比
print("Test Accuracy of the model on the 200 test images: %d %%"%(100*correct/total))
#/********** End *********