import torch
import torchvision.datasets as dsets
import torchvision.transforms as transforms

#/********** Begin *********/
# 下载MNIST数据集
mnist = dsets.MNIST(
    root='./data/mnist',
    train=True,
    transform= transforms.ToTensor(),
    download=True
)
print("Files already downloaded")
# 创建batch_size=100, shuffle=True的DataLoader类型的变量data_loader
data_loader = torch.utils.data.DataLoader(
    mnist,
    batch_size=100,
    shuffle=True
)
# 输出 data_loader中数据类型
# print(type(mnist))
# print(data_loader.dataset)
print(data_loader.dataset.__class__)
#/********** End *********/
