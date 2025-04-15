import torch
import torch.utils.data as Data
import torch.nn.functional as F
from torch.autograd import Variable

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt

import warnings

warnings.filterwarnings('ignore')

import os, sys

path = os.path.split(os.path.abspath(os.path.realpath(sys.argv[0])))[0] + os.path.sep
print("validation path:", path)

# 定义参数
LR = 0.01
BATCH_SIZE = 32
EPOCH = 10

x = torch.unsqueeze(torch.linspace(-1, 1, 100), dim=1)
y = x.pow(2) + 0.1 * torch.randn(x.size())
# torch_dataset = Data.TensorDataset(data_tensor=x, target_tensor=y)
torch_dataset = Data.TensorDataset(x, y)
loader = Data.DataLoader(dataset=torch_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2, )


# 默认的 network 形式
class Net(torch.nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.hidden = torch.nn.Linear(1, 40)
        self.predict = torch.nn.Linear(40, 1)

    def forward(self, x):
        # 隐藏层的激活函数
        x = F.relu(self.hidden(x))
        # 线性输出
        x = self.predict(x)
        return x


net_SGD = Net()
net_Momentum = Net()
net_RMSprop = Net()
net_Adam = Net()
nets = [net_SGD, net_Momentum, net_RMSprop, net_Adam]

# /********** Begin *********/

opt_SGD = torch.optim.SGD(net_SGD.parameters(), lr=LR)

# 声明优化器opt_Momentum，传入对应的模型参数，lr 赋值为 LR，momentum为0.7
opt_Momentum = torch.optim.SGD(net_Momentum.parameters(), lr=LR, momentum=0.7)
# 声明优化器opt_RMSprop，传入对应的模型参数，lr 赋值为 LR，alpha为0.9
opt_RMSprop = torch.optim.RMSprop(net_RMSprop.parameters(), lr=LR, alpha=0.9)
# 声明优化器opt_Adam，传入对应的模型参数，lr 赋值为 LR，betas为(0.9, 0.99)
opt_Adam = torch.optim.Adam(net_Adam.parameters(), lr=LR, betas=(0.9, 0.99))

# /********** End *********/
optimizers = [opt_SGD, opt_Momentum, opt_RMSprop, opt_Adam]
loss_func = torch.nn.MSELoss()

losses_his = [[], [], [], []]

# 训练循环
for epoch in range(EPOCH):
    for step, (batch_x, batch_y) in enumerate(loader):
        b_x = Variable(batch_x)
        b_y = Variable(batch_y)

        for net, opt, l_his in zip(nets, optimizers, losses_his):
            output = net(b_x)
            loss = loss_func(output, b_y)

            # /********** Begin *********/
            # 反向传播
            opt.zero_grad()
            loss.backward()
            # 更新参数
            opt.step()
            # 记录损失
            l_his.append(loss.item())
            # /********** End *********/

# 画图
labels = ['SGD', 'Momentum', 'RMSprop', 'Adam']
for i, l_his in enumerate(losses_his):
    plt.plot(l_his, label=labels[i])
plt.legend(loc='best')
plt.xlabel('Steps')
plt.ylabel('Loss')
plt.ylim((0, 0.2))
plt.savefig(path + "outputimages/mylossTest.png")
