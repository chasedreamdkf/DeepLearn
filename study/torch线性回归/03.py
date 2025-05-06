import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from torch.autograd import Variable

import os
import sys



path = os.path.split(os.path.abspath(os.path.realpath(sys.argv[0])))[0] + os.path.sep

print(path)

# 超参数
input_size = 1
output_size = 1
num_epochs = 60
learning_rate = 0.001

# 数据集
x_train = np.array([[3.3], [4.4], [5.5], [6.71], [6.93], [4.168],
                    [9.779], [6.182], [7.59], [2.167], [7.042],
                    [10.791], [5.313], [7.997], [3.1]], dtype=np.float32)

y_train = np.array([[1.7], [2.76], [2.09], [3.19], [1.694], [1.573],
                    [3.366], [2.596], [2.53], [1.221], [2.827],
                    [3.465], [1.65], [2.904], [1.3]], dtype=np.float32)


# 线性回归模型
class LinearRegression(nn.Module):
    def __init__(self, input_size, output_size):
        super(LinearRegression, self).__init__()
        self.linear = nn.Linear(input_size, output_size)

    def forward(self, x):
        out = self.linear(x)
        return out


model = LinearRegression(input_size, output_size)

# 创建输出文件 output.txt
f = open(path + 'output.txt', 'w')
f.seek(0)
f.truncate()  # 清空文件

# /********** Begin *********/
# 创建损失函数MSELoss
criterion = torch.nn.MSELoss()
# 创建SGD的Optimizer，学习率l'r为0.001
optimizer = torch.optim.SGD(model.parameters(), lr=0.001)
# 训练模型
for epoch in range(num_epochs):
    # 将x_train，y_train数据转换为Variable类型
    inputs = Variable(torch.from_numpy(x_train))
    targets = Variable(torch.from_numpy(y_train))
    # Forward
    optimizer.zero_grad()
    outputs = model(inputs)
    loss = criterion(outputs, targets)
    # Backward
    loss.backward()
    # Optimize
    optimizer.step()
    # 共训练60次，分别10次输出一回loss信息，并将输出信息存到文件中
    if (epoch + 1) % 10 == 0:
        f.writelines('Epoch [%d/%d], Loss: %.4f \n' % (epoch + 1, num_epochs, loss.item()))
        print('Epoch [%d/%d], Loss: %.4f'
              % (epoch + 1, num_epochs, loss.item()))
f.close()

# /********** End *********/

# 保存模型
torch.save(model, path + 'model.pkl')
