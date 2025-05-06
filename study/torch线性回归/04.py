import torch
import numpy as np
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt

from torch.autograd import Variable
import torch.nn as nn

import warnings

warnings.filterwarnings('ignore')

import os, sys

path = os.path.split(os.path.abspath(os.path.realpath(sys.argv[0])))[0] + os.path.sep
path = path[:-6]
print("validation path:", path)


# Linear Regression Model
class LinearRegression(nn.Module):
    def __init__(self, input_size, output_size):
        super(LinearRegression, self).__init__()
        self.linear = nn.Linear(input_size, output_size)

    def forward(self, x):
        out = self.linear(x)
        return out


model = LinearRegression(1, 1)

x_train = np.array([[3.3], [4.4], [5.5], [6.71], [6.93], [4.168],
                    [9.779], [6.182], [7.59], [2.167], [7.042],
                    [10.791], [5.313], [7.997], [3.1]], dtype=np.float32)

y_train = np.array([[1.7], [2.76], [2.09], [3.19], [1.694], [1.573],
                    [3.366], [2.596], [2.53], [1.221], [2.827],
                    [3.465], [1.65], [2.904], [1.3]], dtype=np.float32)

# 加载整个模型
model = torch.load(path + 'step3/model.pkl')

# /********** Begin *********/
# 将模型转化为测试模式
model.eval()
# 利用 model 计算预测值,并赋值给predicted
predict = model(Variable(torch.from_numpy(x_train)))
predicted = predict.data.numpy()
print(predicted)
# 画图
plt.plot(x_train, y_train, 'ro', label='Original data')
plt.plot(x_train, predicted, label='Fitted line')
plt.legend()
plt.savefig(path + "step4/outputimages/mylossTest.png")

# /********** End *********/

