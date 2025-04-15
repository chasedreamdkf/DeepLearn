import torch.nn as nn


# /********** Begin *********/

# 线性回归模型
class LinearRegression(nn.Module):

    def __init__(self):
    # 调用Module的初始化
        super(LinearRegression, self).__init__()
    # 输入和输出分别为一维
        self.linear = nn.Linear(1, 1)
    # module调用forward，将按forward进行前向传播，并构建网络
    def forward(self, x):
        out = self.linear(x)

# 实例化一个新建的模型变量model
model = LinearRegression()

# 输出该模型 model 的'.parameters'属性
print(model.parameters)

# /********** End *********/
