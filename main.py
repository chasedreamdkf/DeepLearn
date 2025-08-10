"""example"""
import os
import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader, TensorDataset


class Net(nn.Module):
    """简单的全连接神经网络模型"""

    def __init__(self, input_size=100, hidden_size=50, output_size=1):
        """初始化网络结构
        :param input_size: 输入特征维度
        :param hidden_size: 隐藏层维度
        :param output_size: 输出维度
        """
        super().__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, output_size)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, inputs):
        """前向传播
        :param: inputs: 输入张量，形状为 [batch_size, input_size]
        :return: 输出张量，形状为 [batch_size, output_size]
        """
        layer = self.fc1(inputs)
        layer = self.relu(layer)
        layer = self.fc2(layer)
        return layer


def create_optimizer(model, lr=1e-2, weight_decay=1e-5):
    """创建优化器，对权重和偏置使用不同的权重衰减
    :param model: 神经网络模型
    :param lr: 学习率
    :param weight_decay: 权重衰减系数
    :return: 优化器实例
    """
    weight_p, bias_p = [], []
    for name, p in model.named_parameters():
        if 'bias' in name:
            bias_p += [p]
        else:
            weight_p += [p]

    optimizer = torch.optim.SGD([
        {'params': weight_p, 'weight_decay': weight_decay},
        {'params': bias_p, 'weight_decay': 0}
    ], lr=lr, momentum=0.9)

    return optimizer


def train_one_step(inputs, labels, model, criterion, optimizer):
    """执行一步训练
    :param inputs: 输入数据
    :param labels: 标签数据
    :param model: 神经网络模型
    :param criterion: 损失函数
    :param optimizer: 优化器
    :return: 当前步骤的损失值
    """
    # 前向传播
    logits = model(inputs)
    loss = criterion(input=logits, target=labels)

    # 反向传播和优化
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    return loss.item()


def generate_dummy_data(num_samples=1000, input_size=100):
    """生成模拟数据用于训练和测试
    :param num_samples: 样本数量
    :param input_size: 特征维度
    :return: 训练数据和测试数据
    """
    # 生成随机特征
    X = torch.randn(num_samples, input_size)

    # 生成目标值 (简单的线性关系加噪声)
    w = torch.randn(input_size, 1) * 0.1
    y = torch.mm(X, w) + torch.randn(num_samples, 1) * 0.1

    # 划分训练集和测试集
    train_size = int(0.8 * num_samples)
    X_train, y_train = X[:train_size], y[:train_size]
    X_test, y_test = X[train_size:], y[train_size:]

    return (X_train, y_train), (X_test, y_test)


def evaluate_model(model, X, y, criterion):
    """评估模型性能

    Args:
        model: 神经网络模型
        X: 输入特征
        y: 目标值
        criterion: 损失函数

    Returns:
        平均损失值
    """
    model.eval()
    with torch.no_grad():
        outputs = model(X)
        loss = criterion(input=outputs, target=y)
    model.train()
    return loss.item()


def train_model(model, train_data, test_data, epochs=100, batch_size=32, lr=1e-2):
    """训练模型的完整流程

    Args:
        model: 神经网络模型
        train_data: 训练数据元组 (X_train, y_train)
        test_data: 测试数据元组 (X_test, y_test)
        epochs: 训练轮数
        batch_size: 批次大小
        lr: 初始学习率
    """
    X_train, y_train = train_data
    X_test, y_test = test_data

    # 创建数据加载器
    train_dataset = TensorDataset(X_train, y_train)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    # 定义损失函数
    criterion = nn.MSELoss()

    # 创建优化器
    optimizer = create_optimizer(model, lr=lr)

    # 学习率调度器
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5, verbose=True
    )

    # 训练循环
    for epoch in range(epochs):
        epoch_loss = 0.0
        batch_count = 0

        for inputs, labels in train_loader:
            loss = train_one_step(inputs, labels, model, criterion, optimizer)
            epoch_loss += loss
            batch_count += 1

        # 计算平均损失
        avg_train_loss = epoch_loss / batch_count

        # 在测试集上评估
        test_loss = evaluate_model(model, X_test, y_test, criterion)

        # 更新学习率
        scheduler.step(test_loss)

        # 每10个epoch打印一次结果
        if (epoch + 1) % 10 == 0:
            print(f'Epoch [{epoch+1}/{epochs}], Train Loss: {avg_train_loss:.4f}, Test Loss: {test_loss:.4f}')

    print('训练完成!')
    return model


def save_model(model, path='model.pth'):
    """保存模型"""
    torch.save(model.state_dict(), path)
    print(f'模型已保存到 {path}')


def load_model(model, path='model.pth'):
    """加载模型"""
    model.load_state_dict(torch.load(path))
    return model


if __name__ == "__main__":
    print(f"PyTorch版本: {torch.__version__}")

    # 设置随机种子以便结果可复现
    torch.manual_seed(42)
    np.random.seed(42)

    # 生成模拟数据
    train_data, test_data = generate_dummy_data(num_samples=1000, input_size=100)

    # 创建模型
    model = Net(input_size=100, hidden_size=50, output_size=1)

    # 训练模型
    trained_model = train_model(
        model,
        train_data,
        test_data,
        epochs=100,
        batch_size=32,
        lr=1e-2
    )

    print(model)

    # 保存模型
    save_model(trained_model, 'trained_model.pth')