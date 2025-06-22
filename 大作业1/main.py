import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch.utils.data import Dataset, DataLoader


# 创建数据集类
class BikeDataset(Dataset):
    def __init__(self, x, y):
        self.X = x
        self.y = y

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


# 定义神经网络模型
class BikeSharingNN(nn.Module):
    def __init__(self):
        super(BikeSharingNN, self).__init__()
        self.layer = nn.Sequential(
            nn.Linear(12, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.BatchNorm1d(32),
            nn.Dropout(0.3),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.ReLU(),
            nn.Dropout(0.2)
        )

    def forward(self, x):
        return self.layer(x)


def main():
    if not os.path.exists('./temp'):
        os.mkdir('./temp')
    if not os.path.exists('./temp/imgs'):
        os.mkdir('./temp/imgs')
    if not os.path.exists('./temp/log'):
        os.mkdir('./temp/log')
    # 设置随机种子
    torch.manual_seed(42)
    np.random.seed(42)

    # 检查CUDA是否可用
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'使用设备: {device}')

    # 加载数据
    data = pd.read_csv('data/hour.csv')

    # 数据预处理
    features = ['season', 'yr', 'mnth', 'hr', 'holiday', 'weekday', 'workingday',
                'weathersit', 'temp', 'atemp', 'hum', 'windspeed']
    X = data[features].values
    y = data['cnt'].values.reshape(-1, 1)

    # 标准化特征
    scaler_X = StandardScaler()
    scaler_y = StandardScaler()
    X_scaled = scaler_X.fit_transform(X)
    y_scaled = scaler_y.fit_transform(y)

    # 首先划分出测试集（10%）
    X_temp, X_test, y_temp, y_test = train_test_split(X_scaled, y_scaled, test_size=0.1, random_state=42)
    # 然后将剩余数据划分为训练集（8/9）和验证集（1/9），这样最终比例为8:1:1
    X_train, X_val, y_train, y_val = train_test_split(X_temp, y_temp, test_size=0.111, random_state=42)

    # 转换为PyTorch张量并移至GPU
    X_train = torch.FloatTensor(X_train).to(device)
    X_val = torch.FloatTensor(X_val).to(device)
    X_test = torch.FloatTensor(X_test).to(device)
    y_train = torch.FloatTensor(y_train).to(device)
    y_val = torch.FloatTensor(y_val).to(device)
    y_test = torch.FloatTensor(y_test).to(device)

    # 创建数据加载器
    train_dataset = BikeDataset(X_train, y_train)
    val_dataset = BikeDataset(X_val, y_val)
    test_dataset = BikeDataset(X_test, y_test)

    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=64)
    test_loader = DataLoader(test_dataset, batch_size=64)

    # 初始化模型、损失函数和优化器，并将模型移至GPU
    model = BikeSharingNN().to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    # 训练模型
    epochs = 100
    train_losses = []
    val_losses = []
    best_val_loss = float('inf')

    for epoch in range(epochs):
        # 训练阶段
        model.train()
        total_train_loss = 0
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)

            optimizer.zero_grad()
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()
            total_train_loss += loss.item()

        avg_train_loss = total_train_loss / len(train_loader)
        train_losses.append(avg_train_loss)

        # 验证阶段
        model.eval()
        total_val_loss = 0
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                outputs = model(X_batch)
                loss = criterion(outputs, y_batch)
                total_val_loss += loss.item()

        avg_val_loss = total_val_loss / len(val_loader)
        val_losses.append(avg_val_loss)

        # 保存最佳模型
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), './temp/best_model.pth')

        if (epoch + 1) % 10 == 0:
            print(f'Epoch [{epoch + 1}/{epochs}], Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}')

    # 绘制训练和验证损失曲线
    plt.figure(figsize=(10, 6))
    plt.plot(train_losses, label='Training Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig('./temp/imgs/loss_plot.png')
    plt.show()
    plt.close()

    # 加载最佳模型进行测试
    model.load_state_dict(torch.load('./temp/best_model.pth'))

    # 模型评估
    model.eval()
    with torch.no_grad():
        test_predictions = []
        test_targets = []
        for X_batch, y_batch in test_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            outputs = model(X_batch)
            test_predictions.extend(outputs.cpu().numpy()) # Collect predictions from current batch
            test_targets.extend(y_batch.cpu().numpy())   # Collect targets from current batch
    
        # Convert lists to numpy arrays after collecting all batches
        test_predictions = np.array(test_predictions)
        test_targets = np.array(test_targets)
            
        # Inverse transform predictions and targets
        test_predictions = scaler_y.inverse_transform(test_predictions)
        test_targets = scaler_y.inverse_transform(test_targets)
            
        # Calculate Root Mean Squared Error (RMSE)
        rmse = np.sqrt(np.mean((test_predictions - test_targets) ** 2))
        print(f'\nTest RMSE: {rmse:.2f}')
            
        # Output predictions for the first 10 test samples
        print('\n前10个测试样本的预测结果：')
        print('预测值\t\t真实值\t\t误差')
        print('-' * 40)
        for i in range(min(10, len(test_predictions))): # Ensure we don't go out of bounds
            pred = test_predictions[i][0]
            target = test_targets[i][0]
            error = abs(pred - target)
            print(f'{pred:.2f}\t\t{target:.2f}\t\t{error:.2f}')


if __name__ == "__main__":
    main()