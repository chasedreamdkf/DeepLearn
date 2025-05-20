import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from torch.serialization import add_safe_globals
from torch.utils.data.dataset import Subset
from DataSet.SharedBikeDataSet import SharedBikeDataSet  # 添加这行导入语句

# 添加Subset到安全全局变量列表
add_safe_globals([Subset])

# 定义神经网络模型
class BikeSharingMLP(nn.Module):
    def __init__(self, input_size):
        super(BikeSharingMLP, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_size, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )
    
    def forward(self, x):
        return self.layers(x)

def train_model(model, train_loader, test_loader, criterion, optimizer, device, num_epochs=100):
    train_losses = []
    test_losses = []
    
    # 将模型移动到指定设备
    model = model.to(device)
    
    for epoch in range(num_epochs):
        # 训练阶段
        model.train()
        running_loss = 0.0
        for features, labels in train_loader:
            # 将数据移动到指定设备
            features = features.to(device)
            labels = labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(features)
            loss = criterion(outputs, labels.unsqueeze(1))
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        
        train_loss = running_loss / len(train_loader)
        train_losses.append(train_loss)
        
        # 测试阶段
        model.eval()
        test_loss = 0.0
        with torch.no_grad():
            for features, labels in test_loader:
                # 将数据移动到指定设备
                features = features.to(device)
                labels = labels.to(device)
                
                outputs = model(features)
                loss = criterion(outputs, labels.unsqueeze(1))
                test_loss += loss.item()
        
        test_loss = test_loss / len(test_loader)
        test_losses.append(test_loss)
        
        if (epoch + 1) % 10 == 0:
            print(f'Epoch [{epoch+1}/{num_epochs}], '
                  f'Train Loss: {train_loss:.4f}, '
                  f'Test Loss: {test_loss:.4f}')
    
    return train_losses, test_losses

def plot_losses(train_losses, test_losses):
    plt.figure(figsize=(10, 6))
    plt.plot(train_losses, label='Training Loss')
    plt.plot(test_losses, label='Testing Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Testing Loss Over Time')
    plt.legend()
    plt.grid(True)
    plt.savefig('loss_plot.png')
    plt.show()

def main():
    os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
    # 检查是否有可用的GPU
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'使用设备: {device}')
    # 加载数据集时指定weights_only=False
    data = torch.load('d:/Code/DeepLearn/大作业1/data/processed_data/bike_sharing_dataset.pt', weights_only=False)
    train_dataset = data['train_dataset']
    test_dataset = data['test_dataset']

    
    # 创建数据加载器
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
    print("train_dataset:\n", train_loader)
    print("test_dataset:\n", test_loader)
    
    # 获取输入特征的维度
    sample_features, _ = next(iter(train_loader))
    input_size = sample_features.shape[1]
    
    # 创建模型
    model = BikeSharingMLP(input_size)
    
    # 定义损失函数和优化器
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    
    # 训练模型
    train_losses, test_losses = train_model(
        model, train_loader, test_loader, criterion, optimizer, device, num_epochs=300
    )

    print("train_losses:\n", train_losses)
    print("test_losses:\n", test_losses)
    
    # 绘制损失曲线
    plot_losses(train_losses, test_losses)
    
    # 保存模型
    torch.save(model.state_dict(), 'bike_sharing_model.pth')
    print("模型训练完成并已保存！")

if __name__ == "__main__":
    main()