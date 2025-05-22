import os
import torch
import torch.nn as nn
from matplotlib import pyplot as plt
from torch.utils.data import DataLoader, Dataset
from DataSet import CatDog

class CatDogClassic(nn.Module):
    """
    经典的猫狗分类模型,5层卷积
    """
    def __init__(self, input_channel):
        super(CatDogClassic, self).__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(input_channel, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),  # 输出: 32 x 112 x 112
            
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),  # 输出: 64 x 56 x 56
            
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),  # 输出: 128 x 28 x 28
            
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),  # 输出: 256 x 14 x 14
            
            nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),  # 输出: 512 x 7 x 7
            
            nn.Flatten(),  # 输出: 512 * 7 * 7 = 25088
            nn.Linear(512 * 7 * 7, 256),  # 修改这里的输入维度
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, 1),
            nn.Sigmoid()
        )

        # 初始化模型参数
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                # 卷积层使用kaiming初始化
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                # 全连接层使用xavier初始化
                nn.init.xavier_normal_(m.weight)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        return self.layers(x)

# 定义训练函数
def train_model(model, train_loader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    for images, labels in train_loader:
        images = images.to(device)
        labels = labels.float().to(device)
        
        # 前向传播
        outputs = model(images)
        loss = criterion(outputs.squeeze(), labels)
        
        # 反向传播和优化
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        
        # 计算准确率
        predicted = (outputs.squeeze() > 0.5).float()
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
    
    epoch_loss = running_loss / len(train_loader)
    epoch_acc = 100 * correct / total
    return epoch_loss, epoch_acc

# 添加验证函数
def validate(model, val_loader, criterion, device):
    model.eval()
    val_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for images, labels in val_loader:
            images = images.to(device)
            labels = labels.float().to(device)
            
            outputs = model(images)
            loss = criterion(outputs.squeeze(), labels)
            
            val_loss += loss.item()
            
            predicted = (outputs.squeeze() > 0.5).float()
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    
    epoch_loss = val_loss / len(val_loader)
    epoch_acc = 100 * correct / total
    return epoch_loss, epoch_acc

def plot_losses(items: dict):
    """绘制损失曲线"""
    plt.figure(figsize=(10, 6))
    labels = items.keys()
    title = " and ".join(labels)
    for label, values in items.items():
        plt.plot(values, label=label + ' loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title(f'{title} Loss Over Time')
    plt.legend()
    plt.grid(True)
    plt.show()
    plt.save('./loss_plot.png')

# 主训练循环
def main():
    """主函数"""
    if os.name == "nt":
        os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataroot = r'.\CatsDogs'
    train = CatDog(dataroot, status="train")
    val = CatDog(dataroot, status="val")
    # print(train[0])
    train_loader = DataLoader(train, batch_size=64, shuffle=True)
    val_loader = DataLoader(val, batch_size=64, shuffle=True)
    # print(train_loader.dataset[2000])
    # 训练模型
    model = CatDogClassic(3)
    model.to(device)
    criterion = nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    num_epochs = 100
    best_val_acc = 0.0
    
    train_losses = []
    val_losses = []

    for epoch in range(num_epochs):
        # 训练阶段
        train_loss, train_acc = train_model(model, train_loader, criterion, optimizer, device)
        train_losses.append(train_loss)
        # 验证阶段
        val_loss, val_acc = validate(model, val_loader, criterion, device)
        val_losses.append(val_loss)
        # 打印训练信息
        print(f'Epoch [{epoch+1}/{num_epochs}]')
        print(f'Training Loss: {train_loss:.4f}, Training Accuracy: {train_acc:.2f}%')
        print(f'Validation Loss: {val_loss:.4f}, Validation Accuracy: {val_acc:.2f}%')
        
        # 保存最佳模型
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), 'best_model.pth')
            print(f'Model saved with validation accuracy: {val_acc:.2f}%')
        print('-' * 60)
    
    # 绘制训练损失曲线
    plot_losses({"train": train_losses, "val": val_losses})

if __name__ == '__main__':
    main()