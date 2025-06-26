import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from dataset import CatDog
import matplotlib.pyplot as plt

# 定义简单的CNN模型
class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)
        )
        self.classifier = nn.Sequential(
            nn.Linear(64 * 28 * 28, 128),
            nn.ReLU(),
            nn.Linear(128, 2)
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x


def main():
    dataroot = './animals'
    cat_dog = CatDog(dataroot)
    # 数据集按8:2分割
    total_len = len(cat_dog)
    train_len = int(0.8 * total_len)
    test_len = total_len - train_len
    train_set, test_set = random_split(cat_dog, [train_len, test_len])

    train_loader = DataLoader(train_set, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_set, batch_size=32, shuffle=False)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = SimpleCNN().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # 训练
    epochs = 5
    train_losses = []
    test_losses = []
    best_acc = 0.0
    best_model_path = 'best_model.pth'
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * images.size(0)
            _, predicted = torch.max(outputs, 1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)
        epoch_loss = running_loss / total
        train_losses.append(epoch_loss)
        print(f'Epoch {epoch+1}/{epochs}, Loss: {epoch_loss:.4f}, Acc: {correct/total:.4f}')

        # 每个epoch后在测试集上评估并保存最优模型
        model.eval()
        correct = 0
        total = 0
        test_running_loss = 0.0
        with torch.no_grad():
            for images, labels in test_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                test_running_loss += loss.item() * images.size(0)
                _, predicted = torch.max(outputs, 1)
                correct += (predicted == labels).sum().item()
                total += labels.size(0)
        acc = correct / total
        test_epoch_loss = test_running_loss / total
        test_losses.append(test_epoch_loss)
        if acc > best_acc:
            best_acc = acc
            torch.save(model.state_dict(), best_model_path)
        print(f'Test Accuracy: {acc:.4f} (Best: {best_acc:.4f}), Test Loss: {test_epoch_loss:.4f}')

    # 绘制损失曲线
    plt.figure()
    plt.plot(range(1, epochs+1), train_losses, marker='o', label='Train Loss')
    plt.plot(range(1, epochs+1), test_losses, marker='s', label='Test Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training & Test Loss Curve')
    plt.legend()
    plt.savefig('loss_curve.png')
    plt.show()
    print(f'最优模型已保存为 {best_model_path}')

if __name__ == "__main__":
    main()