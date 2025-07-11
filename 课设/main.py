"""
数据下载地址: 链接: https://pan.baidu.com/s/1w9X_GsxFdLEH-1RHfnprZg?pwd=6mf3 提取码: 6mf3
"""


import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from DataSet import DehazeDataset
from ghostnet import DehazeNet

from torchvision import models
import torch.nn.functional as F

from tools import plot_losses

class ResNetPerceptualLoss(nn.Module):
    def __init__(self, layer_ids=[1, 2, 3]):
        super().__init__()
        resnet = models.resnet18(pretrained=True)
        layers = [
            nn.Sequential(resnet.conv1, resnet.bn1, resnet.relu, resnet.maxpool),  # 0
            resnet.layer1,  # 1
            resnet.layer2,  # 2
            resnet.layer3,  # 3
            resnet.layer4   # 4
        ]
        self.selected_layers = layer_ids
        self.blocks = nn.ModuleList(layers)
        for p in self.parameters():
            p.requires_grad = False

    def forward(self, x, y):
        mean = torch.tensor([0.485, 0.456, 0.406], device=x.device).view(1,3,1,1)
        std = torch.tensor([0.229, 0.224, 0.225], device=x.device).view(1,3,1,1)
        x = (x - mean) / std
        y = (y - mean) / std
        loss = 0
        x_f, y_f = x, y
        for i, block in enumerate(self.blocks):
            x_f = block(x_f)
            y_f = block(y_f)
            if i in self.selected_layers:
                loss += F.l1_loss(x_f, y_f)
        return loss

def psnr(img1, img2):
    mse = torch.mean((img1 - img2) ** 2)
    if mse == 0:
        return 100
    return 20 * torch.log10(1.0 / torch.sqrt(mse))

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    train_dataset = DehazeDataset(root_dir='data', mode='train', img_size=256)
    val_dataset = DehazeDataset(root_dir='data', mode='test', img_size=256)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False, num_workers=2)

    model = DehazeNet().to(device)
    criterion = nn.L1Loss()
    perceptual_loss = ResNetPerceptualLoss().to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-2)

    best_psnr = 0
    num_epochs = 1000
    save_dir = 'checkpoints'
    os.makedirs(save_dir, exist_ok=True)

    train_losses = list()
    val_losses = list()

    for epoch in range(num_epochs):
        model.train()
        train_loss = 0
        for hazy, gt in train_loader:
            hazy, gt = hazy.to(device), gt.to(device)
            optimizer.zero_grad()
            output = model(hazy)
            loss_pixel = criterion(output, gt)
            loss_perc = perceptual_loss(output, gt)
            loss = loss_pixel + 0.1 * loss_perc
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * hazy.size(0)
        train_loss /= len(train_loader.dataset)
        train_losses.append(train_loss)

        # 验证
        model.eval()
        val_loss = 0
        val_psnr = 0
        with torch.no_grad():
            for hazy, gt in val_loader:
                hazy, gt = hazy.to(device), gt.to(device)
                output = model(hazy)
                loss = criterion(output, gt)
                val_loss += loss.item() * hazy.size(0)
                val_psnr += psnr(output, gt) * hazy.size(0)
        val_loss /= len(val_loader.dataset)
        val_losses.append(val_loss)
        val_psnr /= len(val_loader.dataset)
        print(f"[INFO] Epoch {epoch+1}/{num_epochs} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Val PSNR: {val_psnr:.2f}")

        # PSNR提升则保存模型
        if val_psnr > best_psnr:
            best_psnr = val_psnr
            torch.save(model.state_dict(), os.path.join(save_dir, f'dehaze_best.pth'))
            print(f"[INFO] 模型已保存，当前最佳PSNR: {best_psnr:.2f}")

        # 保存最终模型
        if epoch == num_epochs - 1:
            torch.save(model.state_dict(), os.path.join(save_dir, f'dehaze_final.pth'))
            print(f"[INFO] 最终模型已保存")

        # 绘制损失曲线
    items = {'Train': train_losses, 'Test': val_losses}
    plot_losses(items)
if __name__ == '__main__':
    main()
