"""
加载模型参数，测试模型
"""
import os

import torch
from main import CatDogClassic
from DataSet import CatDog
from torch.utils.data import DataLoader
from torchvision import transforms
import matplotlib.pyplot as plt

# 设置设备
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 模型路径
model_path = './temp/best_model.pth'

# 加载测试数据集
test_dataset = CatDog(root='./CatsDogs', status='test')
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=True)

# 初始化模型
model = CatDogClassic(input_channel=3)

# 加载模型参数
model.load_state_dict(torch.load(model_path))
model = model.to(device)
model.eval()

# 测试模型并显示结果
def test_model():
    correct = 0
    total = 0
    
    with torch.no_grad():
        for i, (images, labels) in enumerate(test_loader):
            images = images.to(device)
            labels = labels.float().to(device)
            
            outputs = model(images)
            predicted = (outputs.squeeze() > 0.5).float()
            
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            # 显示前100张图片的预测结果，分5次显示
            if i < 4:
                plt.figure(figsize=(20, 8))
                for j in range(min(20, len(images))):
                    plt.subplot(4, 5, j+1)
                    img = images[j].cpu().permute(1, 2, 0)
                    # 反归一化
                    img = img * torch.tensor([0.229, 0.224, 0.225]) + torch.tensor([0.485, 0.456, 0.406])
                    img = torch.clamp(img, 0, 1)
                    
                    plt.imshow(img)
                    plt.axis('off')
                    pred_label = "猫" if predicted[j].item() < 0.5 else "狗"
                    true_label = "猫" if labels[j].item() < 0.5 else "狗"
                    title_color = 'green' if predicted[j].item() == labels[j].item() else 'red'
                    plt.title(f'预测: {pred_label}\n实际: {true_label}', color=title_color)
                
                plt.tight_layout()
                plt.show()
    
    accuracy = 100 * correct / total
    print(f'测试集准确率: {accuracy:.2f}%')

if __name__ == '__main__':
    if os.name == "nt":
        os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
    # 设置中文字体
    plt.rcParams['font.sans-serif'] = ['Microsoft YaHei']  # 用来正常显示中文标签
    plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号
    test_model()

