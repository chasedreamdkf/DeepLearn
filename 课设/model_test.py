import os
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from DataSet import DehazeDataset
from ghostnet import DehazeNet
from torchvision.utils import save_image

def sharpen(image):
    """使用拉普拉斯算子对图像进行锐化"""
    kernel = torch.tensor([[0, -1, 0], [-1, 5, -1], [0, -1, 0]], dtype=torch.float32, device=image.device)
    kernel = kernel.repeat(3, 1, 1, 1)  # 为RGB三通道复制
    sharpened_image = F.conv2d(image, kernel, padding=1, groups=3)
    return torch.clamp(sharpened_image, 0, 1)

def psnr(img1, img2):
    mse = torch.mean((img1 - img2) ** 2)
    if mse == 0:
        return 100.0
    return 20 * torch.log10(1.0 / torch.sqrt(mse)).item()

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    test_dataset = DehazeDataset(root_dir='data', mode='test', img_size=256)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

    model = DehazeNet().to(device)
    model.load_state_dict(torch.load('checkpoints/dehaze_final.pth', map_location=device))
    model.eval()

    save_dir = './temp/pred_imgs'
    os.makedirs(save_dir, exist_ok=True)

    total_psnr = 0
    with torch.no_grad():
        for idx, (hazy, gt) in enumerate(test_loader):
            hazy, gt = hazy.to(device), gt.to(device)
            output = model(hazy)
            
            # 计算PSNR（用原始输出）
            single_psnr = psnr(output, gt)
            print(f'Image {idx:04d} PSNR: {single_psnr:.2f}')
            total_psnr += single_psnr

            # 锐化后保存
            sharpened_output = sharpen(output)
            save_image(sharpened_output, os.path.join(save_dir, f'{idx:04d}_pred_sharpened.png'))
            save_image(output, os.path.join(save_dir, f'{idx:04d}_pred.png'))
            save_image(hazy, os.path.join(save_dir, f'{idx:04d}_hazy.png'))
            save_image(gt, os.path.join(save_dir, f'{idx:04d}_gt.png'))
            
    avg_psnr = total_psnr / len(test_loader)
    print(f'Average PSNR on test set: {avg_psnr:.2f}')

if __name__ == '__main__':
    main() 