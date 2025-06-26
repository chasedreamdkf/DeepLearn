import os
from torch.utils.data import Dataset
from PIL import Image
from torchvision import transforms

class DehazeDataset(Dataset):
    def __init__(self, root_dir, mode='train', img_size=256):
        self.hazy_dir = os.path.join(root_dir, mode, 'hazy')
        self.gt_dir = os.path.join(root_dir, mode, 'GT')
        self.img_size = img_size
        self.hazy_imgs = sorted(os.listdir(self.hazy_dir))
        self.gt_imgs = sorted(os.listdir(self.gt_dir))
        self.transform = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor()
        ])

    def __len__(self):
        return len(self.hazy_imgs)

    def __getitem__(self, idx):
        hazy_path = os.path.join(self.hazy_dir, self.hazy_imgs[idx])
        gt_path = os.path.join(self.gt_dir, self.gt_imgs[idx])
        hazy_img = Image.open(hazy_path).convert('RGB')
        gt_img = Image.open(gt_path).convert('RGB')
        hazy_img = self.transform(hazy_img)
        gt_img = self.transform(gt_img)
        return hazy_img, gt_img 