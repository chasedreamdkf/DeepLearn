import os
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms


class CatDog(Dataset):
    def __init__(self, root: str, status: str, transform=None):
        """
        获取图片地址
        :param root: 数据根目录
        :param status: 数据集分类，"train"：训练集；"test"：测试集；"val"：验证集
        """
        self.imgs = list()
        datapath = os.path.join(root, status)
        # print("datapath:", datapath)
        datasirs = os.listdir(datapath)
        # print("datasirs:", datasirs)
        for item in datasirs:
            img_path = os.path.join(datapath, item)
            imgs = [os.path.join(img_path, img) for img in os.listdir(img_path)]
            # print("img_path:", img_path)
            # print("imgs:", imgs)
            self.imgs.extend(imgs)

        if transform is None:
            normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                             std=[0.229, 0.224, 0.225]
                                             )
            if status != "train" or status == "test":
                self.transforms = transforms.Compose([
                    transforms.Resize(224),
                    transforms.CenterCrop(224),
                    transforms.ToTensor(),
                    normalize
                ])
            else:
                self.transforms = transforms.Compose([
                    transforms.Resize(256),
                    transforms.RandomResizedCrop(224),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    normalize
                ])


    def __getitem__(self, item):
        label = None
        data = None
        img_path = self.imgs[item]
        # print(img_path)
        img = Image.open(img_path)
        if "cat" in img_path:
            label = 0
        elif "dog" in img_path:
            label = 1
        data = self.transforms(img)
        return data, label
        # pass

    def __len__(self):
        return len(self.imgs)


if __name__ == "__main__":
    dataroot = r'.\CatsDogs'
    cat_dog = CatDog(dataroot, "val")
    i, l = cat_dog.__getitem__(100)
    print(i, l)