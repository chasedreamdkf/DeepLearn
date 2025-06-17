import os
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

class DehazeData(Dataset):
    def __init__(self, root: str, status: str, transform=None):
        """
        获取图片地址
        :param root:
        :param status:
        :param transform:
        """
        self.imgs = list()
        data_dir = os.listdir(os.path.join(root, status))
        print(data_dir)



if __name__ == "__main__":
    datapath = './DehazeData'
    data = DehazeData(datapath, "train")