import pandas as pd
from torch.utils.data import DataLoader, Dataset

class SharedBikeDataSet(Dataset):
    def __init__(self, DataPath: str, mode: str, transform=None):
        """
        从csv文件读取数据
        :param DataPath: 数据路径
        :param mode: 
        :param transform:
        """