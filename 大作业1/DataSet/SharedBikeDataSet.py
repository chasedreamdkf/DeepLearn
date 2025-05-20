import pandas as pd
import numpy as np
from torch.utils.data import DataLoader, Dataset, random_split
import torch
import os

class SharedBikeDataSet(Dataset):
    def __init__(self, DataPath: str, transform=None):
        """
        从csv文件读取数据并进行预处理
        :param DataPath: 数据路径
        :param transform: 数据转换
        """
        # 读取CSV文件
        self.data = pd.read_csv(DataPath)
        
        # 特征预处理
        # 1. 将日期转换为时间戳
        self.data['dteday'] = pd.to_datetime(self.data['dteday'])
        self.data['timestamp'] = self.data['dteday'].astype(np.int64) // 10**9
        
        # 2. 对类别特征进行独热编码
        categorical_features = ['season', 'yr', 'mnth', 'hr', 'holiday', 'weekday', 
                              'workingday', 'weathersit']
        for feature in categorical_features:
            one_hot = pd.get_dummies(self.data[feature], prefix=feature)
            self.data = pd.concat([self.data, one_hot], axis=1)
            self.data.drop(feature, axis=1, inplace=True)
        
        # 3. 对数值特征进行归一化
        numerical_features = ['temp', 'atemp', 'hum', 'windspeed']
        for feature in numerical_features:
            mean = self.data[feature].mean()
            std = self.data[feature].std()
            self.data[feature] = (self.data[feature] - mean) / std
        
        # 4. 分离特征和标签
        self.features = self.data.drop(['instant', 'dteday', 'casual', 'registered', 'cnt'], axis=1)
        self.labels = self.data['cnt'].values  # 转换为numpy数组
        
        # 5. 确保数据类型正确并将数据转换为张量
        self.features = self.features.astype(np.float32)  # 显式转换为float32
        self.labels = self.labels.astype(np.float32)  # 显式转换为float32
        
        self.features = torch.from_numpy(self.features.values)  # 使用from_numpy替代FloatTensor
        self.labels = torch.from_numpy(self.labels)  # 使用from_numpy替代FloatTensor
        
        # 保存数据转换器
        self.transform = transform

    def __len__(self):
        """
        返回数据集大小
        """
        return len(self.features)

    def __getitem__(self, idx):
        """
        获取单个数据样本
        :param idx: 索引
        :return: 特征和标签
        """
        feature = self.features[idx]
        label = self.labels[idx]
        
        if self.transform:
            feature = self.transform(feature)
            
        return feature, label

    @staticmethod
    def split_and_save_dataset(dataset_path: str, test_size: float = 0.2, seed: int = 42):
        """
        将数据集分割为训练集和测试集并保存
        :param dataset_path: 数据集路径
        :param test_size: 测试集比例
        :param seed: 随机种子
        :return: None
        """
        # 设置随机种子
        torch.manual_seed(seed)
        
        # 创建数据集实例
        dataset = SharedBikeDataSet(dataset_path)
        
        # 计算训练集和测试集的大小
        total_size = len(dataset)
        test_length = int(total_size * test_size)
        train_length = total_size - test_length
        
        # 使用random_split进行数据集分割
        train_dataset, test_dataset = random_split(
            dataset, 
            [train_length, test_length],
            generator=torch.Generator().manual_seed(seed)
        )
        
        # 创建保存目录
        save_dir = os.path.join(os.path.dirname(dataset_path), 'processed_data')
        os.makedirs(save_dir, exist_ok=True)
        
        # 保存数据集
        torch.save({
            'train_dataset': train_dataset,
            'test_dataset': test_dataset,
            'train_size': train_length,
            'test_size': test_length
        }, os.path.join(save_dir, 'bike_sharing_dataset.pt'))
        
        print(f"训练集大小: {train_length}")
        print(f"测试集大小: {test_length}")
        print(f"数据已保存到: {save_dir}")

if __name__ == "__main__":
    # 数据集路径
    dataset_path = 'd:/Code/DeepLearn/大作业1/data/hour.csv'
    
    # 分割并保存数据集
    SharedBikeDataSet.split_and_save_dataset(dataset_path)