# 在这个例子中，我们首先定义了一个 myDataset 类，其中包含了 csv 文件的读取和转换操作。
# 然后我们创建了一个 DataLoader 对象，指定了批次大小和是否打乱数据。最后，我们通过迭代 DataLoader，
# 可以访问每个批次的数据。需要注意的是，每个批次返回的数据是一个张量，张量的大小是 (batch_size, n_features)，
# 其中 batch_size 是批次大小，n_features 是特征的数量（即每个数据点的维数）。

import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

class myDataset(Dataset):
    def __init__(self, csv_file, transform=None):
        # 读取 csv 文件并将数据转换为 numpy 数组
        self.data = pd.read_csv(csv_file)['value'].values
        # 转换为 PyTorch 的张量
        self.data = torch.tensor(self.data, dtype=torch.float32)
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        sample = self.data[index]
        if self.transform:
            sample = self.transform(sample)
        return sample

# 创建 DataLoader 对象
dataset = myDataset(r'E:\Develop\monitor_data\cpu_hw3_096_2023-02-17_2023-03-03.csv')
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

# 迭代数据
for batch in dataloader:
    # 对每个批次的数据进行操作
    print(batch)
