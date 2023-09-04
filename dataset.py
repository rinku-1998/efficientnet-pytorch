# 4. 定義資料集
import cv2
import pandas as pd
import albumentations as A
from torch.utils.data import Dataset


class ImageDataset(Dataset):

    def __init__(self, df_dataset: pd.DataFrame, transform: A.transforms):
        self.df_dataset = df_dataset
        self.transform = transform

    def __len__(self):
        return len(self.df_dataset)

    def __getitem__(self, idx):

        # 圖片
        img_path = self.df_dataset.iloc[idx, 0]
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        transform = self.transform(image=img)

        imgt = transform['image']

        # 標籤
        label = self.df_dataset.iloc[idx, 1]

        return imgt, label
