import cv2
import torch
import albumentations as A
from albumentations.pytorch import ToTensorV2
from torch.utils.data import Dataset

def get_transforms(is_train=True, img_size=224):
    if is_train:
        return A.Compose([
            A.Resize(img_size, img_size),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.RandomRotate90(p=0.5),
            A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5),
            A.Normalize(mean=(0.5,), std=(0.25,)),
            ToTensorV2()
        ])
    return A.Compose([
        A.Resize(img_size, img_size),
        A.Normalize(mean=(0.5,), std=(0.25,)),
        ToTensorV2()
    ])

class SteelDataset(Dataset):
    def __init__(self, df, is_train=True):
        self.df = df.reset_index(drop=True)
        self.transforms = get_transforms(is_train=is_train)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img = cv2.imread(row['path'], cv2.IMREAD_GRAYSCALE)
        augmented = self.transforms(image=img[..., None])
        return augmented['image'], int(row['y'])