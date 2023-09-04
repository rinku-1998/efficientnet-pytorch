import albumentations as A
import cv2
from albumentations.pytorch import ToTensorV2


def define_transform():

    train_transform = A.Compose([
        A.Resize(height=224, width=224),
        A.ColorJitter(p=0.3,
                      brightness=0.3,
                      contrast=0.3,
                      saturation=0.3,
                      hue=0.05),
        # A.Rotate(p=0.3,
        #          crop_border=True,
        #          interpolation=cv2.INTER_CUBIC,
        #          limit=10),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2(),
    ])

    val_transform = A.Compose([
        A.Resize(height=224, width=224),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2(),
    ])

    return train_transform, val_transform
