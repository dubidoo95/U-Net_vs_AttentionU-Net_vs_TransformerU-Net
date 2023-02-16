import os
import cv2
from torch.utils.data import Dataset
import albumentations as A
from albumentations.pytorch import ToTensorV2
import glob
import pandas as pd
from PIL import Image
import numpy as np

# Resize images to put into model and augmentate images to put various data into model.
transforms = A.Compose([ 
    A.Resize(width = 512, height = 512, p=1.0),
    A.HorizontalFlip(p=0.5),
    A.VerticalFlip(p=0.5),
    A.RandomRotate90(p=0.5),
    ToTensorV2(),
])

def make_dataset(path):
    """
    Load data from local PC.
    'path' means where data saved.
    """
    folders = [folder for folder in os.listdir(path) if os.path.isdir(os.path.join(path, folder))]
    images = []
    masks = []

    for folder in folders:
        for file in os.listdir(os.path.join(path, folder)):
            if "mask" in file:
                masks.append(os.path.join(path, folder, file))
            else:
                images.append(os.path.join(path, folder, file))
    return images, masks

class CustomDataset(Dataset):
    """
    Receive path and collect the images under the path.
    """
    def __init__(self, path, transform=None):
        super().__init__()
        self.path = path
        self.transform = transform
        self.images, self.masks = make_dataset(path)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = cv2.imread(self.images[idx])
        mask = cv2.imread(self.masks[idx], 0)

        transform = self.transform(image=image, mask=mask)
        # To normalize images and masks, divide by 255 then all pixels have values between 0 and 1.
        image = transform["image"] / 255.
        mask = transform["mask"] / 255.
        return image, mask