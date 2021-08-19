import os
import cv2
import numpy as np
from torch.utils.data import Dataset as BaseDataset


class Dataset(BaseDataset):
    """Read images, apply augmentation and preprocessing transformations.

    Args:
        images_dir (str): path to images folder
        augmentation (albumentations.Compose): data transfromation pipeline
            (e.g. flip, scale, etc.)
        preprocessing (albumentations.Compose): data preprocessing
            (e.g. noralization, shape manipulation, etc.)

    """

    def __init__(
            self,
            images_dir,
            augmentation=None,
            preprocessing=None,
            channels=3
    ):
        self.ids = os.listdir(images_dir)
        self.images_fps = [os.path.join(images_dir, image_id) for image_id in self.ids]

        self.augmentation = augmentation
        self.preprocessing = preprocessing
        if channels not in (1, 3):
            raise Exception(f'Invalid number of channels: {channels}. Value must be 1 or 3')
        self.channels = channels

    def __getitem__(self, i):

        # read data
        if self.channels == 1:
            image = np.expand_dims(cv2.imread(self.images_fps[i], cv2.IMREAD_GRAYSCALE), axis=2)
        else:
            image = cv2.imread(self.images_fps[i])

        # apply augmentations
        if self.augmentation:
            sample = self.augmentation(image=image)
            image = sample['image']

        # apply preprocessing
        if self.preprocessing:
            sample = self.preprocessing(image=image)
            image = sample['image']

        return image, image

    def __len__(self):
        return len(self.ids)
