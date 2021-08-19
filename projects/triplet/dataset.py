import os
import random
import numpy as np
import cv2
from torch.utils.data import Dataset as BaseDataset


class Dataset(BaseDataset):
    """Read test_images, apply augmentation and preprocessing transformations.

    Args:
        images_dir (str): path to images folder
        masks_dir (str): path to segmentation masks folder
        classes (list): ordered list of integers containing the pixel value of the classes on the mask
        augmentation (albumentations.Compose): data transfromation pipeline
            (e.g. flip, scale, etc.)
        preprocessing (albumentations.Compose): data preprocessing
            (e.g. noralization, shape manipulation, etc.)

    """

    def __init__(
            self,
            images_dir,
            masks_dir,
            classes,
            augmentation=None,
            preprocessing=None,
            random_masks=False
    ):
        self.ids = os.listdir(images_dir)
        self.mask_ids = os.listdir(masks_dir)
        self.images_fps = [os.path.join(images_dir, image_id) for image_id in self.ids]
        self.masks_fps = [os.path.join(masks_dir, mask_id) for mask_id in self.mask_ids]
        self.random_masks = random_masks

        self.class_values = classes

        self.augmentation = augmentation
        self.preprocessing = preprocessing

    def __getitem__(self, i):

        # read data
        image = cv2.imread(self.images_fps[i])
        if self.random_masks:
            j = random.randint(0, len(self.masks_fps) - 1)
            mask = cv2.imread(self.masks_fps[j], 0)
        else:
            mask = cv2.imread(self.masks_fps[i], 0)

        # extract certain classes from mask
        masks = [(mask == v) for v in self.class_values]
        mask = np.stack(masks, axis=-1).astype('float')
        mask = np.expand_dims(cv2.resize(mask, (image.shape[1], image.shape[0])), axis=2)

        # apply augmentations
        if self.augmentation:
            sample = self.augmentation(image=image, mask=mask)
            image, mask = sample['image'], sample['mask']

        # apply preprocessing
        if self.preprocessing:
            sample = self.preprocessing(image=image, mask=mask)
            image, mask = sample['image'], sample['mask']

        mask[mask > 0] = 1
        return image, mask

    def __len__(self):
        return len(self.ids)
