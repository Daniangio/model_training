import os
import random
import numpy as np
import cv2
import scipy
from torch.utils.data import Dataset as BaseDataset

from common.utils import fast_clustering


def normalize(img):
    img[img < 0] = 0
    return (img - np.min(img)) / (np.max(img) - np.min(img))


class Dataset(BaseDataset):
    """Read images, apply augmentation and preprocessing transformations.

    Args:
        images_dir (str): path to images folder
        augmentation (albumentations.Compose): data transformation pipeline
            (e.g. flip, scale, etc.)
        preprocessing (albumentations.Compose): data preprocessing
            (e.g. normalization, shape manipulation, etc.)

    """

    def __init__(
            self,
            master_dir,
            input_size,
            target_size,
            len_dataset=1000,
            augmentation=None,
            preprocessing=None
    ):
        self.ids = os.listdir(master_dir)
        self.ids.sort()
        master = cv2.imread(os.path.join(master_dir, self.ids[2]), cv2.IMREAD_GRAYSCALE)
        self.master = fast_clustering(master, n_clusters=5)

        h, w, _ = self.master.shape
        self.input_size = int(input_size)
        self.target_size = int(target_size)
        self.margin = int((self.input_size - self.target_size) / 2)

        if len_dataset < 0:
            self.yy, self.xx = [], []
            step = int(self.target_size/4) if len_dataset == -1 else self.target_size
            for y in range(0, self.master.shape[0] - self.input_size, step):
                for x in range(0, self.master.shape[1] - self.input_size, self.target_size):
                    if (y + self.input_size > h) or (x + self.input_size > w):
                        continue
                    self.yy.append(y)
                    self.xx.append(x)
            self.len_dataset = len(self.yy)
        else:
            self.yy, self.xx = None, None
            self.len_dataset = len_dataset
        self.augmentation = augmentation
        self.preprocessing = preprocessing

    def __getitem__(self, i):
        if self.yy is None:
            y = random.randint(0, self.master.shape[0] - self.input_size - 1)
            x = random.randint(0, self.master.shape[1] - self.input_size - 1)
        else:
            y = self.yy[i]
            x = self.xx[i]

        complete_image = self.master[y:y + self.input_size, x:x + self.input_size, :].copy()

        # apply augmentations
        if self.augmentation:
            sample = self.augmentation(image=complete_image)
            complete_image = sample['image']

        # apply preprocessing
        if self.preprocessing:
            sample = self.preprocessing(image=complete_image)
            complete_image = sample['image']

        target_image = complete_image[:, self.margin:-self.margin, self.margin:-self.margin].copy()
        assert target_image.shape[1] == target_image.shape[2] == self.target_size
        input_image = complete_image
        input_image[:, self.margin:-self.margin, self.margin:-self.margin] = 0

        return input_image, target_image

    def __len__(self):
        return self.len_dataset
