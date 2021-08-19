import os
import random
import numpy as np
import cv2
from torch.utils.data import Dataset as BaseDataset

from common.utils import fast_clustering


def normalize(img):
    img[img < 0] = 0
    return (img - np.min(img)) / (np.max(img) - np.min(img))


def double_area_side(y, x, h, w):
    h_half_side = h // 2
    y -= h_half_side
    h += h_half_side * 2

    w_half_side = w // 2
    x -= w_half_side
    w += w_half_side * 2
    return y, x, h, w


def crop_image(image, y, x, h, w):
    y_from, x_from = y, x
    y_to, x_to = y + h, x + w
    y_offset, x_offset = 0, 0
    if y_from < 0:
        y_offset = -y_from
        y_from = 0
    if x_from < 0:
        x_offset = -x_from
        x_from = 0

    cropped_image = np.zeros((image.shape[0], h + y_offset, w + x_offset))
    crop = image[:, y_from:y_to, x_from:x_to]
    cropped_image[:, y_offset:y_offset + crop.shape[1], x_offset:x_offset + crop.shape[2]] = crop
    return np.transpose(cropped_image, (1, 2, 0))


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
            image_size,
            len_dataset=1000,
            augmentation=None,
            preprocessing=None
    ):
        self.ids = os.listdir(master_dir)
        self.ids.sort()
        master = cv2.imread(os.path.join(master_dir, self.ids[2]), cv2.IMREAD_COLOR)
        self.master = fast_clustering(master, n_clusters=5)

        h, w, _ = self.master.shape
        self.image_size = int(image_size)

        if len_dataset < 0:
            step = 1
            self.yy, self.xx = [], []
            step = int(self.image_size * step) if len_dataset == -1 else self.image_size
            for y in range(self.image_size, self.master.shape[0] - 2 * self.image_size, step):
                for x in range(self.image_size, self.master.shape[1] - 2 * self.image_size, step):
                    if (y + 2 * self.image_size > h) or (x + 2 * self.image_size > w):
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
            y = random.randint(self.image_size, self.master.shape[0] - 2 * self.image_size - 1)
            x = random.randint(self.image_size, self.master.shape[1] - 2 * self.image_size - 1)
        else:
            y = self.yy[i]
            x = self.xx[i]

        complete_image = self.master.copy()

        # apply augmentations
        if self.augmentation:
            sample = self.augmentation(image=complete_image)
            complete_image = sample['image']

        # apply preprocessing -> numpy to torch format
        if self.preprocessing:
            sample = self.preprocessing(image=complete_image)
            complete_image = sample['image']

        cropped_image = complete_image[:, y - self.image_size:y + 2 * self.image_size,
                        x - self.image_size:x + 2 * self.image_size]

        expert_one_image = self.prepare_expert_one(cropped_image)
        target_image = cropped_image[:, self.image_size:2 * self.image_size, self.image_size:2 * self.image_size]
        expert_two_image = self.prepare_expert_two(complete_image, y, x)

        return expert_one_image, expert_two_image, target_image

    def __len__(self):
        return self.len_dataset

    def prepare_expert_one(self, image):
        final_image = image[:, :self.image_size, :self.image_size]
        final_image = np.concatenate((final_image, image[:, self.image_size:2 * self.image_size, :self.image_size]),
                                     axis=0)
        final_image = np.concatenate((final_image, image[:, -self.image_size:, :self.image_size]),
                                     axis=0)
        final_image = np.concatenate((final_image, image[:, :self.image_size, self.image_size:2 * self.image_size]),
                                     axis=0)
        final_image = np.concatenate((final_image, image[:, -self.image_size:, self.image_size:2 * self.image_size]),
                                     axis=0)
        final_image = np.concatenate((final_image, image[:, :self.image_size, -self.image_size:]),
                                     axis=0)
        final_image = np.concatenate((final_image, image[:, self.image_size:2 * self.image_size, -self.image_size:]),
                                     axis=0)
        final_image = np.concatenate((final_image, image[:, -self.image_size:, -self.image_size:]),
                                     axis=0)
        return final_image

    def prepare_expert_two(self, complete_image, y, x):
        y, x, h, w = y, x, self.image_size, self.image_size
        # close_image = crop_image(complete_image, y, x, h, w)
        y, x, h, w = double_area_side(y, x, h, w)
        mid_image = cv2.resize(crop_image(complete_image, y, x, h, w), (self.image_size, self.image_size), interpolation=cv2.INTER_AREA)
        y, x, h, w = double_area_side(y, x, h, w)
        far_image = cv2.resize(crop_image(complete_image, y, x, h, w), (self.image_size, self.image_size), interpolation=cv2.INTER_AREA)
        return np.transpose(np.concatenate((mid_image, far_image), axis=2), (2, 0, 1)).astype(np.float32)
