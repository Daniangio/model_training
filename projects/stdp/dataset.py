import os
import numpy as np
import cv2
from torch.utils.data import Dataset as BaseDataset
from scipy import ndimage
from matplotlib import pyplot as plt


def unison_shuffled_copies(a: list):
    try:
        p = np.random.permutation(len(a))
        return list(np.array(a)[p])
    except Exception:
        return a


class Dataset(BaseDataset):
    def __init__(
            self,
            images_dir,
            input_size,
            augmentation=None,
            max_batch_size=64
    ):
        self.logger = None
        # read data
        self.id = os.listdir(images_dir)[1]
        self.image_fp = os.path.join(images_dir, self.id)
        self.image = cv2.imread(self.image_fp, cv2.IMREAD_GRAYSCALE)
        self.on_cell_kernel = np.array([[-1/12, -1/12, -1/12, -1/12], [-1/12, 1/4, 1/4, -1/12],
                                       [-1/12, 1/4, 1/4, -1/12], [-1/12, -1/12, -1/12, -1/12]], np.float32)
        self.image = (self.image / 255).astype('float32')
        self.on_cell_image = self.on_cell_filter(self.image)
        self.h, self.w = self.image.shape

        self.v_patches = self.h // input_size
        self.h_patches = self.w // input_size
        self.input_size = input_size
        self.augmentation = augmentation
        self.max_batch_size = max_batch_size

    def __getitem__(self, i):
        j = i // self.h_patches
        k = i % self.h_patches
        patch = self.on_cell_image[j * self.input_size:j * self.input_size + self.input_size,
                                   k * self.input_size:k * self.input_size + self.input_size].reshape((-1,))

        # apply augmentations
        if self.augmentation:
            sample = self.augmentation(image=patch)
            patch = sample['image']

        on_cells_image_array = np.zeros((self.input_size**2,))
        on_cells_image_array[:patch.shape[0]] = patch
        return on_cells_image_array

    def __len__(self):
        return self.v_patches * self.h_patches

    def on_cell_filter(self, image):
        on_cell = ndimage.filters.convolve(
            image, self.on_cell_kernel)  # Image is in range [-1, 1]
        return (on_cell + 1) / 2  # Normalize in [0, 1] range
