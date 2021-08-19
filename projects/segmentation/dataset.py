import os
import random
import numpy as np
import cv2
from sklearn.cluster import KMeans
from torch.utils.data import Dataset as BaseDataset

from settings import settings


def fast_clustering(sub_image):
    x_train = sub_image.reshape(-1, sub_image.shape[2])
    kmeans = KMeans(n_clusters=settings.segmentation_color_clusters, random_state=0, max_iter=3).fit(x_train)
    cc = kmeans.cluster_centers_
    return cc


def reduced_colors_image(image, cc):
    img_batches = None
    for center in cc:
        img_batch = np.linalg.norm(image - center, axis=2)
        if img_batches is None:
            img_batches = np.expand_dims(img_batch, axis=2)
        else:
            img_batches = np.append(img_batches, np.expand_dims(img_batch, axis=2), axis=2)
    nearest_center_indexes_image = np.argmin(img_batches, axis=2)
    return cc[nearest_center_indexes_image].astype(np.uint8)


def unison_shuffled_copies(a: list, b: list):
    assert len(a) == len(b)
    p = np.random.permutation(len(a))
    return list(np.array(a)[p]), list(np.array(b)[p])


class Dataset(BaseDataset):
    def __init__(
            self,
            images_dir,
            masks_dir,
            channels=3,
            augmentation=None
    ):
        self.logger = None
        self.ids = os.listdir(images_dir)
        self.mask_ids = os.listdir(masks_dir)
        self.images_fps = [os.path.join(images_dir, image_id) for image_id in self.ids]
        self.masks_fps = [os.path.join(masks_dir, mask_id) for mask_id in self.mask_ids]

        self.channels = channels
        self.augmentation = augmentation

    def __getitem__(self, i):
        return np.ones((self.channels, 1, settings.segmentation_input_size, settings.segmentation_input_size)).astype('float32'), \
               np.zeros((self.channels, 1, settings.segmentation_input_size, settings.segmentation_input_size)).astype('float32')
        # read data
        image = cv2.imread(self.images_fps[i], cv2.IMREAD_COLOR if self.channels == 3 else cv2.IMREAD_GRAYSCALE)
        if len(image.shape) == 2:
            image = np.expand_dims(image, axis=2)
        mask = np.expand_dims(cv2.imread(self.masks_fps[i], cv2.IMREAD_GRAYSCALE), axis=2)
        h, w, _ = image.shape

        # apply augmentations
        if self.augmentation:
            sample = self.augmentation(image=image, mask=mask)
            image, mask = sample['image'], sample['mask']

        cc = fast_clustering(image[h // 4:h // 4 + 2 * settings.segmentation_input_size,
                             w // 4:w // 4 + 2 * settings.segmentation_input_size, :])

        images = []
        masks = []
        for y in range(0, h, settings.segmentation_input_size):
            for x in range(0, w, settings.segmentation_input_size):
                if np.any(mask[y:y + settings.segmentation_input_size, x:x + settings.segmentation_input_size,
                          :] > 0):
                    if (y + settings.segmentation_input_size > h) or (x + settings.segmentation_input_size > w):
                        y_from, y_to = h - settings.segmentation_input_size, h
                        x_from, x_to = w - settings.segmentation_input_size, w
                    else:
                        y_from, y_to = y, y + settings.segmentation_input_size
                        x_from, x_to = x, x + settings.segmentation_input_size
                    images.append(reduced_colors_image(image[y_from:y_to, x_from:x_to, :], cc))
                    masks.append(mask[y_from:y_to, x_from:x_to, :])
        while len(images) < max(self.channels * 2, 2):
            y = random.randint(0, h - settings.segmentation_input_size)
            x = random.randint(0, w - settings.segmentation_input_size)
            images.append(
                reduced_colors_image(
                    image[y:y + settings.segmentation_input_size, x:x + settings.segmentation_input_size, :], cc))
            masks.append(
                mask[y:y + settings.segmentation_input_size, x:x + settings.segmentation_input_size, :])
        if len(images) > self.channels * 2:
            images, masks = unison_shuffled_copies(images, masks)
            images = images[:self.channels * 2]
            masks = masks[:self.channels * 2]
        images_batch_1 = (np.stack(images[:self.channels], axis=0) / 255).astype('float32')
        masks_batch_1 = (np.stack(masks[:self.channels], axis=0) / 255).astype('float32')
        images_batch_2 = (np.stack(images[self.channels:], axis=0) / 255).astype('float32')
        masks_batch_2 = (np.stack(masks[self.channels:], axis=0) / 255).astype('float32')
        images_batch = np.append(images_batch_1, images_batch_2, axis=-1)
        masks_batch = np.append(masks_batch_1, masks_batch_2, axis=-1)
        return images_batch.transpose((3, 0, 1, 2)), masks_batch.transpose((3, 0, 1, 2))

    def __len__(self):
        return len(self.ids)
