import os
import random
import numpy as np
import cv2
from torch.utils.data import Dataset as BaseDataset

from common.utils import fast_clustering, normalize_channels, equalize
from settings import settings


def unison_shuffled_copies(a: list, b: list):
    try:
        assert len(a) == len(b)
        p = np.random.permutation(len(a))
        return list(np.array(a)[p]), list(np.array(b)[p])
    except Exception:
        return a, b

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

    cropped_image = np.zeros((h + y_offset, w + x_offset, image.shape[-1]))
    crop = image[y_from:y_to, x_from:x_to, :]
    cropped_image[y_offset:y_offset + crop.shape[0], x_offset:x_offset + crop.shape[1], :] = crop
    return cropped_image

def crop_on_multiple_sizes(complete_image, y_from, y_to, x_from, x_to):
    h_orig, w_orig = y_to - y_from, x_to - x_from
    near_image = cv2.resize(crop_image(complete_image, y_from, x_from, h_orig, w_orig), (h_orig, w_orig), interpolation=cv2.INTER_AREA)
    y_from, x_from, h, w = double_area_side(y_from, x_from, h_orig, w_orig)
    mid_image = cv2.resize(crop_image(complete_image, y_from, x_from, h, w), (h_orig, w_orig), interpolation=cv2.INTER_AREA)
    return np.concatenate((near_image, mid_image), axis=2).astype(np.float32)


class Dataset(BaseDataset):
    def __init__(
            self,
            images_dir,
            masks_dir,
            channels=3,
            augmentation=None,
            full_image=False
    ):
        self.logger = None
        self.ids = os.listdir(images_dir)
        self.images_fps = [os.path.join(images_dir, image_id) for image_id in self.ids]
        self.masks_fps = [os.path.join(masks_dir, mask_id) for mask_id in self.ids]

        self.channels = channels
        self.augmentation = augmentation

        self.batch_size = settings.train_batch_size
        self.full_image = full_image

    def __getitem__(self, i):
        # read data
        try:
            image = cv2.imread(self.images_fps[i], cv2.IMREAD_COLOR if self.channels == 3 else cv2.IMREAD_GRAYSCALE)
            if len(image.shape) == 2:
                image = np.expand_dims(image, axis=2)
            image = equalize(image)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
            mask = np.expand_dims(cv2.imread(self.masks_fps[i], cv2.IMREAD_GRAYSCALE), axis=2)
            h, w, _ = image.shape
        except Exception:
            image = cv2.imread(self.images_fps[i+1], cv2.IMREAD_COLOR if self.channels == 3 else cv2.IMREAD_GRAYSCALE)
            if len(image.shape) == 2:
                image = np.expand_dims(image, axis=2)
            image = equalize(image)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
            mask = np.expand_dims(cv2.imread(self.masks_fps[i+1], cv2.IMREAD_GRAYSCALE), axis=2)
            h, w, _ = image.shape

        # apply augmentations
        if self.augmentation:
            sample = self.augmentation(image=image, mask=mask)
            image, mask = sample['image'], sample['mask']
            # print(image.shape, mask.shape, self.images_fps[i], self.masks_fps[i])

        images = []
        masks = []
        for y in range(0, h, settings.segmentation_input_size):
            for x in range(0, w, settings.segmentation_input_size):
                if np.any(mask[y:y + settings.segmentation_input_size, x:x + settings.segmentation_input_size,
                          :] > 0) or self.full_image:
                    y_from, y_to = y, y + settings.segmentation_input_size
                    x_from, x_to = x, x + settings.segmentation_input_size
                    if y + settings.segmentation_input_size > h:
                        y_from, y_to = h - settings.segmentation_input_size, h
                    if x + settings.segmentation_input_size > w:
                        x_from, x_to = w - settings.segmentation_input_size, w
                    cropped_image = crop_on_multiple_sizes(image, y_from, y_to, x_from, x_to)
                    normalized_image = normalize_channels(cropped_image)
                    images.append(normalized_image)
                    masks.append(mask[y_from:y_to, x_from:x_to, :])
        while len(images) < self.batch_size:
            y = random.randint(0, h - settings.segmentation_input_size)
            x = random.randint(0, w - settings.segmentation_input_size)
            cropped_image = crop_on_multiple_sizes(image, y, y + settings.segmentation_input_size, x, x + settings.segmentation_input_size)
            normalized_image = normalize_channels(cropped_image)
            images.append(normalized_image)
            input_mask = mask[y:y + settings.segmentation_input_size, x:x + settings.segmentation_input_size, :]
            masks.append(input_mask)
        if len(images) > self.batch_size and not self.full_image:
            images, masks = unison_shuffled_copies(images, masks)
            images = images[:self.batch_size]
            masks = masks[:self.batch_size]
        try:
            images_batch = (np.stack(images[:self.batch_size], axis=-1) / 255).astype('float32')
            masks_batch = (np.stack(masks[:self.batch_size], axis=-1) / 255).astype('float32')
        except Exception as e:
            print(str(e))
            for ms in masks:
                print(ms.shape)
            print(self.images_fps[i])
            print('image:', image.shape, 'mask:', mask.shape)
            print(y, x)
            raise e
        if self.full_image:
            return images_batch.transpose((3, 2, 0, 1)), masks_batch.transpose((3, 2, 0, 1)), (w // settings.segmentation_input_size) + 1
        return images_batch.transpose((3, 2, 0, 1)), masks_batch.transpose((3, 2, 0, 1))

    def __len__(self):
        return len(self.ids)
