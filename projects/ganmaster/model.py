import collections
import os

import cv2
import numpy as np
import torch
import torchvision
from torch.utils.data import DataLoader
from common.augmentation import base_augmentation
from common.base_model import BaseModel
from common.utils import generous_difference, laplacian_blend, reconstruct_from_laplacian_pyramid
from projects.ganmaster.dataset import Dataset
from projects.ganmaster.modules.completer import Completer
from settings import settings


class GANMaster(BaseModel):
    def __init__(self, **_):
        super(GANMaster, self).__init__()
        b1, b2 = .5, .999
        self.net = Completer(input_size=settings.coordmap_input_size, output_size=settings.coordmap_output_size,
                             features_map=settings.coordmap_features_map, hidden_layers=settings.coordmap_hidden_layers)
        self.optimizer = torch.optim.Adam(self.net.parameters(), lr=settings.coordmap_completer_lr[0], betas=(b1, b2))
        self.net_weights_filepath = os.path.join(settings.models_weights_dir, self.net.get_weights_filename())

        self.train_epochs = settings.coordmap_train_epochs
        self.train_dataset = Dataset(
            master_dir=settings.coordmap_train_images_dir,
            input_size=settings.coordmap_input_size,
            target_size=settings.coordmap_output_size,
            len_dataset=-1,
            augmentation=base_augmentation(size=(settings.coordmap_input_size, settings.coordmap_input_size)),
            preprocessing=self.net.get_preprocessing()
        )
        self.valid_dataset = Dataset(
            master_dir=settings.coordmap_valid_images_dir,
            input_size=settings.coordmap_input_size,
            target_size=settings.coordmap_output_size,
            len_dataset=20,
            augmentation=base_augmentation(size=(settings.coordmap_input_size, settings.coordmap_input_size)),
            preprocessing=self.net.get_preprocessing()
        )
        self.train_loader = DataLoader(self.train_dataset,
                                       shuffle=True,
                                       batch_size=settings.train_batch_size,
                                       num_workers=settings.train_num_workers)
        self.valid_loader = DataLoader(self.valid_dataset,
                                       shuffle=False,
                                       batch_size=settings.valid_batch_size,
                                       num_workers=settings.train_num_workers)
        self.criterion = torch.nn.BCELoss()
        self.lr_decay_steps = 0

        if not os.path.isdir(settings.models_weights_dir):
            os.makedirs(settings.models_weights_dir)

    def initialize(self, logger, device: torch.device, device2: torch.device):
        super(GANMaster, self).initialize(logger, device, device2)
        if os.path.isfile(self.net_weights_filepath):
            print(f'Loading model weights for net from {self.net_weights_filepath}')
            try:
                self.net.load_state_dict(torch.load(self.net_weights_filepath, map_location='cpu'), strict=False)
            except Exception as e:
                print(str(e))
        self.net = self.net.to(device)
        self.net.zero_grad()

    def train(self, epoch: int):
        self.net.train()
        losses_queue = collections.deque(maxlen=settings.dcgan_log_interval_batches)
        for batch_idx, (input_images, target_images) in enumerate(self.train_loader):
            input_images, target_images = input_images.to(self.device), target_images.to(self.device)
            output = self.net(input_images)
            loss = self.criterion(output, target_images)
            losses_queue.append(loss.item())

            loss.backward()
            self.optimizer.step()
            self.net.zero_grad()

            if (batch_idx + 1) % settings.coordmap_log_interval_batches == 0:
                mean_loss = sum(losses_queue) / len(losses_queue)
                self.logger.report_scalar(
                    "train", "loss", iteration=(epoch * len(self.train_loader) + batch_idx + 1), value=mean_loss)

                print(f'Train Epoch: {epoch}/{self.train_epochs} '
                      f'[{(batch_idx + 1) * len(input_images)}/{len(self.train_loader.dataset)} '
                      f'({100. * (batch_idx + 1) / len(self.train_loader):.0f}%)]\t'
                      f'Loss: {mean_loss:.6f}')

    def validate(self, epoch: int):
        self.net.eval()
        losses_queue = collections.deque()
        with torch.no_grad():
            for batch_idx, (input_images, target_images) in enumerate(self.valid_loader):
                input_images, target_images = input_images.to(self.device), target_images.to(self.device)

                output = self.net(input_images)
                loss = self.criterion(output, target_images)
                losses_queue.append(loss.item())

            mean_loss = sum(losses_queue) / len(losses_queue)
            self.logger.report_scalar(
                "test", "loss", iteration=epoch, value=mean_loss)

            print(f'Test Loss: {mean_loss:.6f}')

            # Save as image the last batch prediction
            if epoch % settings.coordmap_save_image_interval_epochs == 0:
                input_images_grid = torchvision.utils.make_grid(input_images.detach().cpu(),
                                                                nrow=input_images.size(0)).permute(1, 2, 0).numpy()
                self.logger.report_image("completer", "input images", iteration=epoch, image=input_images_grid)

                target_images_grid = torchvision.utils.make_grid(target_images.detach().cpu(),
                                                                 nrow=target_images.size(0)).permute(1, 2, 0).numpy()

                predicted_images_grid = torchvision.utils.make_grid(output.detach().cpu(),
                                                                    nrow=output.size(0)).permute(1, 2, 0).numpy()

                completer_grid = np.vstack((target_images_grid, predicted_images_grid))

                self.logger.report_image("completer", "generated images", iteration=epoch, image=completer_grid)
        return mean_loss

    def test(self):
        test_dataset = Dataset(
            master_dir=settings.coordmap_valid_images_dir,
            input_size=settings.coordmap_input_size,
            target_size=settings.coordmap_output_size,
            len_dataset=-2,
            augmentation=base_augmentation(size=(settings.coordmap_input_size, settings.coordmap_input_size)),
            preprocessing=self.net.get_preprocessing()
        )
        test_loader = DataLoader(test_dataset,
                                 shuffle=False,
                                 batch_size=1,
                                 num_workers=1)
        original_image = (test_dataset.master / 255).astype(np.float32)
        h, w, _ = original_image.shape
        final_image = np.zeros((h, w, 1), dtype=np.float32)
        self.net.eval()
        with torch.no_grad():
            for batch_idx, (input_image, target_image) in enumerate(test_loader):
                input_image = input_image.to(self.device)
                output = self.net(input_image)
                output_image = output.detach().cpu()[0, ...].permute(1, 2, 0).numpy()
                final_image[
                test_dataset.yy[batch_idx] + int(test_dataset.target_size * 0.5):test_dataset.yy[batch_idx] + int(
                    test_dataset.target_size * 1.5),
                test_dataset.xx[batch_idx] + int(test_dataset.target_size * 0.5):test_dataset.xx[batch_idx] + int(
                    test_dataset.target_size * 1.5), :] = output_image

        diff = np.abs(original_image - final_image)  # generous_difference(original_image, final_image)
        diff[diff < 0.2] = 0
        diff_image = np.hstack((final_image, original_image, diff))

        self.logger.report_image("completer test", "reconstructed image", iteration=1, image=diff_image)

        original_image = (original_image * 255).astype(np.uint8)
        final_image = (final_image * 255).astype(np.uint8)
        blend = laplacian_blend(original_image, final_image)

        self.logger.report_image("completer test", "blended image", iteration=1, image=blend)

        blend_min = laplacian_blend(original_image, final_image, flag=True)
        defects_image = np.abs(blend - blend_min)
        defects_image[:32, :] = 0
        defects_image[-32:, :] = 0
        defects_image[:, :32] = 0
        defects_image[:, -32:] = 0

        self.logger.report_image("completer test", "laplacian image", iteration=1, image=defects_image)

        _, defects_image = cv2.threshold(defects_image, 20, 255, cv2.THRESH_BINARY)
        defects_image = cv2.erode(defects_image, None, iterations=1)
        defects_image = cv2.dilate(defects_image, None, iterations=1)

        self.logger.report_image("completer test", "laplacian image binary", iteration=1, image=defects_image)

    def can_save_weights(self, epoch) -> bool:
        return epoch % 5 == 0 and epoch > 1

    def save_modules_weights(self):
        torch.save(self.net.state_dict(), self.net_weights_filepath)

    def adjust_lr(self, epoch: int):
        if self.lr_decay_steps < len(settings.coordmap_completer_lr_decay_interval_epochs) and \
                epoch > settings.coordmap_completer_lr_decay_interval_epochs[self.lr_decay_steps]:
            self.lr_decay_steps += 1
            old_lr = self.optimizer.param_groups[0]['lr']
            new_lr = settings.coordmap_completer_lr[
                min(self.lr_decay_steps, len(settings.coordmap_completer_lr) - 1)]
            if new_lr != old_lr:
                self.optimizer.param_groups[0]['lr'] = new_lr
                print(f'Changed learning rate of generator from {old_lr} to {new_lr}')
            self.logger.report_scalar("learning rate", "lr", iteration=epoch,
                                      value=self.optimizer.param_groups[0]['lr'])
