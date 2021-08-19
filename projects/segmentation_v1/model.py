import collections
import os
import time

import cv2
import numpy as np
import torch
import torchvision
from torch.utils.data import DataLoader
from common.augmentation import training_augmentation
from common.base_model import BaseModel, show_gpu_memory_info
from projects.segmentation_v1.dataset import Dataset
from projects.segmentation_v1.modules.segmentator import Segmentator
from settings import settings
import segmentation_models_pytorch as smp


class Segmentation(BaseModel):
    def __init__(self, **_):
        super(Segmentation, self).__init__()
        self.net = Segmentator(in_channels=3 * 2,
                               out_channels=1)
        self.train_epochs = settings.segmentation_train_epochs
        self.train_dataset = Dataset(
            images_dir=settings.segmentation_train_images_dir,
            masks_dir=settings.segmentation_train_masks_dir,
            channels=3,
            augmentation=training_augmentation(size=None)
        )
        self.valid_dataset = Dataset(
            images_dir=settings.segmentation_valid_images_dir,
            masks_dir=settings.segmentation_valid_masks_dir,
            channels=3
        )
        self.train_loader = DataLoader(self.train_dataset,
                                       shuffle=True,
                                       batch_size=1,
                                       num_workers=settings.train_num_workers)
        self.valid_loader = DataLoader(self.valid_dataset,
                                       shuffle=True,
                                       batch_size=1,
                                       num_workers=settings.train_num_workers)
        self.criterion = smp.utils.losses.DiceLoss()
        b1, b2 = .5, .999
        self.optimizer = torch.optim.Adam(self.net.parameters(), lr=settings.segmentation_lr[0], betas=(b1, b2))
        self.lr_decay_steps = 0

        if not os.path.isdir(settings.models_weights_dir):
            os.makedirs(settings.models_weights_dir)
        self.net_weights_filepath = os.path.join(settings.models_weights_dir, self.net.get_weights_filename())

    def initialize(self, logger, device: torch.device, device2: torch.device):
        super(Segmentation, self).initialize(logger, device, device2)
        if os.path.isfile(self.net_weights_filepath):
            print(f'Loading model weights for net from {self.net_weights_filepath}')
            try:
                self.net.load_state_dict(torch.load(self.net_weights_filepath, map_location='cpu'), strict=False)
            except Exception as e:
                print(str(e))

        self.net = self.net.to(device)
        self.train_dataset.logger = logger
        # show_gpu_memory_info()
        self.net.zero_grad()

    def train(self, epoch: int):
        self.adjust_lr(epoch)
        self.net.train()
        losses_queue = collections.deque(maxlen=settings.segmentation_log_interval_batches)
        for batch_idx, (data, target) in enumerate(self.train_loader):
            try:
                data, target = data.squeeze(0), target.squeeze(0)
                data, target = data.to(self.device), target.to(self.device)

                output = self.net(data)
                loss = self.criterion(output, target)
                losses_queue.append(loss.item())

                # Update Segmentator weights
                loss.backward()
                self.optimizer.step()
                self.net.zero_grad()

                # Log batch results
                if (batch_idx + 1) % settings.segmentation_log_interval_batches == 0:
                    mean_loss = sum(losses_queue) / len(losses_queue)
                    self.logger.report_scalar(
                        "train", "loss", iteration=(epoch * len(self.train_loader) + batch_idx + 1), value=mean_loss)

                    print(f'Train Epoch: {epoch}/{settings.segmentation_train_epochs} '
                          f'[{(batch_idx + 1)}/{len(self.train_loader.dataset)} '
                          f'({100. * (batch_idx + 1) / len(self.train_loader):.0f}%)]\t'
                          f'Loss: {mean_loss:.6f}')
            except Exception as e:
                print(str(e))
                continue

    def validate(self, epoch: int) -> float:
        self.net.train()  # Set to train or BatchNorm won't work...
        losses_queue = collections.deque()
        with torch.no_grad():
            for batch_idx, (data, target) in enumerate(self.valid_loader):
                data, target = data.squeeze(0), target.squeeze(0)
                data, target = data.to(self.device), target.to(self.device)

                output = self.net(data)
                loss = self.criterion(output, target)
                losses_queue.append(loss.item())

                # Save as image the last batch prediction
                if epoch % settings.segmentation_save_image_interval_epochs == 0:
                    self.save_prediction_image(epoch, batch_idx, data, output, target, 'validation')

                if batch_idx >= 5:
                    break

        mean_loss = sum(losses_queue) / len(losses_queue)
        self.logger.report_scalar("validation", "loss", iteration=epoch, value=mean_loss)
        print(f'Validation Epoch: {epoch}\tLoss: {mean_loss:.6f}')
        return mean_loss

    def save_prediction_image(self, epoch, batch_idx, data: torch.tensor, output, target, series: str):
        images_grid = torchvision.utils.make_grid(data.detach().cpu(),
                                                  nrow=data.size(0)).permute(1, 2, 0).numpy()
        images_grid_near = images_grid[:, :, :3]
        images_grid_mid = images_grid[:, :, 3:]

        masks_grid = torchvision.utils.make_grid(target.detach().cpu(),
                                                 nrow=target.size(0)).permute(1, 2, 0).numpy()
        predicted_masks_grid = torchvision.utils.make_grid(output.detach().cpu(),
                                                           nrow=output.size(0)).permute(1, 2, 0).numpy()

        segmentator_real_grid = np.vstack((images_grid_near, images_grid_mid, masks_grid, predicted_masks_grid))
        segmentator_real_grid = cv2.cvtColor(segmentator_real_grid, cv2.COLOR_HSV2RGB)
        self.logger.report_image("segmentation", f"{series} {batch_idx}", iteration=epoch,
                                 image=segmentator_real_grid)

    def save_full_image(self, epoch, batch_idx, data: torch.tensor, output, target, series: str, num_cols: int):
        images_grid = torchvision.utils.make_grid(data.detach().cpu(),
                                                  nrow=num_cols).permute(1, 2, 0).numpy()
        images_grid_near = images_grid[:, :, :3]

        masks_grid = torchvision.utils.make_grid(target.detach().cpu(),
                                                 nrow=num_cols).permute(1, 2, 0).numpy()
        im_label = images_grid_near.copy()
        im_label[masks_grid > 0.5] = (im_label[masks_grid > 0.5] + 1) / 2

        predicted_masks_grid = torchvision.utils.make_grid(output.detach().cpu(),
                                                           nrow=num_cols).permute(1, 2, 0).numpy()
        im_pred = images_grid_near.copy()
        im_pred[predicted_masks_grid > 0.5] = (im_pred[predicted_masks_grid > 0.5] + 1) / 2

        segmentator_real_grid = np.vstack((im_label, im_pred))
        segmentator_real_grid = cv2.cvtColor(segmentator_real_grid, cv2.COLOR_HSV2RGB)
        self.logger.report_image("segmentation", f"{series} {batch_idx}", iteration=epoch,
                                 image=segmentator_real_grid)

    def can_save_weights(self, epoch) -> bool:
        return epoch % 5 == 0 and epoch > 1

    def save_modules_weights(self):
        torch.save(self.net.state_dict(), self.net_weights_filepath)

    def adjust_lr(self, epoch: int):
        if self.lr_decay_steps < len(settings.segmentation_lr_decay_interval_epochs) and \
                epoch > settings.segmentation_lr_decay_interval_epochs[self.lr_decay_steps]:
            self.lr_decay_steps += 1
            old_lr = self.optimizer.param_groups[0]['lr']
            new_lr = settings.segmentation_lr[
                min(self.lr_decay_steps, len(settings.segmentation_lr) - 1)]
            if new_lr != old_lr:
                self.optimizer.param_groups[0]['lr'] = new_lr
                print(f'Changed learning rate of generator from {old_lr} to {new_lr}')
        self.logger.report_scalar("learning rate", "lr", iteration=epoch, value=self.optimizer.param_groups[0]['lr'])

    def test(self):
        print('Model parameters:', sum(p.numel() for p in self.net.model.parameters()))
        test_dataset = Dataset(
            images_dir=settings.segmentation_test_images_dir,
            masks_dir=settings.segmentation_test_masks_dir,
            channels=3,
            full_image=True
        )
        test_loader = DataLoader(test_dataset,
                                 shuffle=True,
                                 batch_size=1,
                                 num_workers=settings.train_num_workers)
        print('### MODEL ALLOCATED ###')
        show_gpu_memory_info()
        self.net.train()
        start = time.time()
        with torch.no_grad():
            for batch_idx, (data, target, num_cols) in enumerate(test_loader):
                data, target = data.squeeze(0), target.squeeze(0)
                data, target = data.to(self.device), target.to(self.device)
                step1 = time.time()
                print(f'### INPUT DATA LOADED ON GPU [TIME ELAPSED: {step1 - start}s] [INPUT SIZE: {data.size()}] ###')
                show_gpu_memory_info()

                output = self.net(data)
                end = time.time()
                print(f'### OUTPUT GENERATED [TIME ELAPSED: {end - start}s] [OUTPUT SIZE: {output.size()}] ###')
                print('-------------')
                loss = self.criterion(output, target)
                self.logger.report_scalar("test", "loss", iteration=batch_idx, value=loss.item())
                self.logger.report_scalar("test", "max value", iteration=batch_idx,
                                          value=np.max(output.detach().cpu().numpy()))

                self.save_full_image(1, batch_idx, data, output, target, 'test', num_cols[0])
                if batch_idx > 3:
                    break
