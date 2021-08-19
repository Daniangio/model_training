import collections
import os
from projects.stdp.dataset import Dataset
from projects.stdp.modules.simplenet import SimpleNet
import time

import cv2
import numpy as np
import torch
import torchvision
from torch.utils.data import DataLoader
from common.augmentation import training_augmentation
from common.base_model import BaseModel, show_gpu_memory_info
from settings import settings
import segmentation_models_pytorch as smp


class STDP(BaseModel):
    def __init__(self, **_):
        super(STDP, self).__init__()
        self.net = SimpleNet(in_neurons=8**2, out_neurons=512)
        self.train_epochs = settings.segmentation_train_epochs
        self.train_dataset = Dataset(
            images_dir=settings.segmentation_train_images_dir,
            input_size=8,
            augmentation=None #training_augmentation(size=None)
        )
        self.valid_dataset = Dataset(
            images_dir=settings.segmentation_valid_images_dir,
            input_size=8,
            augmentation=None
        )
        self.train_loader = DataLoader(self.train_dataset,
                                       shuffle=True,
                                       batch_size=64,
                                       num_workers=settings.train_num_workers)
        self.valid_loader = DataLoader(self.valid_dataset,
                                       shuffle=False,
                                       batch_size=64,
                                       num_workers=settings.train_num_workers)
        self.criterion = smp.utils.losses.DiceLoss()
        b1, b2 = .5, .999
        self.optimizer = torch.optim.Adam(self.net.parameters(), lr=settings.segmentation_lr[0], betas=(b1, b2))
        self.lr_decay_steps = 0

        if not os.path.isdir(settings.models_weights_dir):
            os.makedirs(settings.models_weights_dir)
        self.net_weights_filepath = os.path.join(settings.models_weights_dir, self.net.get_weights_filename())

    def initialize(self, logger, device: torch.device, device2: torch.device):
        super(STDP, self).initialize(logger, device, device2)
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
        for batch_idx, data in enumerate(self.train_loader):
            try:
                data = data.to(self.device).float()

                output = self.net(data)
                self.net.stdp_update()

                #loss = self.criterion(output, target)
                #losses_queue.append(loss.item())

                #loss.backward()
                #self.optimizer.step()
                #self.net.zero_grad()

                # Log batch results
                if (batch_idx + 1) % settings.segmentation_log_interval_batches == 0:
                    #mean_loss = sum(losses_queue) / len(losses_queue)
                    #self.logger.report_scalar(
                    #    "train", "loss", iteration=(epoch * len(self.train_loader) + batch_idx + 1), value=mean_loss)

                    print(f'Train Epoch: {epoch}/{settings.segmentation_train_epochs} '
                          f'[{(batch_idx + 1) * 64}/{len(self.train_loader.dataset)} '
                          f'({100. * (batch_idx + 1) / len(self.train_loader):.0f}%)]\t')
            except Exception as e:
                print(str(e))
                continue

    def validate(self, epoch: int) -> float:
        return 0
        self.net.train()  # Set to train or BatchNorm won't work...
        losses_queue = collections.deque()
        with torch.no_grad():
            for batch_idx, data in enumerate(self.valid_loader):
                data = data.to(self.device).float()

                output = self.net(data)
                if batch_idx >= 5:
                    break
        return 0

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
        #self.logger.report_scalar("learning rate", "lr", iteration=epoch, value=self.optimizer.param_groups[0]['lr'])

    def test(self):
        self.net.visualize_weights()