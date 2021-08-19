import collections
import os

import numpy as np
import torch
import torchvision
from torch.utils.data import DataLoader
from common.base_model import BaseModel, show_gpu_memory_info
from projects.expert_neuron.dataset import Dataset
from projects.expert_neuron.modules.completer import Completer

from settings import settings


class ExpertNeuron(BaseModel):
    def __init__(self, **_):
        super(ExpertNeuron, self).__init__()
        self.expert_one = Completer(in_channels=24, out_channels=3, input_size=settings.coordmap_input_size,
                                    output_size=settings.coordmap_input_size, features_map=settings.coordmap_features_map)
        self.expert_two = Completer(in_channels=6, out_channels=3, input_size=settings.coordmap_input_size,
                                    output_size=settings.coordmap_input_size, features_map=settings.coordmap_features_map)
        self.train_epochs = settings.segmentation_train_epochs
        self.train_dataset = Dataset(
            master_dir=settings.coordmap_train_images_dir,
            image_size=settings.coordmap_input_size,
            len_dataset=-1,
            preprocessing=self.expert_one.get_preprocessing()
        )
        self.valid_dataset = Dataset(
            master_dir=settings.coordmap_valid_images_dir,
            image_size=settings.coordmap_input_size,
            len_dataset=settings.valid_batch_size * 2,
            preprocessing=self.expert_one.get_preprocessing()
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
        b1, b2 = .5, .999
        self.optimizer_one = torch.optim.Adam(self.expert_one.parameters(), lr=settings.segmentation_lr[0],
                                              betas=(b1, b2))
        self.optimizer_two = torch.optim.Adam(self.expert_two.parameters(), lr=settings.segmentation_lr[0],
                                              betas=(b1, b2))

        self.lr_decay_steps = 0

        if not os.path.isdir(settings.models_weights_dir):
            os.makedirs(settings.models_weights_dir)
        self.expert_one_weights_filepath = os.path.join(settings.models_weights_dir,
                                                        self.expert_one.get_weights_filename())
        self.expert_two_weights_filepath = os.path.join(settings.models_weights_dir,
                                                        self.expert_two.get_weights_filename())

    def initialize(self, logger, device: torch.device, device2: torch.device):
        super(ExpertNeuron, self).initialize(logger, device, device2)
        if os.path.isfile(self.expert_one_weights_filepath):
            print(f'Loading model weights for net from {self.expert_one_weights_filepath}')
            try:
                self.expert_one.load_state_dict(torch.load(self.expert_one_weights_filepath, map_location='cpu'),
                                                strict=False)
            except Exception as e:
                print(str(e))
        if os.path.isfile(self.expert_two_weights_filepath):
            print(f'Loading model weights for net from {self.expert_two_weights_filepath}')
            try:
                self.expert_two.load_state_dict(torch.load(self.expert_two_weights_filepath, map_location='cpu'),
                                                strict=False)
            except Exception as e:
                print(str(e))

        self.expert_one = self.expert_one.to(device)
        self.expert_two = self.expert_two.to(device2)
        # show_gpu_memory_info()
        self.expert_one.zero_grad()
        self.expert_two.zero_grad()

    def train(self, epoch: int):
        self.adjust_lr(epoch)
        self.expert_one.train()
        self.expert_two.train()
        losses_queue = collections.deque(maxlen=settings.segmentation_log_interval_batches)
        for batch_idx, (data_one, data_two, target) in enumerate(self.train_loader):
            data_one, data_two, = data_one.to(self.device), data_two.to(self.device2)

            output_one = self.expert_one(data_one)
            output_two = self.expert_two(data_two)
            loss_dissimilarity_one = self.criterion(output_one, output_two.detach().to(self.device))
            # loss_dissimilarity_two = self.criterion(output_two, output_one.detach())

            loss_target_one = self.criterion(output_one, target.to(self.device))
            loss_target_two = self.criterion(output_two, target.to(self.device2))

            loss_one = loss_dissimilarity_one + loss_target_one
            loss_two = loss_target_two  # loss_dissimilarity_two

            loss = loss_one.detach().cpu() + loss_two.detach().cpu()
            losses_queue.append(loss.item())
            loss_one.backward()
            loss_two.backward()
            # loss_target_two.backward()
            self.optimizer_one.step()
            self.optimizer_two.step()
            self.expert_one.zero_grad()
            self.expert_two.zero_grad()

            # Log batch results
            if (batch_idx + 1) % settings.segmentation_log_interval_batches == 0:
                mean_loss = sum(losses_queue) / len(losses_queue)
                self.logger.report_scalar(
                    "train", "loss", iteration=(epoch * len(self.train_loader) + batch_idx + 1), value=mean_loss)

                print(f'Train Epoch: {epoch}/{settings.segmentation_train_epochs} '
                      f'[{(batch_idx + 1) * settings.train_batch_size}/{len(self.train_loader.dataset)} '
                      f'({100. * (batch_idx + 1) / len(self.train_loader):.0f}%)]\t'
                      f'Loss: {mean_loss:.6f}')

    def validate(self, epoch: int) -> float:
        self.expert_one.eval()
        self.expert_two.eval()
        losses_queue = collections.deque()
        with torch.no_grad():
            for batch_idx, (data_one, data_two, target) in enumerate(self.valid_loader):
                data_one, data_two = data_one.to(self.device), data_two.to(self.device2)

                output_one = self.expert_one(data_one)
                output_two = self.expert_two(data_two)
                loss_dissimilarity_one = self.criterion(output_one, output_two.detach().to(self.device))
                # loss_dissimilarity_two = self.criterion(output_two, output_one.detach())

                loss_target_one = self.criterion(output_one, target.to(self.device))
                loss_target_two = self.criterion(output_two, target.to(self.device2))

                loss_one = loss_dissimilarity_one + loss_target_one
                loss_two = loss_target_two # + loss_dissimilarity_two

                loss = loss_one.detach().cpu() + loss_two.detach().cpu()
                losses_queue.append(loss.item())

                # Save as image the last batch prediction
                if epoch % settings.segmentation_save_image_interval_epochs == 0:
                    self.save_prediction_image(epoch, batch_idx, data_one, data_two, output_one, output_two, target,
                                               'validation')

        mean_loss = sum(losses_queue) / len(losses_queue)
        self.logger.report_scalar("validation", "loss", iteration=epoch, value=mean_loss)
        print(f'Validation Epoch: {epoch}\tLoss: {mean_loss:.6f}')
        return mean_loss

    def save_prediction_image(self, epoch, batch_idx, data_one, data_two, output_one, output_two, target, series: str):
        data_one_grid = torchvision.utils.make_grid(data_one.detach().cpu()[:, :3, :, :],
                                                    nrow=data_one.size(0)).permute(1, 2, 0).numpy()
        data_one_grid2 = torchvision.utils.make_grid(data_one.detach().cpu()[:, 3:6, :, :],
                                                     nrow=data_one.size(0)).permute(1, 2, 0).numpy()
        data_two_grid = torchvision.utils.make_grid(data_two.detach().cpu()[:, :3, :, :],
                                                    nrow=data_two.size(0)).permute(1, 2, 0).numpy()
        data_two_grid2 = torchvision.utils.make_grid(data_two.detach().cpu()[:, 3:, :, :],
                                                     nrow=data_two.size(0)).permute(1, 2, 0).numpy()
        data_one_out_grid = torchvision.utils.make_grid(output_one.detach().cpu(),
                                                        nrow=output_one.size(0)).permute(1, 2, 0).numpy()
        data_two_out_grid = torchvision.utils.make_grid(output_two.detach().cpu(),
                                                        nrow=output_two.size(0)).permute(1, 2, 0).numpy()
        target_grid = torchvision.utils.make_grid(target.detach().cpu(),
                                                  nrow=target.size(0)).permute(1, 2, 0).numpy()
        segmentator_real_grid = np.vstack(
            (data_one_grid, data_one_grid2, data_two_grid, data_two_grid2, data_one_out_grid, data_two_out_grid,
             target_grid))
        self.logger.report_image("segmentation", f"{series} {batch_idx}", iteration=epoch,
                                 image=segmentator_real_grid)

    def can_save_weights(self, epoch) -> bool:
        return epoch % 5 == 0 and epoch > 1

    def save_modules_weights(self):
        torch.save(self.expert_one.state_dict(), self.expert_one_weights_filepath)
        torch.save(self.expert_two.state_dict(), self.expert_two_weights_filepath)

    def adjust_lr(self, epoch: int):
        if self.lr_decay_steps < len(settings.segmentation_lr_decay_interval_epochs) and \
                epoch > settings.segmentation_lr_decay_interval_epochs[self.lr_decay_steps]:
            self.lr_decay_steps += 1
            old_lr = self.optimizer_one.param_groups[0]['lr']
            new_lr = settings.segmentation_lr[
                min(self.lr_decay_steps, len(settings.segmentation_lr) - 1)]
            if new_lr != old_lr:
                self.optimizer_one.param_groups[0]['lr'] = new_lr
                self.optimizer_two.param_groups[0]['lr'] = new_lr
                print(f'Changed learning rate of generator from {old_lr} to {new_lr}')
        self.logger.report_scalar("learning rate", "lr", iteration=epoch,
                                  value=self.optimizer_one.param_groups[0]['lr'])

    def test(self):
        self.expert_one.eval()
        self.expert_two.eval()

        test_dataset = Dataset(
            master_dir=settings.coordmap_valid_images_dir,
            image_size=settings.coordmap_input_size,
            len_dataset=-1,
            preprocessing=self.expert_one.get_preprocessing()
        )
        test_loader = DataLoader(test_dataset,
                                 shuffle=False,
                                 batch_size=1,
                                 num_workers=2)

        original_image = (self.valid_dataset.master / 255).astype(np.float32)
        h, w, c = original_image.shape
        final_image_one = np.zeros((h, w, c), dtype=np.float32)
        final_image_two = np.zeros((h, w, c), dtype=np.float32)
        with torch.no_grad():
            for batch_idx, (data_one, data_two, target) in enumerate(test_loader):
                data_one, data_two = data_one.to(self.device), data_two.to(self.device)

                output_one = self.expert_one(data_one)
                output_image = output_one.detach().cpu()[0, ...].permute(1, 2, 0).numpy()
                final_image_one[
                test_dataset.yy[batch_idx] + int(test_dataset.image_size):test_dataset.yy[
                                                                              batch_idx] + int(
                    self.valid_dataset.image_size * 2),
                test_dataset.xx[batch_idx] + int(test_dataset.image_size):test_dataset.xx[
                                                                              batch_idx] + int(
                    self.valid_dataset.image_size * 2),
                :] = output_image

                output_two = self.expert_two(data_two)
                output_image = output_two.detach().cpu()[0, ...].permute(1, 2, 0).numpy()
                final_image_two[
                test_dataset.yy[batch_idx] + int(test_dataset.image_size):test_dataset.yy[
                                                                              batch_idx] + int(
                    self.valid_dataset.image_size * 2),
                test_dataset.xx[batch_idx] + int(test_dataset.image_size):test_dataset.xx[
                                                                              batch_idx] + int(
                    self.valid_dataset.image_size * 2),
                :] = output_image
                if batch_idx % 50 == 0:
                    print(batch_idx, '/', len(test_loader))

        self.logger.report_image("test", "reconstructed image 1", iteration=1, image=final_image_one)
        self.logger.report_image("test", "reconstructed image 2", iteration=1, image=final_image_two)
        self.logger.report_image("test", "reconstructed image diff", iteration=1,
                                 image=np.abs(final_image_one - final_image_two))
