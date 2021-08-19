import collections
import os
import random

import numpy as np
import torch
import torchvision
from torch.utils.data import DataLoader
from common.augmentation import base_augmentation
from common.base_model import BaseModel, show_gpu_memory_info
from projects.triplet.dataset import Dataset
from projects.triplet.modules.discriminator import Discriminator
from projects.triplet.modules.generator import Generator
from projects.triplet.modules.segmentator import Segmentator
from settings import settings


def get_generator_fake_input(fake_images, real_good_images, random_masks):
    if settings.triplet_use_whole_generated_image:
        return fake_images
    fake_images_defects_mask = fake_images * random_masks
    real_images_anti_mask = real_good_images * (1 - random_masks)
    del real_good_images
    del random_masks
    return fake_images_defects_mask + real_images_anti_mask


class Triplet(BaseModel):
    def __init__(self, **_):
        super(Triplet, self).__init__()
        self.netG = Generator(input_size=settings.triplet_input_size,
                              features_map=settings.triplet_generator_features_map)
        self.netD = Discriminator(input_size=settings.triplet_input_size)
        self.netS = Segmentator()
        self.train_epochs = settings.triplet_train_epochs
        self.train_dataset = Dataset(
            images_dir=settings.triplet_train_images_dir,
            masks_dir=settings.triplet_train_masks_dir,
            classes=[255],
            augmentation=base_augmentation(size=(settings.triplet_input_size, settings.triplet_input_size)),
            preprocessing=self.netG.get_preprocessing()
        )
        self.good_dataset = Dataset(
            images_dir=settings.triplet_good_images_dir,
            masks_dir=settings.triplet_random_masks_dir,
            classes=[255],
            augmentation=base_augmentation(size=(settings.triplet_input_size, settings.triplet_input_size)),
            preprocessing=self.netG.get_preprocessing(),
            random_masks=True
        )
        self.valid_dataset = Dataset(
            images_dir=settings.triplet_valid_images_dir,
            masks_dir=settings.triplet_valid_masks_dir,
            classes=[255],
            augmentation=base_augmentation(size=(settings.triplet_input_size, settings.triplet_input_size)),
            preprocessing=self.netG.get_preprocessing()
        )
        self.train_loader = DataLoader(self.train_dataset,
                                       shuffle=True,
                                       batch_size=settings.train_batch_size,
                                       num_workers=settings.train_num_workers)
        self.good_loader = DataLoader(self.good_dataset,
                                      shuffle=False,
                                      batch_size=settings.train_batch_size,
                                      num_workers=settings.train_num_workers)
        self.valid_loader = DataLoader(self.valid_dataset,
                                       shuffle=False,
                                       batch_size=settings.valid_batch_size,
                                       num_workers=settings.train_num_workers)
        self.criterion = torch.nn.BCELoss()
        b1, b2 = .5, .999
        self.optimizerG = torch.optim.Adam(self.netG.parameters(), lr=settings.triplet_generator_lr[0], betas=(b1, b2))
        self.optimizerD = torch.optim.Adam(self.netD.parameters(), lr=settings.triplet_discriminator_lr[0],
                                           betas=(b1, b2))
        self.optimizerS = torch.optim.Adam(self.netS.parameters(), lr=settings.triplet_segmentator_lr[0],
                                           betas=(b1, b2))
        self.lr_decay_stepsG, self.lr_decay_stepsD, self.lr_decay_stepsS = 0, 0, 0

        if not os.path.isdir(settings.models_weights_dir):
            os.makedirs(settings.models_weights_dir)
        self.netG_weights_filepath = os.path.join(settings.models_weights_dir, self.netG.get_weights_filename())
        self.netD_weights_filepath = os.path.join(settings.models_weights_dir, self.netD.get_weights_filename())
        self.netS_weights_filepath = os.path.join(settings.models_weights_dir, self.netS.get_weights_filename())

    def initialize(self, logger, device: torch.device, device2: torch.device):
        super(Triplet, self).initialize(logger, device, device2)
        if os.path.isfile(self.netG_weights_filepath):
            print(f'Loading model weights for netG from {self.netG_weights_filepath}')
            try:
                self.netG.load_state_dict(torch.load(self.netG_weights_filepath, map_location='cpu'), strict=False)
            except Exception as e:
                print(str(e))
        if os.path.isfile(self.netD_weights_filepath):
            print(f'Loading model weights for netD from {self.netD_weights_filepath}')
            try:
                self.netD.load_state_dict(torch.load(self.netD_weights_filepath, map_location='cpu'), strict=False)
            except Exception as e:
                print(str(e))
        if os.path.isfile(self.netS_weights_filepath):
            print(f'Loading model weights for netS from {self.netS_weights_filepath}')
            try:
                self.netS.load_state_dict(torch.load(self.netS_weights_filepath, map_location='cpu'), strict=False)
            except Exception as e:
                print(str(e))

        self.netG = self.netG.to(device)
        # show_gpu_memory_info()
        self.netD = self.netD.to(device)
        self.netS = self.netS.to(device2)
        self.netG.zero_grad(), self.netD.zero_grad(), self.netS.zero_grad()

    def train(self, epoch: int):
        self.adjust_lr(epoch)
        self.netG.train()
        self.netD.train()
        self.netS.train()
        losses_queue_GG = collections.deque(maxlen=settings.triplet_log_interval_batches)
        losses_queue_GD = collections.deque(maxlen=settings.triplet_log_interval_batches)
        losses_queue_GS = collections.deque(maxlen=settings.triplet_log_interval_batches)
        mean_pred_queue_Dreal = collections.deque(maxlen=settings.triplet_log_interval_batches)
        mean_pred_queue_Dfake = collections.deque(maxlen=settings.triplet_log_interval_batches)
        losses_queue_Dreal = collections.deque(maxlen=settings.triplet_log_interval_batches)
        losses_queue_Dfake = collections.deque(maxlen=settings.triplet_log_interval_batches)
        losses_queue_Sreal = collections.deque(maxlen=settings.triplet_log_interval_batches)
        losses_queue_Sfake = collections.deque(maxlen=settings.triplet_log_interval_batches)
        for batch_idx, (data, target) in enumerate(self.train_loader):
            real_images = data.to(self.device)
            b_size = real_images.size(0)

            # Generate real label for Discriminator
            real_label = random.uniform(0.95, 1.)
            real_labelD = torch.full((b_size,), real_label, dtype=torch.float, device=settings.device)

            # Train Discriminator with all-real batch
            for p in self.netD.parameters():
                p.requires_grad = True

            outputDreal = self.netD(real_images)
            del real_images
            mean_pred_queue_Dreal.append(torch.mean(outputDreal).item())
            lossD_real = self.criterion(outputDreal.view(b_size, ), real_labelD)
            losses_queue_Dreal.append(lossD_real.item())

            # Create fake images with Generator
            good_data, random_data = next(iter(self.good_loader))
            real_good_images, random_masks = good_data.to(self.device), random_data.to(self.device)

            good_b_size, _, h, w = random_masks.shape
            z = torch.rand(h, w).to(self.device)
            z = z * random_masks
            input_imagesG = torch.hstack((real_good_images, z))

            fake_images = self.netG(input_imagesG)

            # Train Generator on Generator
            # fake_images_anti_mask = fake_images * (1 - random_masks)
            # real_images_anti_mask = real_good_images * (1 - random_masks)
            # lossGG = self.criterion(fake_images_anti_mask, real_images_anti_mask)
            # losses_queue_GG.append(lossGG.item())

            # lossGG.backward()
            # del lossGG
            # self.optimizerG.step()
            # self.netG.zero_grad()

            # Generate fake label for Discriminator
            fake_label = random.uniform(0., 0.05)
            fake_labelD = torch.full((good_b_size,), fake_label, dtype=torch.float, device=self.device)

            # Train Discriminator with all-fake batch
            outputDfake = self.netD(fake_images.detach())
            mean_pred_queue_Dfake.append(torch.mean(outputDfake).item())
            lossD_fake = self.criterion(outputDfake.view(good_b_size, ), fake_labelD)
            losses_queue_Dfake.append(lossD_fake.item())
            lossD = lossD_fake + lossD_real

            # Update Discriminator weights
            lossD.backward()
            del lossD
            self.optimizerD.step()
            self.netD.zero_grad()

            # Train Generator on Discriminator
            real_labelD = torch.full((good_b_size,), 1., dtype=torch.float, device=self.device)

            for p in self.netD.parameters():
                p.requires_grad = False  # to avoid computation on Discriminator

            outputGD = self.netD(fake_images).view(-1)
            lossGD = self.criterion(outputGD, real_labelD)
            losses_queue_GD.append(lossGD.item())
            lossG = lossGD

            # Train Segmentator with all-real batch
            real_images, real_masks = data.to(self.device2), target.to(self.device2)
            outputS = self.netS(real_images)
            lossS_real = self.criterion(outputS, real_masks)
            del real_masks
            losses_queue_Sreal.append(lossS_real.item())
            lossS = lossS_real

            # Train Generator on Segmentator
            if settings.triplet_gd_good_score_ranges[0] < torch.mean(outputDfake).item() < \
                    settings.triplet_gd_good_score_ranges[1]:
                fake_images, random_masks = fake_images.to(self.device2), random_data.to(self.device2)
                for p in self.netS.parameters():
                    p.requires_grad = False  # to avoid computation on Segmentator

                outputGS = self.netS(fake_images)
                lossGS = self.criterion(outputGS, random_masks)
                losses_queue_GS.append(lossGS.item())
                lossG = lossGD + lossGS.to(self.device)

                for p in self.netS.parameters():
                    p.requires_grad = True

                # Train Segmentator with all-fake batch
                if settings.triplet_gs_good_loss_ranges[0] < lossGS.item() < settings.triplet_gs_good_loss_ranges[1]:
                    outputS = self.netS(fake_images.detach())
                    lossS_fake = self.criterion(outputS, random_masks)
                    losses_queue_Sreal.append(lossS_fake.item())
                    lossS = lossS + lossS_fake
                del fake_images
                del random_masks

            # Update Generator weights
            lossG.backward()
            del lossG
            self.optimizerG.step()
            self.netG.zero_grad()

            # Update Segmentator weights
            lossS.backward()
            del lossS
            self.optimizerS.step()
            self.netS.zero_grad()

            # Log batch results
            if (batch_idx + 1) % settings.triplet_log_interval_batches == 0:
                mean_loss_GG = sum(losses_queue_GG) / len(losses_queue_GG) if len(
                    losses_queue_GG) > 0 else -1
                self.logger.report_scalar(
                    "train", "loss GG", iteration=(epoch * len(self.train_loader) + batch_idx + 1), value=mean_loss_GG)
                mean_loss_GD = sum(losses_queue_GD) / len(losses_queue_GD)
                self.logger.report_scalar(
                    "train", "loss GD", iteration=(epoch * len(self.train_loader) + batch_idx + 1), value=mean_loss_GD)
                mean_loss_GS = sum(losses_queue_GS) / len(losses_queue_GS) if len(
                    losses_queue_GS) > 0 else -1
                self.logger.report_scalar(
                    "train", "loss GS", iteration=(epoch * len(self.train_loader) + batch_idx + 1), value=mean_loss_GS)
                mean_pred_Dreal = sum(mean_pred_queue_Dreal) / len(mean_pred_queue_Dreal)
                self.logger.report_scalar(
                    "discriminator", "pred Dreal", iteration=(epoch * len(self.train_loader) + batch_idx + 1),
                    value=mean_pred_Dreal)
                mean_pred_Dfake = sum(mean_pred_queue_Dfake) / len(mean_pred_queue_Dfake)
                self.logger.report_scalar(
                    "discriminator", "pred Dfake", iteration=(epoch * len(self.train_loader) + batch_idx + 1),
                    value=mean_pred_Dfake)
                mean_loss_Dreal = sum(losses_queue_Dreal) / len(losses_queue_Dreal)
                self.logger.report_scalar(
                    "train", "loss Dreal", iteration=(epoch * len(self.train_loader) + batch_idx + 1),
                    value=mean_loss_Dreal)
                mean_loss_Dfake = sum(losses_queue_Dfake) / len(losses_queue_Dfake)
                self.logger.report_scalar(
                    "train", "loss Dfake", iteration=(epoch * len(self.train_loader) + batch_idx + 1),
                    value=mean_loss_Dfake)
                mean_loss_Sreal = sum(losses_queue_Sreal) / len(losses_queue_Sreal)
                self.logger.report_scalar(
                    "train", "loss Sreal", iteration=(epoch * len(self.train_loader) + batch_idx + 1),
                    value=mean_loss_Sreal)
                mean_loss_Sfake = sum(losses_queue_Sfake) / len(losses_queue_Sfake) if len(
                    losses_queue_Sfake) > 0 else -1
                self.logger.report_scalar(
                    "train", "loss Sfake", iteration=(epoch * len(self.train_loader) + batch_idx + 1),
                    value=mean_loss_Sfake)

                print(f'Train Epoch: {epoch}/{settings.ae_svm_train_epochs} '
                      f'[{(batch_idx + 1) * len(data)}/{len(self.train_loader.dataset)} '
                      f'({100. * (batch_idx + 1) / len(self.train_loader):.0f}%)]\t'
                      f'Loss GG: {mean_loss_GG:.6f}\tLoss GD: {mean_loss_GD:.6f}\tLoss GS: {mean_loss_GS:.6f}\t'
                      f'Pred Dreal: {mean_pred_Dreal:.6f}\tPred Dfake: {mean_pred_Dfake:.6f}\t'
                      f'Loss Dreal: {mean_loss_Dreal:.6f}\tLoss Dfake: {mean_loss_Dfake:.6f}\t'
                      f'Loss Sreal: {mean_loss_Sreal:.6f}\tLoss Sfake: {mean_loss_Sfake:.6f}')

    def validate(self, epoch: int) -> float:
        self.netG.eval()
        self.netD.eval()
        self.netS.eval()
        losses_queue_Sreal = collections.deque()
        with torch.no_grad():
            for batch_idx, (data, target) in enumerate(self.valid_loader):
                real_images, real_masks = data.to(self.device), target.to(self.device)
                real_good_images, random_masks = next(iter(self.good_loader))
                real_good_images, random_masks = real_good_images.to(settings.device), random_masks.to(settings.device)

                # Create fake test_images with Generator
                b, c, h, w = random_masks.shape
                z = torch.rand(h, w).to(self.device)
                z = z * random_masks
                input_images = torch.hstack((real_good_images, z))

                fake_images = self.netG(input_images)

                # Test Discriminator on Generator
                # outD = self.netD(fake_images.detach())

                # Test Segmentator on good and defected test_images
                outS_good = self.netS(real_good_images.to(self.device2))
                outS_defected = self.netS(real_images.to(self.device2))

                lossS_real = self.criterion(outS_defected, real_masks.to(self.device2))
                losses_queue_Sreal.append(lossS_real.item())

                # Save as image the last batch prediction
                if epoch % settings.triplet_save_image_interval_epochs == 0 and batch_idx == 2:
                    real_good_images_grid = torchvision.utils.make_grid(real_good_images.detach().cpu(),
                                                                        nrow=real_good_images.size(0)).permute(1, 2,
                                                                                                               0).numpy()

                    random_masks_grid = torchvision.utils.make_grid(random_masks.detach().cpu(),
                                                                    nrow=random_masks.size(0)).permute(1, 2, 0).numpy()

                    fake_images_grid = torchvision.utils.make_grid(fake_images.detach().cpu(),
                                                                   nrow=fake_images.size(0)).permute(1, 2, 0).numpy()

                    generator_grid = np.vstack((real_good_images_grid, random_masks_grid, fake_images_grid))
                    self.logger.report_image("generator", "fake images", iteration=epoch, image=generator_grid)

                    if not settings.triplet_use_whole_generated_image:
                        fake_images_modified = get_generator_fake_input(fake_images, real_good_images, random_masks)
                        fake_images_modified_grid = torchvision.utils.make_grid(fake_images_modified.detach().cpu(),
                                                                                nrow=fake_images_modified.size(
                                                                                    0)).permute(1, 2, 0).numpy()
                        self.logger.report_image("generator", "fake images modified", iteration=epoch,
                                                 image=fake_images_modified_grid)

                    real_images_grid = torchvision.utils.make_grid(real_images.detach().cpu(),
                                                                   nrow=real_images.size(0)).permute(1, 2, 0).numpy()
                    real_masks_grid = torchvision.utils.make_grid(real_masks.detach().cpu(),
                                                                  nrow=real_masks.size(0)).permute(1, 2, 0).numpy()
                    predicted_masks_grid = torchvision.utils.make_grid(outS_defected.detach().cpu(),
                                                                       nrow=outS_defected.size(0)).permute(1, 2,
                                                                                                           0).numpy()
                    segmentator_real_grid = np.vstack((real_images_grid, real_masks_grid, predicted_masks_grid))
                    self.logger.report_image("segmentator", "real images", iteration=epoch, image=segmentator_real_grid)

                    predicted_good_masks_grid = torchvision.utils.make_grid(outS_good.detach().cpu(),
                                                                            nrow=outS_good.size(0)).permute(1, 2,
                                                                                                            0).numpy()
                    segmentator_real_good_grid = np.vstack((real_good_images_grid, predicted_good_masks_grid))
                    self.logger.report_image("segmentator", "real good images", iteration=epoch,
                                             image=segmentator_real_good_grid)

        mean_loss_Sreal = sum(losses_queue_Sreal) / len(losses_queue_Sreal)
        self.logger.report_scalar("test", "Loss Sreal", iteration=epoch, value=mean_loss_Sreal)
        print(f'Test Epoch: {epoch}\tLoss Sreal: {mean_loss_Sreal:.6f}')
        return mean_loss_Sreal

    def can_save_weights(self, epoch) -> bool:
        return epoch % 5 == 0 and epoch > 1

    def save_modules_weights(self):
        torch.save(self.netG.state_dict(), self.netG_weights_filepath)
        torch.save(self.netD.state_dict(), self.netD_weights_filepath)
        torch.save(self.netS.state_dict(), self.netS_weights_filepath)

    def adjust_lr(self, epoch: int):
        if self.lr_decay_stepsG < len(settings.triplet_generator_lr_decay_interval_epochs) and \
                epoch > settings.triplet_generator_lr_decay_interval_epochs[self.lr_decay_stepsG]:
            self.lr_decay_stepsG += 1
            old_lrG = self.optimizerG.param_groups[0]['lr']
            new_lrG = settings.triplet_generator_lr[
                min(self.lr_decay_stepsG, len(settings.triplet_generator_lr) - 1)]
            if new_lrG != old_lrG:
                self.optimizerG.param_groups[0]['lr'] = new_lrG
                print(f'Changed learning rate of generator from {old_lrG} to {new_lrG}')

        if self.lr_decay_stepsD < len(settings.triplet_discriminator_lr_decay_interval_epochs) and \
                epoch > settings.triplet_discriminator_lr_decay_interval_epochs[self.lr_decay_stepsD]:
            self.lr_decay_stepsD += 1
            old_lrD = self.optimizerD.param_groups[0]['lr']
            new_lrD = settings.triplet_discriminator_lr[
                min(self.lr_decay_stepsD, len(settings.triplet_discriminator_lr) - 1)]
            if new_lrD != old_lrD:
                self.optimizerD.param_groups[0]['lr'] = new_lrD
                print(f'Changed learning rate of discriminator from {old_lrD} to {new_lrD}')

        if self.lr_decay_stepsS < len(settings.triplet_segmentator_lr_decay_interval_epochs) and \
                epoch > settings.triplet_segmentator_lr_decay_interval_epochs[self.lr_decay_stepsS]:
            self.lr_decay_stepsS += 1
            old_lrS = self.optimizerS.param_groups[0]['lr']
            new_lrS = settings.triplet_segmentator_lr[
                min(self.lr_decay_stepsS, len(settings.triplet_segmentator_lr) - 1)]
            if new_lrS != old_lrS:
                self.optimizerS.param_groups[0]['lr'] = new_lrS
                print(f'Changed learning rate of segmentator from {old_lrS} to {new_lrS}')

        self.logger.report_scalar("learning rate", "lrG", iteration=epoch, value=self.optimizerG.param_groups[0]['lr'])
        self.logger.report_scalar("learning rate", "lrD", iteration=epoch, value=self.optimizerD.param_groups[0]['lr'])
        self.logger.report_scalar("learning rate", "lrS", iteration=epoch, value=self.optimizerS.param_groups[0]['lr'])
