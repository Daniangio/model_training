import collections
import os
import random

import numpy as np
import torch
import torchvision
from torch.utils.data import DataLoader
from common.augmentation import base_augmentation
from common.base_model import BaseModel
from projects.dcgan.dataset import Dataset
from projects.dcgan.modules.discriminator import Discriminator
from projects.dcgan.modules.encoder import Encoder
from projects.dcgan.modules.generator import Generator
from settings import settings


class DCGAN(BaseModel):
    def __init__(self, **_):
        super(DCGAN, self).__init__()
        b1, b2 = .5, .999
        self.netG = Generator(input_vector_size=settings.dcgan_latent_vector_size,
                              output_size=settings.get_dcgan_image_size(), feature_maps=32)
        self.optimizerG = torch.optim.Adam(self.netG.parameters(), lr=settings.dcgan_generator_lr[0], betas=(b1, b2))
        self.netG_weights_filepath = os.path.join(settings.models_weights_dir, self.netG.get_weights_filename())

        if settings.dcgan_training_phase == 1:
            self.netD = Discriminator(input_size=settings.get_dcgan_image_size(), feature_maps=32)
            self.optimizerD = torch.optim.Adam(self.netD.parameters(), lr=settings.dcgan_discriminator_lr[0],
                                               betas=(b1, b2))
            self.netD_weights_filepath = os.path.join(settings.models_weights_dir, self.netD.get_weights_filename())

        if settings.dcgan_training_phase == 2:
            self.netE = Encoder(input_size=settings.get_dcgan_image_size(),
                                output_size=settings.dcgan_latent_vector_size, feature_maps=32)
            self.optimizerE = torch.optim.Adam(self.netE.parameters(), lr=settings.dcgan_encoder_lr[0], betas=(b1, b2))
            self.netE_weights_filepath = os.path.join(settings.models_weights_dir, self.netE.get_weights_filename())

        self.train_epochs = settings.dcgan_train_epochs
        self.train_dataset = Dataset(
            images_dir=settings.dcgan_train_images_dir,
            augmentation=base_augmentation(size=settings.get_dcgan_image_size()),
            preprocessing=self.netG.get_preprocessing()
        )
        self.valid_dataset = Dataset(
            images_dir=settings.dcgan_valid_images_dir,
            augmentation=base_augmentation(size=settings.get_dcgan_image_size()),
            preprocessing=self.netG.get_preprocessing()
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
        self.criterion2 = torch.nn.MSELoss()
        self.lr_decay_stepsG, self.lr_decay_stepsD, self.lr_decay_stepsE = 0, 0, 0

        if not os.path.isdir(settings.models_weights_dir):
            os.makedirs(settings.models_weights_dir)

    def initialize(self, logger, device: torch.device, device2: torch.device):
        super(DCGAN, self).initialize(logger, device, device2)
        if os.path.isfile(self.netG_weights_filepath):
            print(f'Loading model weights for netG from {self.netG_weights_filepath}')
            try:
                self.netG.load_state_dict(torch.load(self.netG_weights_filepath, map_location='cpu'), strict=False)
            except Exception as e:
                print(str(e))
        if settings.dcgan_training_phase == 1 and os.path.isfile(self.netD_weights_filepath):
            print(f'Loading model weights for netD from {self.netD_weights_filepath}')
            try:
                self.netD.load_state_dict(torch.load(self.netD_weights_filepath, map_location='cpu'), strict=False)
            except Exception as e:
                print(str(e))
        if settings.dcgan_training_phase == 2 and os.path.isfile(self.netE_weights_filepath):
            print(f'Loading model weights for netE from {self.netE_weights_filepath}')
            try:
                self.netE.load_state_dict(torch.load(self.netE_weights_filepath, map_location='cpu'), strict=False)
            except Exception as e:
                print(str(e))
        self.netG = self.netG.to(device)
        self.netG.zero_grad()
        if settings.dcgan_training_phase == 1:
            self.netD = self.netD.to(device2)
            self.netD.zero_grad()
        if settings.dcgan_training_phase == 2:
            self.netE = self.netE.to(device2)
            self.netE.zero_grad()

    def train(self, epoch: int):
        self.adjust_lr(epoch)
        if settings.dcgan_training_phase == 1:
            self.train_phase_1(epoch)
        elif settings.dcgan_training_phase == 2:
            self.train_phase_2(epoch)

    def train_phase_1(self, epoch):
        self.netG.train()
        self.netD.train()
        losses_queue_G = collections.deque(maxlen=settings.dcgan_log_interval_batches)
        mean_pred_queue_Dreal = collections.deque(maxlen=settings.dcgan_log_interval_batches)
        mean_pred_queue_Dfake = collections.deque(maxlen=settings.dcgan_log_interval_batches)
        losses_queue_Dreal = collections.deque(maxlen=settings.dcgan_log_interval_batches)
        losses_queue_Dfake = collections.deque(maxlen=settings.dcgan_log_interval_batches)
        for batch_idx, data in enumerate(self.train_loader):
            real_images = data.to(self.device2)
            b_size = real_images.size(0)
            real_label = random.uniform(0.95, 1.)
            real_labelD = torch.full((b_size,), real_label, dtype=torch.float, device=self.device2)

            # Train Discriminator with all-real batch
            for p in self.netD.parameters():
                p.requires_grad = True

            outputDreal = self.netD(real_images)
            del real_images
            mean_pred_queue_Dreal.append(torch.mean(outputDreal).item())
            lossD_real = self.criterion(outputDreal.view(b_size, ), real_labelD)
            losses_queue_Dreal.append(lossD_real.item())

            # Create fake test_images with Generator
            # Generate batch of latent vectors
            noise = torch.randn(b_size, settings.dcgan_latent_vector_size, 1, 1, device=self.device)

            fake_images = self.netG(noise).to(self.device2)

            # Train Discriminator with all-fake batch
            fake_label = random.uniform(0., 0.05)
            fake_labelD = torch.full((b_size,), fake_label, dtype=torch.float, device=self.device2)

            outputDfake = self.netD(fake_images.detach())
            mean_pred_queue_Dfake.append(torch.mean(outputDfake).item())
            lossD_fake = self.criterion(outputDfake.view(b_size, ), fake_labelD)
            losses_queue_Dfake.append(lossD_fake.item())
            lossD = lossD_fake + lossD_real

            lossD.backward()
            del lossD
            self.optimizerD.step()
            self.netD.zero_grad()

            # Train Generator on Discriminator
            real_labelD = torch.full((b_size,), 1., dtype=torch.float, device=self.device2)

            for p in self.netD.parameters():
                p.requires_grad = False  # to avoid computation on Discriminator

            outputGD = self.netD(fake_images).view(-1)
            lossG = self.criterion(outputGD, real_labelD)
            losses_queue_G.append(lossG.item())

            lossG.backward()
            del lossG
            self.optimizerG.step()
            self.netG.zero_grad()

            if (batch_idx + 1) % settings.dcgan_log_interval_batches == 0:
                mean_loss_G = sum(losses_queue_G) / len(losses_queue_G)
                self.logger.report_scalar(
                    "train", "loss G", iteration=(epoch * len(self.train_loader) + batch_idx + 1), value=mean_loss_G)
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

                print(f'Train Epoch: {epoch}/{self.train_epochs} '
                      f'[{(batch_idx + 1) * len(data)}/{len(self.train_loader.dataset)} '
                      f'({100. * (batch_idx + 1) / len(self.train_loader):.0f}%)]\t'
                      f'Loss G: {mean_loss_G:.6f}\t'
                      f'Pred Dreal: {mean_pred_Dreal:.6f}\tPred Dfake: {mean_pred_Dfake:.6f}\t'
                      f'Loss Dreal: {mean_loss_Dreal:.6f}\tLoss Dfake: {mean_loss_Dfake:.6f}\t')

    def train_phase_2(self, epoch):
        self.netG.eval()
        for p in self.netG.parameters():
            p.requires_grad = False

        self.netE.train()
        losses_queue_E = collections.deque(maxlen=settings.dcgan_log_interval_batches)
        b_size = settings.train_batch_size
        num_batches = int(len(self.train_loader.dataset) / b_size)
        for batch_idx in range(num_batches):
            # Create fake test_images with Generator
            # Generate batch of latent vectors
            with torch.no_grad():
                noise = torch.randn(b_size, settings.dcgan_latent_vector_size, 1, 1, device=self.device)

                fake_images = self.netG(noise).to(self.device2)

            # Train Encoder with all-fake batch
            outputE = self.netE(fake_images.detach())
            lossE = self.criterion2(outputE.view(b_size, -1), noise.view(b_size, -1))
            # print(outputE.view(b_size, -1)[0], noise.view(b_size, -1)[0])
            losses_queue_E.append(lossE.item())

            lossE.backward()
            del lossE
            self.optimizerE.step()
            self.netE.zero_grad()

            if (batch_idx + 1) % settings.dcgan_log_interval_batches == 0:
                mean_loss_E = sum(losses_queue_E) / len(losses_queue_E)
                self.logger.report_scalar(
                    "train", "loss E", iteration=(epoch * len(self.train_loader) + batch_idx + 1), value=mean_loss_E)

                print(f'Train Epoch: {epoch}/{self.train_epochs} '
                      f'[{(batch_idx + 1) * b_size}/{len(self.train_loader.dataset)} '
                      f'({100. * (batch_idx + 1) / num_batches:.0f}%)]\t'
                      f'Loss E: {mean_loss_E:.6f}')

        for p in self.netG.parameters():
            p.requires_grad = True

    def validate(self, epoch: int) -> float:
        if settings.dcgan_training_phase == 1:
            return self.validate_phase_1(epoch)
        elif settings.dcgan_training_phase == 2:
            return self.validate_phase_2(epoch)
        raise Exception(f'Invalid training phase: {settings.dcgan_training_phase}')

    def validate_phase_1(self, epoch):
        self.netG.eval()
        self.netD.eval()
        losses_queue_G = collections.deque()
        with torch.no_grad():
            for i in range(10):
                # Create fake test_images with Generator
                # Generate batch of latent vectors
                b_size = settings.valid_batch_size
                # noise = torch.randn(b_size, settings.dcgan_latent_vector_size, 1, 1, device=self.device)
                noise = ((torch.arange(b_size, dtype=torch.float32) - b_size / 2) * 4 / b_size).reshape(
                    (b_size, 1, 1, 1)).to(self.device)
                real_labelD = torch.full((settings.valid_batch_size,), 1., dtype=torch.float, device=self.device2)

                fake_images = self.netG(noise).to(self.device2)

                outputGD = self.netD(fake_images).view(-1)
                lossG = self.criterion(outputGD, real_labelD)
                losses_queue_G.append(lossG.item())

            mean_loss_G = sum(losses_queue_G) / len(losses_queue_G)
            self.logger.report_scalar("test", "Loss G", iteration=epoch, value=mean_loss_G)
            print(f'Test Epoch: {epoch}\tLoss G: {mean_loss_G:.6f}')

            # Save as image the last batch prediction
            if epoch % settings.dcgan_save_image_interval_epochs == 0:
                fake_images_grid = torchvision.utils.make_grid(fake_images.detach().cpu(),
                                                               nrow=int(np.sqrt(fake_images.size(0)))).permute(1, 2,
                                                                                                               0).numpy()

                self.logger.report_image("generator", "generated images", iteration=epoch, image=fake_images_grid)
        return mean_loss_G

    def validate_phase_2(self, epoch):
        if epoch % settings.dcgan_save_image_interval_epochs > 0:
            return 0

        self.netG.eval()
        self.netE.eval()

        # Create fake test_images with Generator
        data = next(iter(self.valid_loader))
        images = data.to(self.device2)

        with torch.no_grad():
            outputE = self.netE(images).to(self.device)
            outputG = self.netG(outputE)

            original_images_grid = torchvision.utils.make_grid(images.detach().cpu(),
                                                               nrow=images.size(0)).permute(1, 2, 0).numpy()

            reconstructed_images_grid = torchvision.utils.make_grid(outputG.detach().cpu(),
                                                                    nrow=outputG.size(0)).permute(1, 2, 0).numpy()

            combined_grid = np.vstack((original_images_grid, reconstructed_images_grid))

            self.logger.report_image("encoder+generator", "reconstructed images", iteration=epoch, image=combined_grid)
        return 0

    def can_save_weights(self, epoch) -> bool:
        return epoch % 10 == 0 and epoch > 1

    def save_modules_weights(self):
        torch.save(self.netG.state_dict(), self.netG_weights_filepath)
        if settings.dcgan_training_phase == 1:
            torch.save(self.netD.state_dict(), self.netD_weights_filepath)
        if settings.dcgan_training_phase == 2:
            torch.save(self.netE.state_dict(), self.netE_weights_filepath)

    def adjust_lr(self, epoch: int):
        if self.lr_decay_stepsG < len(settings.dcgan_generator_lr_decay_interval_epochs) and \
                epoch > settings.dcgan_generator_lr_decay_interval_epochs[self.lr_decay_stepsG]:
            self.lr_decay_stepsG += 1
            old_lrG = self.optimizerG.param_groups[0]['lr']
            new_lrG = settings.dcgan_generator_lr[
                min(self.lr_decay_stepsG, len(settings.dcgan_generator_lr) - 1)]
            if new_lrG != old_lrG:
                self.optimizerG.param_groups[0]['lr'] = new_lrG
                print(f'Changed learning rate of generator from {old_lrG} to {new_lrG}')
            self.logger.report_scalar("learning rate", "lrG", iteration=epoch,
                                      value=self.optimizerG.param_groups[0]['lr'])

        if settings.dcgan_training_phase == 1 and \
                self.lr_decay_stepsD < len(settings.dcgan_discriminator_lr_decay_interval_epochs) and \
                epoch > settings.dcgan_discriminator_lr_decay_interval_epochs[self.lr_decay_stepsD]:
            self.lr_decay_stepsD += 1
            old_lrD = self.optimizerD.param_groups[0]['lr']
            new_lrD = settings.dcgan_discriminator_lr[
                min(self.lr_decay_stepsD, len(settings.dcgan_discriminator_lr) - 1)]

            if new_lrD != old_lrD:
                self.optimizerD.param_groups[0]['lr'] = new_lrD
                print(f'Changed learning rate of discriminator from {old_lrD} to {new_lrD}')
            self.logger.report_scalar("learning rate", "lrD", iteration=epoch,
                                      value=self.optimizerD.param_groups[0]['lr'])

        if settings.dcgan_training_phase == 2 and \
                self.lr_decay_stepsE < len(settings.dcgan_encoder_lr_decay_interval_epochs) and \
                epoch > settings.dcgan_encoder_lr_decay_interval_epochs[self.lr_decay_stepsE]:
            self.lr_decay_stepsE += 1
            old_lrE = self.optimizerE.param_groups[0]['lr']
            new_lrE = settings.dcgan_encoder_lr[
                min(self.lr_decay_stepsE, len(settings.dcgan_encoder_lr) - 1)]

            if old_lrE != new_lrE:
                self.optimizerE.param_groups[0]['lr'] = new_lrE
                print(f'Changed learning rate of discriminator from {old_lrE} to {new_lrE}')
            self.logger.report_scalar("learning rate", "lrE", iteration=epoch,
                                      value=self.optimizerE.param_groups[0]['lr'])
