import collections
import os

import cv2
import numpy as np
import torch
import torchvision
from sklearn.svm import OneClassSVM
from torch.utils.data import DataLoader
from common.augmentation import base_augmentation, training_augmentation
from common.base_model import BaseModel
from common.utils import generous_difference
from projects.ae_svm.dataset import Dataset
from projects.ae_svm.modules.autoencoder import Autoencoder, VAE
from settings import settings


class AeSvm(BaseModel):
    def __init__(self, **_):
        super(AeSvm, self).__init__()
        self.ae = VAE(input_size=settings.ae_svm_input_size, channels=settings.ae_svm_input_channels,
                      zdims=settings.ae_svm_zdims)  # Autoencoder(in_channels=3, decoder_block_sizes=(64, ), decoder_depths=(5, ))
        self.train_epochs = settings.ae_svm_train_epochs
        self.train_dataset = Dataset(
            images_dir=settings.get_ae_svm_train_images_dir(),
            augmentation=base_augmentation(size=(settings.ae_svm_input_size, settings.ae_svm_input_size)),
            preprocessing=self.ae.get_preprocessing(),
            channels=settings.ae_svm_input_channels
        )
        self.valid_dataset = Dataset(
            images_dir=settings.get_ae_svm_valid_images_dir(),
            augmentation=base_augmentation(size=(settings.ae_svm_input_size, settings.ae_svm_input_size)),
            preprocessing=self.ae.get_preprocessing(),
            channels=settings.ae_svm_input_channels
        )
        self.train_loader = DataLoader(self.train_dataset,
                                       shuffle=True,
                                       batch_size=settings.train_batch_size,
                                       num_workers=settings.train_num_workers)
        self.valid_loader = DataLoader(self.valid_dataset,
                                       shuffle=False,
                                       batch_size=settings.valid_batch_size,
                                       num_workers=settings.train_num_workers)
        self.criterion = self.ae.loss_function
        self.optimizer = torch.optim.Adam(self.ae.parameters(), lr=settings.ae_svm_lr[0], betas=(.5, .999))
        self.lr_decay_steps = 0

        if not os.path.isdir(settings.models_weights_dir):
            os.makedirs(settings.models_weights_dir)
        self.ae_weights_filepath = os.path.join(settings.models_weights_dir, self.ae.get_weights_filename())

    def initialize(self, logger, device: torch.device, device2: torch.device):
        super(AeSvm, self).initialize(logger, device, device2)
        self.ae.bind_logger(logger)
        if os.path.isfile(self.ae_weights_filepath):
            print(f'Loading model weights from {self.ae_weights_filepath}')
            self.ae.load_state_dict(torch.load(self.ae_weights_filepath, map_location='cpu'), strict=False)
        self.ae = self.ae.to(device)

    def train(self, epoch: int):
        self.adjust_lr(epoch)
        self.ae.train()
        losses_queue = collections.deque(maxlen=settings.ae_svm_log_interval_batches)
        for batch_idx, (data, target) in enumerate(self.train_loader):
            data, target = data.to(self.device), target.to(self.device)
            recon, mu, logvar = self.ae(data)
            loss = self.criterion(recon, target, mu, logvar, epoch)

            self.ae.zero_grad()
            loss.backward()
            # print(self.ae.conv_e1.weight.grad[0,0,:,:])
            # print(self.ae.conv_e3.weight.grad[0,0,:,:])
            if batch_idx == 0 and epoch % 50 == 1:
                self.ae.plot_mean_var(mu, logvar, epoch)
                self.ae.plot_gradients(epoch)
            # print('----------------------------------------------------------')
            self.optimizer.step()

            losses_queue.append(loss.item())

            if (batch_idx + 1) % settings.ae_svm_log_interval_batches == 0:
                mean_loss = sum(losses_queue) / len(losses_queue)
                self.logger.report_scalar(
                    "train", "loss", iteration=(epoch * len(self.train_loader) + batch_idx + 1), value=mean_loss)

                print(f'Train Epoch: {epoch}/{self.train_epochs} '
                      f'[{(batch_idx + 1) * len(data)}/{len(self.train_loader.dataset)} '
                      f'({100. * (batch_idx + 1) / len(self.train_loader):.0f}%)]\tLoss: {mean_loss:.6f}')

    def validate(self, epoch: int) -> float:
        self.ae.train()
        losses_queue = collections.deque()
        with torch.no_grad():
            for data, target in self.valid_loader:
                data, target = data.to(self.device), target.to(self.device)
                recon, mu, logvar = self.ae(data)
                loss = self.criterion(recon, target, mu, logvar, epoch)

                losses_queue.append(loss.item())
            # Save as image the last batch prediction
            if epoch % settings.ae_svm_save_image_interval_epochs == 0:
                original_grid = torchvision.utils.make_grid(data.detach().cpu(), nrow=recon.size(0)).permute(1, 2,
                                                                                                             0).numpy()
                recon = recon.view(-1, self.ae.channels, self.ae.input_size, self.ae.input_size)

                output_grid = torchvision.utils.make_grid(recon.detach().cpu(), nrow=recon.size(0)).permute(1, 2,
                                                                                                            0).numpy()
                composed_grid = np.vstack((original_grid, output_grid))
                self.logger.report_image("image", "test image reconstructed", iteration=epoch, image=composed_grid)

                # Prediction on train images
                for data, target in self.train_loader:
                    data, target = data.to(self.device), target.to(self.device)
                    recon, mu, logvar = self.ae(data)
                    recon = recon.view(-1, self.ae.channels, self.ae.input_size, self.ae.input_size)
                    original_grid = torchvision.utils.make_grid(data.detach().cpu(), nrow=recon.size(0)).permute(1, 2,
                                                                                                                 0).numpy()
                    output_grid = torchvision.utils.make_grid(recon.detach().cpu(), nrow=recon.size(0)).permute(1, 2,
                                                                                                                0).numpy()
                    composed_grid = np.vstack((original_grid, output_grid))
                    self.logger.report_image("image", "train image reconstructed", iteration=epoch, image=composed_grid)
                    break

        mean_loss = sum(losses_queue) / len(losses_queue)
        self.logger.report_scalar("test", "loss", iteration=epoch, value=mean_loss)
        print(f'Test Epoch: {epoch}\tLoss: {mean_loss:.6f}')
        return mean_loss

    def can_save_weights(self, epoch) -> bool:
        return epoch % 20 == 0

    def save_modules_weights(self):
        torch.save(self.ae.state_dict(), self.ae_weights_filepath)

    def adjust_lr(self, epoch: int):
        if epoch % settings.ae_svm_lr_decay_interval_epochs == 0 and epoch > 1:
            self.lr_decay_steps += 1
            old_lr = self.optimizer.param_groups[0]['lr']
            new_lr = settings.ae_svm_lr[
                min(self.lr_decay_steps, len(settings.ae_svm_lr) - 1)]
            self.optimizer.param_groups[0]['lr'] = new_lr
            print(f'Changing learning rate from {old_lr} to {new_lr}')
        self.logger.report_scalar("learning rate", "lr", iteration=epoch, value=self.optimizer.param_groups[0]['lr'])

    def test(self):
        # self.hidden_space_svm_outliers_detection()
        self.reconstruction_generous_difference()

    def reconstruction_generous_difference(self):
        test_images = None
        test_outputs = None
        for batch_idx, (data, _) in enumerate(self.valid_loader):
            if test_images is None:
                test_images = data.permute(0, 2, 3, 1)
            else:
                test_images = torch.vstack((test_images, data.permute(0, 2, 3, 1)))
            data = data.to(self.device)
            recon, _, _ = self.ae(data)
            recon = recon.view(-1, self.ae.channels, self.ae.input_size, self.ae.input_size)
            if test_outputs is None:
                test_outputs = recon.detach().cpu().permute(0, 2, 3, 1)
            else:
                test_outputs = torch.vstack((test_outputs, recon.detach().cpu().permute(0, 2, 3, 1)))
        test_images = test_images.numpy()
        test_outputs = test_outputs.numpy()
        for i in range(test_outputs.shape[0]):
            original_image = test_images[i, ...]
            reconstructed_image = test_outputs[i, ...]
            generous_difference_image = generous_difference(original_image, reconstructed_image)
            composite_image_up = np.hstack((original_image, reconstructed_image))
            composite_image_down = np.hstack((np.abs(original_image - reconstructed_image), generous_difference_image))
            composite_image = np.vstack((composite_image_up, composite_image_down))
            self.logger.report_image('test', f'generous difference {i}', iteration=1, image=composite_image)

    def hidden_space_svm_outliers_detection(self):
        self.train_loader = DataLoader(self.train_dataset,
                                       shuffle=True,
                                       batch_size=1,
                                       num_workers=settings.train_num_workers)
        self.valid_loader = DataLoader(self.valid_dataset,
                                       shuffle=False,
                                       batch_size=1,
                                       num_workers=settings.train_num_workers)

        # SVM Training data
        batch_mu = None
        batch_logvar = None
        self.ae.eval()
        with torch.no_grad():
            for batch_idx, (data, _) in enumerate(self.train_loader):
                data = data.to(self.device)
                _, mu, logvar = self.ae(data)
                if batch_mu is None:
                    batch_mu = mu.detach().cpu()[:1, ...]
                    batch_logvar = logvar.detach().cpu()[:1, ...]
                else:
                    batch_mu = torch.vstack((batch_mu, mu.detach().cpu()[:1, ...]))
                    batch_logvar = torch.vstack((batch_logvar, logvar.detach().cpu()[:1, ...]))
            x_train = batch_mu.numpy()
            print(f'Train shape: {x_train.shape}')
            # SVM Testing data
            test_images = []
            batch_mu = None
            batch_logvar = None
            for batch_idx, (data, _) in enumerate(self.valid_loader):
                test_images.append(data.permute(0, 2, 3, 1).numpy().copy())
                data = data.to(self.device)
                _, mu, logvar = self.ae(data)
                if batch_mu is None:
                    batch_mu = mu.detach().cpu()[:1, ...]
                    batch_logvar = logvar.detach().cpu()[:1, ...]
                else:
                    batch_mu = torch.vstack((batch_mu, mu.detach().cpu()[:1, ...]))
                    batch_logvar = torch.vstack((batch_logvar, logvar.detach().cpu()[:1, ...]))
            x_test = batch_mu.numpy()
            print(f'Test shape: {x_test.shape}')
            test_images = np.concatenate(test_images, axis=0)
            print(f'Test images shape: {test_images.shape}')
            clf = OneClassSVM(nu=0.1, kernel="rbf", gamma=settings.ae_svm_gamma)
            clf.fit(x_train)
            y_pred_train = clf.predict(x_train)
            y_pred_test = clf.predict(x_test)
            n_error_train = y_pred_train[y_pred_train == -1].size
            n_error_test = y_pred_test[y_pred_test == -1].size
            print(f'Outliers in train: {n_error_train}/{y_pred_train.size}')
            print(f'Outliers in test: {n_error_test}/{y_pred_test.size}')
            # z_train = clf.decision_function(x_train)
            # print('Train scores', z_train)
            z_test = clf.decision_function(x_test)
            # print('Test scores', z_test)
            for idx, score in enumerate(z_test):
                severity = max(0, -score)
                if severity > 0:
                    print(severity)
                    severity_mask = np.zeros_like(test_images[idx, :, :, :])
                    severity_mask[:, :, 0] = min(severity, 1)
                    test_images[idx, :, :, 0] = cv2.addWeighted(test_images[idx, ...], 0.5, severity_mask, 0.5, 0)
            # Compose test images in a single image
            n_rows = int(np.sqrt(test_images.shape[0]))
            n_columns = int(np.ceil(test_images.shape[0] / n_rows))
            composed_grid = np.zeros(
                (n_rows * test_images.shape[1], n_columns * test_images.shape[2], test_images.shape[3]))
            for row in range(n_rows):
                for col in range(n_columns):
                    i = row * n_rows + col
                    if i < test_images.shape[0]:
                        composed_grid[row * test_images.shape[1]:row * test_images.shape[1] + test_images.shape[1],
                        col * test_images.shape[2]:col * test_images.shape[2] + test_images.shape[2], :] = test_images[
                            i, ...]
            self.logger.report_image("test images", "anomalies grading", iteration=1, image=composed_grid)
