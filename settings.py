import os
from pydantic import BaseSettings

ROOT = os.path.dirname(os.path.abspath(__file__))


class AppSettings(BaseSettings):
    class Config:
        env_file = '.env'
        env_file_encoding = 'utf-8'

    # CLEAR ML
    model_snapshots_path: str
    project_name: str
    task_name: str

    # COMMON
    cuda_visible_devices: str
    device: str
    device2: str
    models_weights_dir: str
    train_batch_size: int
    valid_batch_size: int
    train_num_workers: int
    valid_num_workers: int
    train: bool

    # COORDMAP PROJECT
    coordmap_model_version: str = '1'
    coordmap_train_images_dir: str = 'data/coordmap/master'
    coordmap_valid_images_dir: str = 'data/coordmap/target'
    coordmap_input_size: int = 256
    coordmap_output_size: int = 128
    coordmap_features_map: int = 64
    coordmap_hidden_layers: int = 2
    coordmap_train_epochs: int = 100
    coordmap_save_image_interval_epochs: int = 1
    coordmap_completer_lr: list = [1e-4]
    coordmap_completer_lr_decay_interval_epochs: list = []
    coordmap_log_interval_batches: int = 100

    # DCGAN PROJECT
    dcgan_model_version: str = '1'
    dcgan_train_images_dir: str = 'data/ae_svm/train/128x128'
    dcgan_valid_images_dir: str = 'data/ae_svm/test/128x128'
    dcgan_latent_vector_size: int = 100
    dcgan_image_size: list = [128, 128, 3]
    dcgan_train_epochs: int = 100
    dcgan_save_image_interval_epochs: int = 1
    dcgan_generator_lr: list = [1e-4]
    dcgan_generator_lr_decay_interval_epochs: list = []
    dcgan_discriminator_lr: list = [5e-4]
    dcgan_discriminator_lr_decay_interval_epochs: list = []
    dcgan_encoder_lr: list = [1e-4]
    dcgan_encoder_lr_decay_interval_epochs: list = []
    dcgan_log_interval_batches: int = 50
    dcgan_training_phase: int = 1

    def get_dcgan_image_size(self):
        return tuple(self.dcgan_image_size)

    # AE_SVM PROJECT
    ae_svm_model_version: str
    ae_svm_train_images_dir: str
    ae_svm_valid_images_dir: str
    ae_svm_input_size: int
    ae_svm_input_channels: int
    ae_svm_zdims: int
    ae_svm_train_epochs: int
    ae_svm_log_interval_batches: int
    ae_svm_save_image_interval_epochs: int
    ae_svm_lr: list
    ae_svm_lr_decay_interval_epochs: int
    ae_svm_gamma: float = 1e-2

    def get_ae_svm_train_images_dir(self):
        size = self.ae_svm_input_size
        return os.path.join(self.ae_svm_train_images_dir, f'{str(size)}x{str(size)}')

    def get_ae_svm_valid_images_dir(self):
        size = self.ae_svm_input_size
        return os.path.join(self.ae_svm_valid_images_dir, f'{str(size)}x{str(size)}')

    # TRIPLET PROJECT
    triplet_train_images_dir: str = 'data/triplet/256x256/train_images'
    triplet_train_masks_dir: str = 'data/triplet/256x256/train_masks'
    triplet_good_images_dir: str = 'data/triplet/256x256/good_images'
    triplet_random_masks_dir: str = 'data/triplet/256x256/train_masks'
    triplet_valid_images_dir: str = 'data/triplet/256x256/test_images'
    triplet_valid_masks_dir: str = 'data/triplet/256x256/test_masks'
    triplet_use_whole_generated_image: bool = False
    triplet_model_version: str = '1'
    triplet_input_size: int = 512
    triplet_generator_features_map: int = 32
    triplet_train_epochs: int = 1000
    triplet_save_image_interval_epochs: int = 1
    triplet_generator_lr: list = [1e-3]
    triplet_generator_lr_decay_interval_epochs: list = []
    triplet_discriminator_lr: list = [1e-5]
    triplet_discriminator_lr_decay_interval_epochs: list = []
    triplet_segmentator_lr: list = [1e-3]
    triplet_segmentator_lr_decay_interval_epochs: list = []
    triplet_log_interval_batches: int = 50
    triplet_gd_good_score_ranges: list = [0.4, 0.6]
    triplet_gs_good_loss_ranges: list = [1e-10, 1e-4]

    # SEGMENTATION PROJECT
    segmentation_model_version: str = '1'
    segmentation_train_epochs: int = 1000
    segmentation_train_images_dir: str
    segmentation_train_masks_dir: str
    segmentation_valid_images_dir: str
    segmentation_valid_masks_dir: str
    segmentation_test_images_dir: str
    segmentation_test_masks_dir: str
    segmentation_color_clusters: int = 3
    segmentation_input_size: int = 512
    segmentation_input_channels: int = 3
    segmentation_lr: list = [1e-3]
    segmentation_lr_decay_interval_epochs: list = []
    segmentation_log_interval_batches: int = 5
    segmentation_save_image_interval_epochs: int = 1


settings = AppSettings()
