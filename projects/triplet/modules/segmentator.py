import segmentation_models_pytorch as smp
from common.base_module import BaseModule
from settings import settings


class Segmentator(BaseModule):
    def __init__(self, in_channels=3, **_):
        super(Segmentator, self).__init__()
        self.model = smp.FPN(
            in_channels=in_channels,
            encoder_name='se_resnext50_32x4d',
            encoder_weights='imagenet',
            classes=1,
            activation='sigmoid')

    def forward(self, x):
        return self.model.forward(x)

    def get_weights_filename(self):
        return f'triplet_segmentator_v{settings.triplet_model_version}.pt'
