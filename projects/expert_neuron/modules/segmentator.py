import segmentation_models_pytorch as smp

from common.base_module import BaseModule
from settings import settings


class Segmentator(BaseModule):
    def __init__(self, in_channels=3, out_channels=1, **_):
        super(Segmentator, self).__init__()
        self.model = smp.DeepLabV3Plus(
            in_channels=in_channels,
            classes=out_channels,
            encoder_name='se_resnext50_32x4d',
            encoder_weights='imagenet',
            activation='sigmoid')
        self.in_channels = in_channels

    def forward(self, x):
        return self.model.forward(x)

    def set_preprocessing_params(self, params):
        params['unitary_range'] = True
        return params

    def get_weights_filename(self):
        return f'expert_{self.in_channels}c_v{settings.segmentation_model_version}.pt'
