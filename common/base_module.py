import functools
import cv2
import torch.nn as nn
import numpy as np
import albumentations as albu


def to_tensor(x, **kwargs):
    return x.transpose(2, 0, 1).astype('float32')


def preprocess_input(x, mean=None, std=None, unitary_range=False, input_size=None, **kwargs):
    if unitary_range:
        if x.max() > 1:
            x = x / 255.0

    if mean is not None:
        mean = np.array(mean)
        x = x - mean

    if std is not None:
        std = np.array(std)
        x = x / std

    if input_size is not None:
        y = cv2.resize(x, dsize=input_size, interpolation=cv2.INTER_CUBIC)
        if len(x.shape) == 2 or x.shape[2] == 1:
            x = y.reshape((y.shape[0], -1, 1))
        else:
            x = y

    return x


class BaseModule(nn.Module):
    def __init__(self):
        super(BaseModule, self).__init__()
        self.logger = None

    def bind_logger(self, logger):
        self.logger = logger

    # Overwrite this if necessary
    def init_weights(self):
        pass

    # Overwrite this if necessary
    def set_preprocessing_params(self, params):
        return params

    def get_preprocessing(self, params=None):
        if params is None:
            params = {
                'unitary_range': True
            }
        _transform = [
            albu.Lambda(image=self.get_preprocessing_function(params),
                        mask=self.get_preprocessing_function(params)),
            albu.Lambda(image=to_tensor, mask=to_tensor),
        ]
        return albu.Compose(_transform)

    def get_preprocessing_function(self, params, **kwargs):
        params = self.set_preprocessing_params(params)
        return functools.partial(preprocess_input, **params)

    def plot_gradients(self, iteration: int):
        raise NotImplementedError
        # self.plot_layer_gradient(iteration, None)

    def plot_layer_gradient(self, iteration: int, layer: nn.Module):
        matrix = layer.weight.grad.cpu().numpy()[:100, :100]
        self.logger.report_surface(
            "example_scatter_3d",
            "series_xyz",
            iteration=iteration,
            matrix=matrix,
            xaxis="title x",
            yaxis="title y",
            zaxis="title z",
        )

    def get_weights_filename(self):
        raise NotImplementedError

    def __str__(self):
        model_parameters = filter(lambda p: p.requires_grad, self.parameters())
        nbr_params = sum([np.prod(p.size()) for p in model_parameters])
        return super(BaseModule, self).__str__() + f'\nNbr of trainable parameters: {nbr_params}'
