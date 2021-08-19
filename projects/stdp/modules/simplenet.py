import math
from torch import nn
import torch
import torchvision
from common.base_module import BaseModule
from settings import settings
from matplotlib import pyplot as plt
import numpy as np


class SimpleNet(BaseModule):
    def __init__(self, in_neurons, out_neurons, **_):
        super(SimpleNet, self).__init__()
        self.linear = nn.Linear(in_neurons, out_neurons, bias=False)
        self.weight_image_size = int(math.sqrt(in_neurons))
        self.lr = 0.1
        self.nu = 0.95
        self.wmax = 1
        self.wmin = 0
        self.tau = 1
        self.offset = 1
        self.init_weights()
    
    def init_weights(self):
        self.linear.weight.data.fill_(0.01)
        

    def forward(self, x):
        self.last_input = x
        self.last_output = self.linear(x)
        return self.last_output
    
    def stdp_update(self):
        for batch_idx in range(self.last_output.size(0)):
            sorted_out, indices = torch.sort(self.last_output[batch_idx, :], descending=True) # sorted: (out_neurons), indices: (out_neurons)
            to_update = self.linear.weight.data[indices[0], :] # to_update: (in_neurons)
            #print('pre', to_update)
            #print('input', self.last_input[batch_idx, :])
            dw = self.lr * (torch.exp(self.last_input[batch_idx, :] - torch.mean(self.last_input[batch_idx, :])) / self.tau - self.offset)
            #print('dw', dw)
            #plt.imshow(self.last_input[batch_idx, :].detach().cpu().numpy().reshape(8,8,1))
            #plt.show()
            limit_increment = torch.zeros_like(dw)
            limit_increment[dw > 0] = (self.wmax - to_update[dw > 0])
            limit_increment[dw <= 0] = (to_update[dw <= 0] - self.wmin)
            #print('incr', limit_increment)
            to_update = to_update + dw * limit_increment**self.nu
            #print('to_update', to_update)
            self.linear.weight.data[indices[0], :] = to_update
    
    def visualize_weights(self):
        data = self.linear.weight.data.detach().cpu().view(-1, self.weight_image_size, self.weight_image_size)
        data = torch.unsqueeze(data, 1)
        weights_grid = torchvision.utils.make_grid(data, nrow=int(math.sqrt(data.size(0)))).permute(1, 2, 0).numpy()
        weights_grid = (weights_grid - np.min(weights_grid)) / (np.max(weights_grid) - np.min(weights_grid))
        plt.imshow(weights_grid)
        plt.show()

    def set_preprocessing_params(self, params):
        pass

    def get_weights_filename(self):
        return f'stdp_model{settings.segmentation_model_version}.pt'
