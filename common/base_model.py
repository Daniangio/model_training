from abc import ABC
from typing import Any

import torch
from clearml import Logger


def format_bytes(size):
    # 2**10 = 1024
    power = 2 ** 10
    n = 0
    power_labels = {0: '', 1: 'kilo', 2: 'mega', 3: 'giga', 4: 'tera'}
    while size > power:
        size /= power
        n += 1
    return size, power_labels[n] + 'bytes'


def show_gpu_memory_info():
    t = format_bytes(torch.cuda.get_device_properties(1).total_memory)
    r = torch.cuda.memory_reserved(1)
    a = torch.cuda.memory_allocated(1)
    f = format_bytes(r - a)  # free inside reserved
    r = format_bytes(r)
    a = format_bytes(a)
    print(f'GPU Memory: Total {t}\tReserved {r}\t Allocated {a}\tFree {f}')


class BaseModel(ABC):
    def __init__(self):
        self.logger = None
        self.device = None
        self.device2 = None

        self.train_epochs = 100

    def initialize(self, logger: Logger, device: torch.device, device2: torch.device):
        self.logger = logger
        self.device = device
        self.device2 = device2

    def train(self, epoch: int) -> Any:
        raise NotImplementedError

    def validate(self, epoch: int) -> Any:
        raise NotImplementedError

    def test(self) -> Any:
        raise NotImplementedError

    def can_save_weights(self, **kwargs) -> bool:
        raise NotImplementedError

    def save_modules_weights(self):
        raise NotImplementedError
