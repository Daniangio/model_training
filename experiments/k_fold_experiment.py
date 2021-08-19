import torch

from common.base_model import BaseModel
from experiments.base_experiment import BaseExperiment
from settings import settings


class KFoldExperiment(BaseExperiment):
    def __init__(self, project_name: str, task_name: str, model: BaseModel):
        super().__init__(project_name, task_name, model)

    def run_experiment(self):
        device = torch.device(settings.device)
        device2 = torch.device(settings.device2)
        self.model.initialize(self.logger, device, device2)

        min_test_loss = None
        for epoch in range(1, self.model.train_epochs + 1):
            self.model.train(epoch=epoch)
            test_loss = self.model.validate(epoch=epoch)
            if self.model.can_save_weights(epoch=epoch):
                min_test_loss = test_loss
                self.model.save_modules_weights()