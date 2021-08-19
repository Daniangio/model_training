import os
import torch
from clearml import Task, Logger
from common.base_model import BaseModel
from settings import settings


class BaseExperiment:
    def __init__(self, project_name: str, task_name: str, model: BaseModel):
        if not os.path.exists(settings.model_snapshots_path):
            os.makedirs(settings.model_snapshots_path)
        self.task = Task.init(project_name=project_name, task_name=task_name, output_uri=settings.model_snapshots_path)
        self.model = model
        self.logger = Logger.current_logger()

    def get_logger(self):
        return self.logger

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

    def run_testing_experiment(self):
        device = torch.device(settings.device)
        device2 = torch.device(settings.device2)
        self.model.initialize(self.logger, device, device2)

        self.model.test()
