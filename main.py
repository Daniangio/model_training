import os

from experiments.base_experiment import BaseExperiment
from projects.expert_neuron.model import ExpertNeuron
from projects.ganmaster.model import GANMaster
from projects.dcgan.model import DCGAN
# from projects.segmentation.model import Segmentation
from projects.segmentation_v1.model import Segmentation
from settings import settings

os.environ['CUDA_VISIBLE_DEVICES'] = settings.cuda_visible_devices


def main():
    model = Segmentation()
    experiment = BaseExperiment(project_name=settings.project_name, task_name=settings.task_name, model=model)
    if settings.train:
        experiment.run_experiment()
    else:
        experiment.run_testing_experiment()


if __name__ == '__main__':
    main()
