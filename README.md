# README #

### What is this repository for? ###

* This repository contains training experiments for DL models that perform defects segmentation on fabric images.

### How do I get set up? ###

* The different models of the experiments are contained in the `projects` folder. The latest and good performing project
  is the `segmentation_v1` folder. Each project contains:
  - the `model.py` file, which implements the training, validation and testing logic
  - the `dataset.py` file, responsible of loading and pre-processing data to be yielded to the DL model
  - the `modules` folder, with the PyTorch implementation of the DNNs
* In order to run an experiment, one has to first configure the `.env` file with all the necessary parameters.
  A compiled `.env` file example can be found on the "cubo" machine (130.192.86.86), at location
  `/home/angioletti/model_training/.env`. It is worth mentioning that the parameter `TRAIN` is a boolean that determines
   whether the script will run the `train()` or the `test()` method of the `model.py` file.
* Experiments and results are recorded using ClearML, accessible at `http://130.192.86.99:8080/dashboard`. The
  `PROJECT_NAME` and `TASK_NAME` of the ClearML experiments are configurable with the corresponding parameters of the
  `.env` file
* To run the repository, just create a virtual environment, activate it and install the required packages, by running
  the `pip install -r requirements.txt` command (Python >=3.6 is required, otherwise there may be problems installing
  the required packages). Then, once configured the `.env` file, open the `main.py` file, import the model
  of the desired project (the default project imported is the segmentation_v1) and run `python main.py`.
* The dataset used by the projects is located on the "cubo" machine at `/mnt/data2/angioletti_data/fastml/images/...`,
  while the model weights are saved at `/mnt/data2/angioletti_data/fastml/models`.