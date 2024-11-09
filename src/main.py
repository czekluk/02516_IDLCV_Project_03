import torch
import torch.nn as nn

from experiments.experiment import Experiment
from models.test_classifier import TestClassifier
from data_loader.make_dataset import PotholeDataModule
from data_loader.custom_transforms import base_transform

def main():
    exp = Experiment([TestClassifier], 
                     [{"optimizer": torch.optim.Adam, "params": {"lr": 1e-3}}], 
                     [(nn.BCEWithLogitsLoss(), "Binary Cross Entropy")], 
                     [3],
                     PotholeDataModule(), 
                     transforms=[None],
                     description=["Baseline experiment"])
    exp.run(save=True, visualize=True)

if __name__ == "__main__":
    main()