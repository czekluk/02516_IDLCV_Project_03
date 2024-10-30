import torch
import torch.nn as nn

from experiments.experiment import Experiment
from models.test_classifier import TestClassifier
from data_loader.make_dataset import PotholeDataModule

def main():
    exp = Experiment([TestClassifier], 
                     [{"optimizer": torch.optim.Adam, "params": {"lr": 1e-3}}], 
                     [nn.CrossEntropyLoss], 
                     [100],
                     PotholeDataModule(), 
                     description="Baseline experiment")
    exp.run(save=True, visualize=True)

if __name__ == "__main__":
    main()