import torch
import torch.nn as nn

from experiments.experiment import Experiment
from models.classifiers import ClassifierAlexNet64, ClassifierAlexNet
from data_loader.make_dataset import PotholeDataModule

def main():
    exp = Experiment([ClassifierAlexNet], 
                     [{"optimizer": torch.optim.Adam, "params": {"lr": 1e-3}}], 
                     [(nn.BCEWithLogitsLoss(), "Binary Cross Entropy")], 
                     [5],
                     PotholeDataModule(), 
                     transforms=[None],
                     description=["Baseline experiment"])
    exp.run(save=True, visualize=True)
    # exp.visualize()

if __name__ == "__main__":
    main()