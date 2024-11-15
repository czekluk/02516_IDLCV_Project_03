import torch
import os
import datetime
import json
import glob
import numpy as np
from tqdm import tqdm

from evaluation.evaluation import Evaluation
from models.classifiers import ClassifierAlexNet64
from region_proposals.edgeboxes import EdgeBoxesProposer
from trainer import Trainer
from models import *
from data_loader import *
from loss_functions import *
from visualizer import Visualizer
import torchvision.transforms as transforms
from inference import Inference

PROJECT_BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA_DIR = os.path.join(PROJECT_BASE_DIR, 'data')
XIMGPROC_MODEL = os.path.join(PROJECT_BASE_DIR, 'src', 'region_proposals', 'ximgproc_model.yml.gz')

class Experiment:
    def __init__(self, models, optimizers, losses, epochs, datamodule, transforms, description="Baseline experiment", save_dir=os.path.join(PROJECT_BASE_DIR, 'results')):
        '''
        Initialize the experiment with the models, optimizers, losses, data, and epochs.

        models (list): List of models to be trained.
        optimizers (list): List of optimizers to be used for training.
        losses (tuple): List of loss functions to be used for training and their names.
        epochs (list): List of number of epochs to train the models.
        datamodule (DataModule): DataModule object that contains the data.
        transforms (nn.Module): transform module
        description (str): Description of the experiment.
        '''
        self.models = models
        self.optimizers = optimizers
        self.losses = losses
        self.epochs = epochs
        self.trainloader = datamodule.train_dataloader()
        self.testloader = datamodule.test_dataloader()
        self.datamodule = datamodule
        self.description = description
        self.save_dir = save_dir
        self.timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        self.results = {}
        self.transforms = transforms

    def run(self, save=True, visualize=False):
        '''
        Run the defined experiment.
        '''
        print("Running experiment: ", self.description)
        trainer = Trainer(self.models, self.optimizers, self.losses, self.epochs, self.trainloader, self.testloader, self.transforms, self.description)
        self.results = trainer.train()
        if save:
            self.save(top_k=1)
        if visualize:
            self.visualize()

    def save(self, top_k=1):
        '''
        Save the experiment results.
        '''
        if self.results is None:
            print("No results to save. Run the experiment first.")
            return
        
        result_path = os.path.join(self.save_dir, f"{self.timestamp}_{self.description}")
        model_path = os.path.join(result_path, "saved_models")
        json_path = os.path.join(result_path, f"{self.timestamp}_{self.description}.json")

        # Create directories if they don't exist
        if not os.path.exists(result_path):
            os.makedirs(result_path)

        if not os.path.exists(model_path):
            os.makedirs(model_path)

        if not os.path.exists(json_path):
            with open(json_path, 'w') as f:
                json.dump([], f)

        # Save the models
        for k in range(top_k):
            best_model = self.results[k]
            torch.save(best_model["model"], os.path.join(model_path, f"{self.timestamp}_{self.description}_{best_model['test_acc'][-1]:.4f}_{best_model['model_name']}.pth"))

        # Save the results
        with open(json_path, 'r') as f:
            try:
                data = json.load(f)
            except:
                data = []

        self.format_results()

        # Append the new results to the existing results
        data.extend(self.results)
        # Sort the results based on the test accuracy
        data = sorted(data, key=lambda x: x['test_acc'][-1], reverse=True)
        # Save the results
        with open(json_path, 'w') as f:
            json.dump(data, f)

    def format_results(self):
        '''
        Format the results in a human-readable format.
        '''
        for entry in self.results:
            if "model" in entry:
                entry["model"] = str(entry["model"])
            if "optimizer" in entry["optimizer_config"]:
                entry["optimizer_config"]["optimizer"] = str(entry["optimizer_config"]["optimizer"])
            if "loss" in entry:
                entry["loss"] = str(entry["loss"])
            if "transform" in entry:
                entry["transform"] = str(entry["transform"])
            if "timestamp" in entry:
                entry["timestamp"] = str(entry["timestamp"])

    def visualize(self, result_path=None, model_path=None):
        '''
        Visualize the experiment results.
        '''
        if result_path is not None and model_path is not None:
            figure_path = os.path.join(result_path, "figures")
            models_path = os.path.join(result_path, "saved_models")
        else:
            if self.results is None:
                print("No results to visualize. Run the experiment first.")
                return
            
            best_model = self.results[0]

            result_path = os.path.join(self.save_dir, f"{self.timestamp}_{self.description}")
            figure_path = os.path.join(result_path, "figures")
            models_path = os.path.join(result_path, "saved_models")
            model_path = os.path.join(models_path, f"{self.timestamp}_{self.description}_{best_model['test_acc'][-1]:.4f}_{best_model['model_name']}.pth")

        # Create directories if they don't exist
        if not os.path.exists(figure_path):
            os.makedirs(figure_path)

        # Here you can specify which of the best models you want to choose, it is scuffed so feel free to change if you use multiple models
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        model = ClassifierAlexNet64()
        model = torch.load(model_path, map_location=device, weights_only=False)

        edgebox_params = {
            'max_boxes': 4000,
            'min_score': 0.0001,
            "alpha": 0.8,
            "beta": 0.75,
            "edge_min_mag": 0.05
        }
        edgebox_proposer = EdgeBoxesProposer(XIMGPROC_MODEL, edgebox_params)
        inf = Inference(model, edgebox_proposer, nms_iou=0.5, map_iou=0.5, score_threshold=0.5, 
                    figure_path=os.path.join(figure_path))
        inf.run_inference(self.testloader)
            


