import torch
import os
import datetime
import json
import glob

from trainer import Trainer
from models import *
from data import *
from loss_functions import *
from visualizer import Visualizer

PROJECT_BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA_DIR = os.path.join(PROJECT_BASE_DIR, 'data')

class Experiment:
    def __init__(self, models, optimizers, losses, epochs, datamodule, description="Baseline experiment", save_dir=os.path.join(PROJECT_BASE_DIR, 'results')):
        '''
        Initialize the experiment with the models, optimizers, losses, data, and epochs.

        models (list): List of models to be trained.
        optimizers (list): List of optimizers to be used for training.
        losses (list): List of loss functions to be used for training.
        epochs (list): List of number of epochs to train the models.
        datamodule (DataModule): DataModule object that contains the data.
        description (str): Description of the experiment.
        '''
        self.models = models
        self.optimizers = optimizers
        self.losses = losses
        self.epochs = epochs
        self.trainloader = datamodule.train_loader()
        self.testloader = datamodule.test_loader()
        self.datamodule = datamodule
        self.description = description
        self.save_dir = save_dir
        self.timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        self.results = {}

    def run(self, save=True, visualize=False):
        '''
        Run the defined experiment.
        '''
        print("Running experiment: ", self.description)
        trainer = Trainer(self.models, self.optimizers, self.losses, self.epochs, self.trainloader, self.testloader, self.description, self.datamodule)
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

    def visualize(self):
        '''
        Visualize the experiment results.
        '''
        if self.results is None:
            print("No results to visualize. Run the experiment first.")
            return
        
        result_path = os.path.join(self.save_dir, f"{self.timestamp}_{self.description}")
        figure_path = os.path.join(result_path, "figures")

        # Create directories if they don't exist
        if not os.path.exists(figure_path):
            os.makedirs(figure_path)

        # Visualize the results
        vis = Visualizer()
        
