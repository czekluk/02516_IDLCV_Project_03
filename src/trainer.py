import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm
from data_loader.custom_transforms import base_transform
from typing import List
from torch.utils.data import DataLoader
import datetime
from region_proposals.edgeboxes import EdgeBoxesProposer
import os
import torchvision.transforms as transforms

XIMGPROC_MODEL = os.path.join(os.path.dirname(__file__), 'region_proposals', 'ximgproc_model.yml.gz')


class Trainer:
    def __init__(self, models: List[nn.Module], optimizer_functions: List[dict], 
                 criterion_functions,
                 epochs: int, train_loader: DataLoader, test_loader: DataLoader,
                 train_transform, description) -> None:
        """
        Class for training different models with different optimizers and different numbers of epochs.
        
        Args:   models              -   list of models. The models are not instances but classes. example: [AlexNet, ResNet]
                optimizer_funcitons -   list of dictionaries specifying different optimizers.
                                        example: optimizers = [{"optimizer": torch.optim.Adam, "params": {"lr": 1e-3}}]
                epochs              -   list of different epochs to train. example: [10, 15]
                train_loader        -   torch.utils.data.DataLoader
                test_loader         -   torch.utils.data.DataLoader
        """
        assert len(models) == len(description)

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.criterions = criterion_functions
        self.models = models
        self.optimizer_functions = optimizer_functions
        print("optimizer_functions: ", self.optimizer_functions)
        self.epochs = epochs
        print("epochs: ", self.epochs)
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.train_transform = train_transform
        self.description = description
        print(f"Training on device: {self.device}")
    
    
    def train(self) -> List[dict]:
        """
        Train the different models, with different optimizers, with different number of epochs.
        
        Returns:    List of dictionaries representing different experiments.
                    The list is sorted in descending order based on the achieved accuracy
                    after the final epoch.
        """
        outputs = []
        count = 0
        for network in self.models:
            for criterion, criterion_description in self.criterions:
                for optimizer_config in self.optimizer_functions:
                    for epoch_no in self.epochs:
                        print("#########################################################")
                        print(f"Training model: {network.__name__}")
                        print(f"Description: {self.description[count]}")
                        optimizer = optimizer_config["optimizer"]
                        optimizer_name = optimizer.__name__
                        print(f"Optimizer: {optimizer_name}")
                        print(f"Criterion: {criterion}")
                        print(f"Training for {epoch_no} epochs")
                        model = network()
                        print("optimizer_config, epoch_no, [criterion]", optimizer_config, epoch_no, [criterion, criterion_description])
                        out_dict = self._train_single_configuration(model, optimizer_config, epoch_no, (criterion, criterion_description))
                        out_dict["description"] = self.description[count]
                        out_dict["timestamp"] = datetime.datetime.now()
                        out_dict["transform"] = self.train_transform
                        outputs.append(out_dict)
            count += 1
        outputs_sorted = sorted(outputs, key=lambda x: x['test_acc'][-1], reverse=True)
        return outputs_sorted
    
    
    def _train_single_configuration(self, model: nn.Module, optimizer_config: dict, num_epochs: int, criterion: tuple) -> dict:
        model.to(self.device)
        optimizer = optimizer_config["optimizer"](model.parameters(), **optimizer_config["params"])
        criterion, criterion_description = criterion
        out_dict = {
            'model_name':       model.__class__.__name__,
            'description':      None,
            'timestamp':        None,
            'model':            model,
            'train_acc':        [],
            'test_acc':         [],
            'train_loss':       [],
            'test_loss':        [],
            'epochs':           num_epochs,
            'optimizer_config': optimizer_config,
            'criterion':        criterion_description,
            'transform':        None
            }
        
        
        edgebox_params = {
            'max_boxes': 4000,
            'min_score': 0.0001,
            "alpha": 0.8,
            "beta": 0.75,
            "edge_min_mag": 0.05
        }
        edgebox_proposer = EdgeBoxesProposer(XIMGPROC_MODEL, edgebox_params)
        n = 500
        for epoch in tqdm(range(num_epochs), unit='epoch'):
            model.train()
            train_loss = []
            train_acc = []

            for minibatch_no, (data, targets) in tqdm(enumerate(self.train_loader), total=len(self.train_loader)):
                optimizer.zero_grad()
                train_loss_in_batch = []
                train_acc_in_batch = []
                # For each image in the batch
                for batch_image, true_boxes in zip(data, targets):
                    train_correct = 0
                    crops, target = edgebox_proposer.get_n_proposals_train(batch_image.numpy(), true_boxes.numpy(), iou_threshold=0.75, n=n, positive_class_ratio=0.3)
                
                    # Define the resize transform to a uniform size
                    resize_transform = transforms.Compose([
                        transforms.ToPILImage(),
                        transforms.Resize((224, 224)),
                        transforms.ToTensor()
                    ])
                    tensor_crops = torch.stack([resize_transform(crop) for crop in crops])
                    tensor_crops = tensor_crops.to(self.device)

                    target = torch.tensor(target, dtype=torch.float32)  # Convert to tensor with float dtype
                    target = target.view(-1).to(self.device)  # Reshape to match the shape of output

                    output = model(tensor_crops).view(-1)
                    loss = criterion(output, target.clone().detach().float().requires_grad_(True))
                    loss.backward()
                    optimizer.step()

                    train_loss_in_batch.append(loss.item())
                    predicted = (torch.sigmoid(output) > 0.5).float()
                    train_correct += (target==predicted).sum().cpu().item()
                    train_acc_in_batch.append(train_correct / n)
                
                train_loss.append(np.mean(train_loss_in_batch))
                train_acc.append(np.mean(train_acc_in_batch))

            test_loss = []
            test_acc = []
            model.eval()
            for minibatch_no, (data, targets) in enumerate(self.test_loader):
                test_acc_in_batch = []
                test_loss_in_batch = []
                for batch_image, true_boxes in zip(data, targets):
                    test_correct = 0
                    crops, target = edgebox_proposer.get_n_proposals_train(batch_image.numpy(), true_boxes.numpy(), iou_threshold=0.75, n=n, positive_class_ratio=0.3)
            
                    # Define the resize transform to a uniform size
                    resize_transform = transforms.Compose([
                        transforms.ToPILImage(),
                        transforms.Resize((224, 224)),
                        transforms.ToTensor()
                    ])
                    tensor_crops = torch.stack([resize_transform(crop) for crop in crops])
                    tensor_crops = tensor_crops.to(self.device)

                    target = torch.tensor(target, dtype=torch.float32)  # Convert to tensor with float dtype
                    target = target.view(-1).to(self.device)  # Reshape to match the shape of output

                    with torch.no_grad():
                        output = model(tensor_crops).view(-1)
                    loss = criterion(output, target.clone().detach().float())

                    test_loss_in_batch.append(loss.item())
                    predicted = (torch.sigmoid(output) > 0.5).float()
                    test_correct += (target==predicted).sum().cpu().item()
                    test_acc_in_batch.append(test_correct / n)
                
                test_acc.append(np.mean(test_acc_in_batch))
                test_loss.append(np.mean(test_loss_in_batch))

            # Add entries output json
            mean_train_loss = np.mean(train_loss)
            mean_test_loss = np.mean(test_loss)
            out_dict['train_loss'].append(mean_train_loss)
            out_dict['test_loss'].append(mean_test_loss)

            out_dict['train_acc'].append(np.mean(train_acc))
            out_dict['test_acc'].append(np.mean(test_acc))

            # Print results of this epoch
            print(f"Loss train: {mean_train_loss:.3f}\t test: {mean_test_loss:.3f}\t",
                f"Accuracy train: {out_dict['train_acc'][-1]*100:.1f}%\t test: {out_dict['test_acc'][-1]*100:.1f}%\t")
            
        # Print final results
        print(f"Loss train: {mean_train_loss:.3f}\t test: {mean_test_loss:.3f}\t",
                f"Accuracy train: {out_dict['train_acc'][-1]*100:.1f}%\t test: {out_dict['test_acc'][-1]*100:.1f}%\t")
        
        return out_dict