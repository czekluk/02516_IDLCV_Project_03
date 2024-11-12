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
        model.to(device)
        model.eval()

        edgebox_params = {
            'max_boxes': 4000,
            'min_score': 0.0001,
            "alpha": 0.8,
            "beta": 0.75,
            "edge_min_mag": 0.05
        }
        edgebox_proposer = EdgeBoxesProposer(XIMGPROC_MODEL, edgebox_params)
        eval = Evaluation(nms_iou_threshold=0.2, map_iou_threshold=0.5, score_threshold=0.6)
        mAPs = []
        all_precisions = []
        all_recalls = []
        tested_images = []
        tested_true_boxes = []
        proposed_boxes = []
        proposed_scores = []
        proposed_ious = []
        for minibatch_no, (data, targets) in tqdm(enumerate(self.testloader), total=len(self.testloader)):
            for batch_image, true_boxes in zip(data, targets):
                crops, predicted_boxes = edgebox_proposer.get_n_proposals_test(batch_image.numpy(), n=1000)
        
                # Define the resize transform to a uniform size
                resize_transform = transforms.Compose([
                    transforms.ToPILImage(),
                    transforms.Resize((224, 224)),
                    transforms.ToTensor()
                ])
                tensor_crops = torch.stack([resize_transform(crop) for crop in crops])
                tensor_crops = tensor_crops.to(device)

                with torch.no_grad():
                    output = model(tensor_crops).view(-1)
                
                predicted = (torch.sigmoid(output)).float()
                # Filter predicted boxes with probability < 0.5 for detecting a pothole
                boxes, scores = eval.filter_output(predicted_boxes, predicted.cpu().numpy())
                # Do a non max supression for overlapping boxes that detect the same object
                boxes, scores = eval.non_max_suppression(boxes, scores)
                # Get the mAP
                mAP, precision, recall, ious = eval.mAP(boxes, scores, true_boxes)
                mAPs.append(mAP)
                all_precisions.append(precision)
                all_recalls.append(recall)
                tested_images.append(batch_image)
                tested_true_boxes.append(true_boxes)
                proposed_boxes.append(boxes)
                proposed_scores.append(scores)
                proposed_ious.append(ious)
        
        mAPs = np.array(mAPs)
        # This sumarrizes our whole model and we can use it later in plots or whatever so keep it here
        mean_mAP = np.mean(mAPs)
        print(f"Mean mAP: {mean_mAP}")
        print("#############################")
        # Find images that had the highest mAP
        N = 20
        # N_indices = np.argpartition(mAPs, -N)[-N:]
        N_indices = np.random.choice(len(mAPs), N, replace=False)
        top_N_mAPs = [mAPs[i] for i in N_indices]
        # Extract the precision and recall lists for the top 5 mAPs
        N_precisions = [all_precisions[i] for i in N_indices]
        N_recalls = [all_recalls[i] for i in N_indices]
        N_images, N_true_boxes = [tested_images[i] for i in N_indices], [tested_true_boxes[i] for i in N_indices]
        N_proposed_boxes = [proposed_boxes[i] for i in N_indices]
        N_proposed_scores = [proposed_scores[i] for i in N_indices]
        N_proposed_ious = [proposed_ious[i] for i in N_indices]
        for i in range(N):
            # Make precision recall curve
            print(f"precision_recall_curve_{i+1}")
            print(f"Precision: {N_precisions[i]}")
            print(f"Recall: {N_recalls[i]}")
            print(top_N_mAPs[i])
            print("#############################")
            eval.plot_precision_recall_curve(precision=N_precisions[i], 
                                            recall=N_recalls[i], 
                                            path=os.path.join(figure_path, f"precision_recall_curve_{i+1}.png"), 
                                            title='Precision-Recall Curve (IoU=0.5)', 
                                            mAP=top_N_mAPs[i])
            eval.plot_image_with_boxes(image_tensor=N_images[i],
                                       true_boxes=N_true_boxes[i],
                                       proposed_boxes=N_proposed_boxes[i],
                                       proposed_scores=N_proposed_scores[i],
                                       proposed_ious=N_proposed_ious[i],
                                       save_path=os.path.join(figure_path, f"prediction_{i+1}.png"))
            


