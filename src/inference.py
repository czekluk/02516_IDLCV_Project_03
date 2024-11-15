import torch
import os
import json
import glob
import numpy as np
from tqdm import tqdm
import datetime

from evaluation.evaluation import Evaluation
from data_loader.make_dataset import PotholeDataModule
from models.classifiers import ClassifierAlexNet64, ClassifierAlexNet
from region_proposals.edgeboxes import EdgeBoxesProposer
from trainer import Trainer
from models import *
from data_loader import *
from loss_functions import *
from visualizer import Visualizer
import torchvision.transforms as transforms

PROJECT_BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(PROJECT_BASE_DIR, 'data')
XIMGPROC_MODEL = os.path.join(PROJECT_BASE_DIR, 'src', 'region_proposals', 'ximgproc_model.yml.gz')

class Inference:
    def __init__(self, model, edgebox_proposer, nms_iou, map_iou, score_threshold, figure_path):
        self.model = model
        self.edgebox_proposer = edgebox_proposer
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
        self.model.eval()
        self.nms_iou = nms_iou
        self.map_iou = map_iou
        self.score_threshold = score_threshold
        self.figure_path = os.path.join(figure_path,f"{datetime.datetime.now().strftime("%Y%m%d-%H%M%S")}_NMS-{nms_iou}_mAP-{map_iou}_inference")
        if not os.path.exists(self.figure_path):
            os.makedirs(self.figure_path)

    def run_inference(self, dataloader):
        # Initialize evaluation object
        eval = Evaluation(nms_iou_threshold=self.nms_iou, map_iou_threshold=self.map_iou, score_threshold=self.score_threshold)

        # do evaluation per image
        APs, all_precisions, all_recalls, tested_images, tested_true_boxes, proposed_boxes, proposed_scores, proposed_ious, pre_nms_boxes = [], [], [], [], [], [], [], [], []
        for minibatch_no, (data, targets) in tqdm(enumerate(dataloader), total=len(dataloader)):
            for img_idx, (batch_image, true_boxes) in enumerate(zip(data, targets)):
                crops, predicted_boxes = self.edgebox_proposer.get_n_proposals_test(batch_image.numpy(), n=1000)
        
                # Define the resize transform to a uniform size
                resize_transform = transforms.Compose([
                    transforms.ToPILImage(),
                    transforms.Resize((224, 224)),
                    transforms.ToTensor()
                ])
                tensor_crops = torch.stack([resize_transform(crop) for crop in crops])
                tensor_crops = tensor_crops.to(self.device)

                with torch.no_grad():
                    output = self.model(tensor_crops).view(-1)
                
                predicted = (torch.sigmoid(output)).float()
                # Filter predicted boxes with probability < 0.5 for detecting a pothole
                boxes, scores = eval.filter_output(predicted_boxes, predicted.cpu().numpy())
                # Do a non max supression for overlapping boxes that detect the same object
                pre_nms_boxes.append(boxes)
                boxes, scores = eval.non_max_suppression(boxes, scores)
                # Get the mAP
                AP, precision, recall, ious = eval.AP(boxes, scores, true_boxes)
                eval.prep_mAP(boxes, scores, true_boxes, img_idx=img_idx)
                APs.append(AP)
                all_precisions.append(precision)
                all_recalls.append(recall)
                tested_images.append(batch_image)
                tested_true_boxes.append(true_boxes)
                proposed_boxes.append(boxes)
                proposed_scores.append(scores)
                proposed_ious.append(ious)
        
        # calculate mAP & plot precision-recall curve
        mAP, precision, recall = eval.mAP()
        eval.plot_precision_recall_curve(precision=precision, 
                                            recall=recall, 
                                            path=os.path.join(self.figure_path, f"precision_recall_curve_mAP.png"), 
                                            title=f'Precision-Recall Curve (IoU={self.map_iou})', 
                                            mAP=mAP)
        
        # plot the best NMS result
        eval.plot_NMS_results(tested_images, proposed_boxes, pre_nms_boxes, self.nms_iou, save_path=os.path.join(self.figure_path, f"nms_results.png"))
        print(f"mAP: {mAP}")
        print(f"mean precision: {precision[-1]}")
        print(f"mean recall: {recall[-1]}")

        res_dict =  {
            "mAP": mAP,
            "mean_precision": precision[-1],
            "mean_recall": recall[-1],
        }

        with open(os.path.join(self.figure_path, "results.json"), 'w') as f:
            json.dump(res_dict, f)

        # plot 4 results with highest AP
        eval.plot_best_predictions(tested_images, tested_true_boxes, proposed_boxes, proposed_scores, proposed_ious, APs, 5, save_path=self.figure_path)
        # plot 4 results with lowest AP
        eval.plot_worst_predictions(tested_images, tested_true_boxes, proposed_boxes, proposed_scores, proposed_ious, APs, 5, save_path=self.figure_path)

if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model_path = os.path.join(PROJECT_BASE_DIR,"results/shared_models/20241112-171528_['Baseline experiment']_0.8984_ClassifierAlexNet.pth")
    model = ClassifierAlexNet()
    model = torch.load(model_path, map_location=device, weights_only=False)

    testloader = PotholeDataModule().test_dataloader()

    # Do edgebox detection
    edgebox_params = {
        'max_boxes': 4000,
        'min_score': 0.0001,
        "alpha": 0.8,
        "beta": 0.75,
        "edge_min_mag": 0.05
    }
    edgebox_proposer = EdgeBoxesProposer(XIMGPROC_MODEL, edgebox_params)

    inf = Inference(model, edgebox_proposer, nms_iou=0.8, map_iou=0.8, score_threshold=0.5, 
                    figure_path=os.path.join(PROJECT_BASE_DIR, "results","inference"))
    inf.run_inference(testloader)

    
