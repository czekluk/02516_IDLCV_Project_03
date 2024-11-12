import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import os
import random

class Evaluation:
    def __init__(self, nms_iou_threshold=0.7, map_iou_threshold=0.5, score_threshold=0.5):
        self.nms_iou_threshold = nms_iou_threshold
        self.map_iou_threshold = map_iou_threshold
        self.score_threshold = score_threshold

    def non_max_suppression(self, boxes: list, scores: list) -> list:
        '''
        Calculates the non-max suppression for the given boxes and scores.
        Inputs:
            boxes: list of boxes in the format [xmin, ymin, xmax, ymax]
            scores: list of scores for each box for positive class (pothole present)
        Outputs:
            boxes: list of boxes after non-max suppression in the format [xmin, ymin, xmax, ymax]
            scores: list of scores after non-max suppression for positive class (pothole present)
        '''
        # Sort the boxes based on scores
        indices = np.argsort(scores)
        boxes = list(np.array(boxes)[indices])
        scores = list(np.array(scores)[indices])

        # Reverse the order of the boxes and scores to have them in descending order
        boxes.reverse()
        scores.reverse()

        # Perform non-max suppression
        base_idx = 0
        comp_idx = 1
        while base_idx < len(boxes) - 1:
            while comp_idx < len(boxes):
                if self.iou(boxes[base_idx], boxes[comp_idx]) > self.nms_iou_threshold or \
                    self.get_intersection(boxes[base_idx], boxes[comp_idx]) >= 0.65 * self.get_area(boxes[comp_idx]):
                    boxes.pop(comp_idx)
                    scores.pop(comp_idx)
                else:
                    comp_idx += 1
            base_idx += 1
            comp_idx = base_idx + 1

        return boxes, scores

    def iou(self, area1: list, area2: list) -> float:
        '''
        Calculates the Intersection over Union (IoU) of two boxes.
        Inputs:
            area1: list of coordinates of the first box in the format [xmin, ymin, xmax, ymax]
            area2: list of coordinates of the second box in the format [xmin, ymin, xmax, ymax]
        Outputs:
            iou: Intersection over Union of the two boxes
        '''

        return self.get_intersection(area1, area2) / self.get_union(area1, area2)

    def get_intersection(self, area1: list, area2:list) -> float:
        '''
        taken from Alex part
        '''
        xmin_1, ymin_1, xmax_1, ymax_1 = area1
        xmin_2, ymin_2, xmax_2, ymax_2 = area2
        
        x_left = max(xmin_1, xmin_2) # Leftmost x-coordinate of the intersection
        y_top = max(ymin_1, ymin_2) # Topmost y-coordinate of the intersection
        x_right = min(xmax_1, xmax_2) # Rightmost x-coordinate of the intersection
        y_bottom = min(ymax_1, ymax_2) # Bottommost y-coordinate of the intersection
        
        if x_right < x_left or y_bottom < y_top:
            return 0
        else:
            return (x_right - x_left + 1) * (y_bottom - y_top + 1)
    
    def get_area(self, box_coords):
        xmin, ymin, xmax, ymax = box_coords
        area = (xmax - xmin + 1) * (ymax - ymin + 1)
        return area

    def get_union(self, area1: list, area2: list) -> float:
        '''
        taken from Alex part
        '''
        area_1 = self.get_area(area1)
        area_2 = self.get_area(area2)
        
        return area_1 + area_2 - self.get_intersection(area1, area2)

    def mAP(self, boxes: list, scores: list, ground_truth: list) -> float:
        '''
        Calculates the mean Average Precision (mAP) for given boxes & ground truths.
        Should be called after non-max suppression.
        To be used for only one image. Loop over all images to get mAP for the dataset.
        Inputs:
            boxes: list of boxes in the format [xmin, ymin, xmax, ymax]
            scores: list of scores for each box for positive class (pothole present)
            ground_truth: list of ground truth boxes in the format [xmin, ymin, xmax, ymax]
        Outputs:
            mAP: mean Average Precision
        '''
        # sort detections in decreasing order of confidence scores
        indices = np.argsort(scores)
        boxes = list(np.array(boxes)[indices])
        boxes.reverse()

        gt_mask = np.zeros(len(ground_truth)) # mask to keep track of ground truths (1 = matched, 0 = not matched)
        box_mask = np.zeros(len(boxes)) # mask to keep track of boxes (1 = matched, 0 = not matched)

        precision = []
        recall = []
        ious = np.zeros(len(boxes))

        # go through all boxes and compare with ground truths
        for box_id in range(len(boxes)):
            # calculate iou with all ground truths
            for gt_id in range(len(ground_truth)):
                if gt_mask[gt_id] == 0:
                    if self.iou(boxes[box_id], ground_truth[gt_id]) > self.map_iou_threshold:
                        gt_mask[gt_id] = 1
                        box_mask[box_id] = 1
                        ious[box_id] = self.iou(boxes[box_id], ground_truth[gt_id])
                        break
                    if self.iou(boxes[box_id], ground_truth[gt_id]) > 0:
                        ious[box_id] = self.iou(boxes[box_id], ground_truth[gt_id])

            # calculate precision and recall
            precision.append(np.sum(box_mask) / (box_id + 1))
            recall.append(np.sum(gt_mask) / len(ground_truth))
    
        # calculate average precision
        ap = 0
        for i in range(len(precision)):
            if i == 0:
                ap += recall[i] * precision[i]
            else:
                ap += (recall[i] - recall[i-1]) * precision[i]
        
        return ap, precision, recall, ious
    
    def plot_precision_recall_curve(self, precision: list, recall: list, path: str = os.path.join(os.getcwd(),'precision_recall_curve.png'), title: str = 'Precision-Recall Curve (IoU=0.5)', mAP = 0):
        '''
        Plots the Precision-Recall curve.
        Inputs:
            precision: list of precision values
            recall: list of recall values
        Outputs:
            None
        '''
        plt.plot(recall, precision, color='r', linewidth=4)
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title(title)
        plt.xlim(-0.05, 1.05)
        plt.ylim(-0.05, 1.05)
        plt.grid(linestyle='--')

         # Add text displaying the mean mAP
        plt.text(0.6, 0.1, f'mAP: {mAP:.3f}', fontsize=12, color='blue', 
                bbox=dict(facecolor='white', alpha=0.8))
        plt.savefig(path)
        plt.close()
    
    def filter_output(self, boxes: list, scores: list) -> list:
        '''
        Returns the boxes and scores for the pothole class. 
        Assumes a sigmoid function is applied to the model output to get the scores.
        Inputs:
            boxes: list of boxes in the format [xmin, ymin, xmax, ymax]
            scores: list of scores for each box for positive class (pothole present)
        Outputs:
            class_boxes: list of boxes for potholes
            class_scores: list of scores for potholes
        '''
        class_boxes = []
        class_scores = []
        for i in range(len(scores)):
            if scores[i] > self.score_threshold:
                class_boxes.append(boxes[i])
                class_scores.append(scores[i])

        return class_boxes, class_scores
    
    def post_processing(self, boxes: list, scores: list, ground_truth: list):
        '''
        Post-processing function to do filtering, non-max suppression and mAP calculation.
        Inputs:
            boxes: list of boxes in the format [xmin, ymin, xmax, ymax]
            scores: list of scores for each box for positive class (pothole present)
            ground_truth: list of ground truth boxes in the format [xmin, ymin, xmax, ymax]
        Outputs:
            boxes: list of boxes after non-max suppression in the format [xmin, ymin, xmax, ymax]
            scores: list of scores after non-max suppression for positive class (pothole present)
            mAP: mean Average Precision
        '''
        boxes, scores = self.filter_output(boxes, scores)
        boxes, scores = self.non_max_suppression(boxes, scores)
        mAP = self.mAP(boxes, scores, ground_truth)
        return boxes, scores, mAP
    
    def plot_image_with_boxes(self, image_tensor, true_boxes, proposed_boxes, proposed_scores, proposed_ious, save_path):
        # Convert the tensor image to a numpy array for plotting
        image = image_tensor.cpu().numpy()  # Convert (C, H, W) to (H, W, C)
        
        # Create a figure and axis for plotting
        fig, ax = plt.subplots(1, figsize=(10, 10))
        ax.imshow(image)
        
        # Plot ground truth boxes (in green)
        for box in true_boxes:
            xmin, ymin, xmax, ymax = box
            rect = patches.Rectangle((xmin, ymin), xmax - xmin, ymax - ymin, linewidth=2, edgecolor='g', facecolor='none')
            ax.add_patch(rect)

        # Plot proposed boxes (in random colors)
        for idx, box in enumerate(proposed_boxes):
            xmin, ymin, xmax, ymax = box
            
            # Generate a random color, avoiding green
            color = (random.random(), random.random(), random.random())
            while color[1] > 0.5 and color[0] < 0.5 and color[2] < 0.5:  # Avoid colors close to green
                color = (random.random(), random.random(), random.random())
            
            # Draw the proposed box
            rect = patches.Rectangle((xmin, ymin), xmax - xmin, ymax - ymin, linewidth=2, edgecolor=color, facecolor='none')
            ax.add_patch(rect)
            
            # Display the score on the top-left corner of the box
            ax.text(xmin, ymin, f'{proposed_scores[idx]:.2f}, iou: {proposed_ious[idx]:.2f}', color=color, fontsize=10, verticalalignment='top', fontweight='bold')

        # Set the title with the image index
        ax.set_title(f"Model prediction:")
        plt.axis('off')  # Turn off axis 

        # Save the plot to the specified directory
        plt.savefig(save_path, bbox_inches='tight')
        plt.close()

if __name__ == "__main__":
    # Test the Evaluation class
    boxes = [np.array([0, 0, 10, 10]), 
             np.array([1, 0, 12, 10]),
             np.array([0, 2, 12, 11]),
             np.array([2, 0, 11, 10]), 
             np.array([0, 1, 11, 9]),
             np.array([0, 4, 11, 13])]
    scores = [0.9, 0.75, 0.8, 0.6, 0.7, 0.3]

    ground_truth = [np.array([0, 0, 10, 10])]

    eval = Evaluation(nms_iou_threshold=0.7, map_iou_threshold=0.5, score_threshold=0.5)

    # Test filter_output
    boxes, scores = eval.filter_output(boxes, scores)
    print(f"Boxes after filtering: {boxes}")
    print(f"Scores after filtering: {scores}")

    # Test non-max suppression
    boxes, scores = eval.non_max_suppression(boxes, scores)
    print(f"Boxes after non-max suppression: {boxes}")
    print(f"Scores after non-max suppression: {scores}")

    # Test mAP
    mAP, precision, recall = eval.mAP(boxes, scores, ground_truth)
    print(f"mAP: {mAP}")
    print(f"Precision: {precision[-1]}")
    print(f"Recall: {recall[-1]}")

    # Test plot_precision_recall_curve
    eval.plot_precision_recall_curve(precision, recall, path=os.path.join(os.getcwd(),'precision_recall_curve.png'), title='Precision-Recall Curve (IoU=0.5)', mAP=mAP)