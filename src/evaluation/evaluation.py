import torch
import numpy as np

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
                if self.iou(boxes[base_idx], boxes[comp_idx]) > self.nms_iou_threshold:
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

    def get_union(self, area1: list, area2: list) -> float:
        '''
        taken from Alex part
        '''
        xmin_1, ymin_1, xmax_1, ymax_1 = area1
        xmin_2, ymin_2, xmax_2, ymax_2 = area2
        
        area_1 = (xmax_1 - xmin_1 + 1) * (ymax_1 - ymin_1 + 1)
        area_2 = (xmax_2 - xmin_2 + 1) * (ymax_2 - ymin_2 + 1)
        
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

        # go through all boxes and compare with ground truths
        for box_id in range(len(boxes)):
            # calculate iou with all ground truths
            for gt_id in range(len(ground_truth)):
                if gt_mask[gt_id] == 0:
                    if self.iou(boxes[box_id], ground_truth[gt_id]) > self.map_iou_threshold:
                        gt_mask[gt_id] = 1
                        box_mask[box_id] = 1

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
        
        return ap
    
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
    mAP = eval.mAP(boxes, scores, ground_truth)
    print(f"mAP: {mAP}")