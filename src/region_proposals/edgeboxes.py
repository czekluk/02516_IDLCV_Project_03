# Needs the opencv-contrib-python package. If you have opencv-python, uninstall that first, then install the new one
# https://stackoverflow.com/questions/57427233/module-cv2-cv2-has-no-attribute-ximgproc
import cv2
import matplotlib.pyplot as plt
import os
import numpy as np
import xml.etree.ElementTree as ET
from tqdm import tqdm
import random


PROJECT_BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath('')))
XIMGPROC_MODEL = os.path.join(os.path.abspath(''), 'ximgproc_model.yml.gz')
RAW_IMG_DIR = os.path.join(PROJECT_BASE_DIR, 'data', 'raw', 'Potholes', 'annotated-images')

#---------------------------------#
# HELPER FUNCTIONS
#---------------------------------#
def load_bbox(xml_file):
    """Loads the bounding boxes from the given xml file

    Args:
        xml_file: Path to the xml file containing the bounding boxes

    Returns:
        gt_boxes: List of bounding boxes in the format [xmin, ymin, xmax, ymax]
    """
    gt_boxes = []
    tree = ET.parse(xml_file)
    root = tree.getroot()
    for items in root.findall('object/bndbox'):
        xmin = items.find('xmin')
        ymin = items.find('ymin')
        xmax = items.find('xmax')
        ymax = items.find('ymax')
        gt_boxes.append([int(xmin.text), int(ymin.text), int(xmax.text), int(ymax.text)])
    return gt_boxes

def load_image_and_bboxes (img_dir, img_names):
    """Reads the images and the corresponding bounding boxes from the given directory

    Args:
        img_dir: Path to the directory containing the images
        img_names: List of image names to be loaded along with their bounding boxes

    Returns:
        images, boxes: List of images and list of corresponding bounding boxes
    """
    images = [] # List (batch_size) of images
    bboxes = [] # List (batch_size) of lists (bboxes per image) of lists (xmin, ymin, xmax, ymax)
    for image in img_names:
        imgfile = os.path.join(img_dir, image) + '.jpg'
        bboxfile = os.path.join(img_dir, image) + '.xml'
        images.append(cv2.imread(imgfile))
        bboxes.append(load_bbox(bboxfile))
    return images, bboxes

def compare_images(img1, img2):
    """Compares two images and displays them side by side

    Args:
        img1: First image to be compared
        img2: Second image to be compared
    """
    fig, ax = plt.subplots(1, 2, figsize=(16, 8))
    ax[0].imshow(img1)
    ax[0].axis('off')
    ax[1].imshow(img2, cmap='gray')
    ax[1].axis('off')
    plt.show()
    
#---------------------------------#
# EDGEBOXES
#---------------------------------#
class EdgeBoxesProposer:
    def __init__(self, edge_detection_model_path, edgebox_params = None):
        # OpenCV model taken from https://github.com/opencv/opencv_extra/blob/master/testdata/cv/ximgproc/model.yml.gz
        self.edge_detection = cv2.ximgproc.createStructuredEdgeDetection(edge_detection_model_path)
        self.edge_boxes = cv2.ximgproc.createEdgeBoxes()
        if edgebox_params:
            self.set_edgebox_params(edgebox_params)
        
    def set_edgebox_params(self, edgebox_params):
        # Set the parameters for EdgeBoxes
        # https://docs.opencv.org/3.4/d4/d0d/group__ximgproc__edgeboxes.html
        self.edge_boxes.setAlpha(edgebox_params['alpha']) if 'alpha' in edgebox_params else None
        self.edge_boxes.setBeta(edgebox_params['beta']) if 'beta' in edgebox_params else None
        self.edge_boxes.setEta(edgebox_params['eta']) if 'eta' in edgebox_params else None
        self.edge_boxes.setMinScore(edgebox_params['min_score']) if 'min_score' in edgebox_params else None
        self.edge_boxes.setMaxBoxes(edgebox_params['max_boxes']) if 'max_boxes' in edgebox_params else None
        self.edge_boxes.setEdgeMinMag(edgebox_params['edge_min_mag']) if 'edge_min_mag' in edgebox_params else None
        self.edge_boxes.setEdgeMergeThr(edgebox_params['edge_merge_thr']) if 'edge_merge_thr' in edgebox_params else None
        self.edge_boxes.setClusterMinMag(edgebox_params['cluster_min_mag']) if 'cluster_min_mag' in edgebox_params else None
        self.edge_boxes.setMaxAspectRatio(edgebox_params['max_aspect_ratio']) if 'max_aspect_ratio' in edgebox_params else None
        self.edge_boxes.setMinBoxArea(edgebox_params['min_box_area']) if 'min_box_area' in edgebox_params else None
        self.edge_boxes.setGamma(edgebox_params['gamma']) if 'gamma' in edgebox_params else None
        self.edge_boxes.setKappa(edgebox_params['kappa']) if 'kappa' in edgebox_params else None
    
    def detect_edges(self, image):
        """Detects edges in the given image using the Structured Edge Detection model
        https://docs.opencv.org/3.4/d0/da5/tutorial_ximgproc_prediction.html

        Args:
            image: Input image in BGR format

        Returns:
            edges: Image with edges detected
            orimap: Orientation map of the edges
        """
        rgb_im = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        if (rgb_im.max() < 1.01):
            # Image is already normalized
            edges = self.edge_detection.detectEdges(np.float32(rgb_im))
        else:
            edges = self.edge_detection.detectEdges(np.float32(rgb_im) / 255.0)
        orimap = self.edge_detection.computeOrientation(edges)
        edges = self.edge_detection.edgesNms(edges, orimap)
        return edges, orimap
    
    def detect_edges_canny(self, image):
        """Detects edges in the given image using the Canny edge detection algorithm

        Args:
            image: Input image in BGR format

        Returns:
            edges: Image with edges detected
        """
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 100, 200)
        return edges
    
    def get_proposals(self, image):
        """Get the bounding box proposals for the given image. Parameters can be set via
        the set_edgebox_params() method.

        Args:
            image: Input image in BGR format

        Returns:
            boxes: List of bounding boxes in the format [xmin, ymin, xmax, ymax]
            scores: List of scores corresponding to each bounding box
        """
        edges, orimap = self.detect_edges(image)
        prep_boxes, scores = self.edge_boxes.getBoundingBoxes(edges, orimap)
        # Convert the boxes from [x, y, w, h] to [xmin, ymin, width, height]
        boxes = [[box[0], box[1], box[0] + box[2], box[1] + box[3]] for box in prep_boxes]
        return boxes, scores
        
    def get_intersection(self, box1, box2):
        """Calculates the intersection area of two bounding boxes

        Args:
            box1: List of the format [xmin, ymin, xmax, ymax]
            box2: List of the format [xmin, ymin, xmax, ymax]

        Returns:
            intersection: Intersection area of the two bounding boxes
        """
        xmin_1, ymin_1, xmax_1, ymax_1 = box1
        xmin_2, ymin_2, xmax_2, ymax_2 = box2
        
        x_left = max(xmin_1, xmin_2) # Leftmost x-coordinate of the intersection
        y_top = max(ymin_1, ymin_2) # Topmost y-coordinate of the intersection
        x_right = min(xmax_1, xmax_2) # Rightmost x-coordinate of the intersection
        y_bottom = min(ymax_1, ymax_2) # Bottommost y-coordinate of the intersection
        
        if x_right < x_left or y_bottom < y_top:
            return 0
        else:
            return (x_right - x_left + 1) * (y_bottom - y_top + 1)

        
    def get_union(self, box1, box2):
        """Calculates the union area of two bounding boxes

        Args:
            box1: List of the format [xmin, ymin, xmax, ymax]
            box2: List of the format [xmin, ymin, xmax, ymax]

        Returns:
            iou: Union area of the two bounding boxes
        """
        xmin_1, ymin_1, xmax_1, ymax_1 = box1
        xmin_2, ymin_2, xmax_2, ymax_2 = box2
        
        area_1 = (xmax_1 - xmin_1 + 1) * (ymax_1 - ymin_1 + 1)
        area_2 = (xmax_2 - xmin_2 + 1) * (ymax_2 - ymin_2 + 1)
        
        return area_1 + area_2 - self.get_intersection(box1, box2)


    def get_iou(self, box1, box2):
        """Calculates the Intersection over Union (IoU) value for two bounding boxes

        Args:
            box1: List of the format [xmin, ymin, xmax, ymax]
            box2: List of the format [xmin, ymin, xmax, ymax]

        Returns:
            iou: Intersection over Union (IoU) value (float between 0.0 and 1.0)
        """
        intersection = self.get_intersection(box1, box2)
        union = self.get_union(box1, box2)
        iou = float(intersection) / float(union)
        return iou
    
    def filter_by_iou_threshold(self, prop_boxes, gt_boxes, iou_threshold = 0.5):
        """Filters the bounding boxes based on the IoU threshold

        Args:
            boxes: List of bounding boxes in the format [xmin, ymin, xmax, ymax]
            iou_threshold: Threshold value for filtering the bounding boxes
        
        Returns:
            qualified_boxes: List of bounding boxes that meet the IoU threshold
            best_boxes: List of bounding boxes with the highest IoU value for a given ground truth box
        """
        qualified_boxes = []
        best_boxes = []
        best_ious = []
        for gt_box in gt_boxes:
            best_iou = 0.0
            best_box = None
            for prop_box in prop_boxes:
                iou = self.get_iou(gt_box, prop_box)
                if iou >= iou_threshold:
                    qualified_boxes.append(prop_box)
                    if iou > best_iou:
                        best_iou = iou
                        best_box = prop_box
            if best_box:
                best_boxes.append(best_box)
                best_ious.append(best_iou)
        return qualified_boxes, best_boxes, best_ious
    
    def get_metrics(self, prop_boxes, qualified_boxes, best_boxes, best_ious, gt_boxes, print_output=False):
        """Prints the metrics for the object proposal.

        Args:
            boxes: List of bounding boxes in the format [xmin, ymin, xmax, ymax]
            qualified_boxes: List of bounding boxes that meet the IoU threshold
            best_boxes: List of bounding boxes with the highest IoU value for a given ground truth box
            gt_boxes: List of ground truth bounding boxes
            
        Returns:
            qualified_pct: % of boxes that meet the IoU threshold
            recall: % of ground truth boxes that have a match
            mabo: MABO (Mean Average Best Overlap)
        """
        # % of boxes that meet the IoU threshold
        qualified_pct = len(qualified_boxes) / len(prop_boxes) 
        
        # % of ground truth boxes that have a match
        recall = len(best_boxes) / len(gt_boxes)
        
        # MABO (Mean Average Best Overlap)
        mabo = np.sum(best_ious) / len(gt_boxes)

        if print_output:
            print('Total Proposed Boxes:', len(prop_boxes))
            print(f'Qualified Ratio: {len(qualified_boxes) / len(prop_boxes):.3f}')
            print(f'Recall: {recall:.3f}')
            print(f'MABO: {mabo:.3f}')
        
        return qualified_pct, recall, mabo
    
    def evaluate_dataset(self, images, bboxes):
        """Creates and evaluates the object proposals for the dataset. Prints metrics for comparison.

        Args:
            images: List of images
            bboxes: List of lists of bounding boxes for each image

        Returns:
            qualified_pct: % of boxes that meet the IoU threshold
            recall: % of ground truth boxes that have a match
            mabo: MABO (Mean Average Best Overlap)
        """
        print('Evaluating dataset...')
        qualified_pcts = []
        recalls = []
        mabos = []
        for i, image in tqdm(enumerate(images), total=len(images)):
            boxes, scores = self.get_proposals(image)
            if len(boxes) == 0:
                print(f'WARNING: No boxes found for image {i} (likely a bug)')
                continue
            qualified_boxes, best_boxes, best_ious = self.filter_by_iou_threshold(boxes, bboxes[i])
            qualified_pct, recall, mabo = self.get_metrics(boxes, qualified_boxes, best_boxes, best_ious, bboxes[i])
            qualified_pcts.append(qualified_pct)
            recalls.append(recall)
            mabos.append(mabo)
        
        total_qualified = np.mean(qualified_pcts)
        total_recall = np.mean(recalls)
        total_mabo = np.mean(mabos)
        print(f'Qualified Ratio: {total_qualified:.3f}')
        print(f'Recall: {total_recall:.3f}')
        print(f'MABO: {total_mabo:.3f}')
        return total_qualified, total_recall, total_mabo

    def plot_bboxes(self, image, prop_boxes = None, qualified_boxes = None, best_boxes = None, gt_boxes = None):
        """Plots the bounding boxes on the image. Different colors are used for different types of boxes."""
        im = image.copy()
        prop_color = (255, 255, 255) # Color of the proposed boxes
        qualified_color = (0, 255, 0) # Color of the qualified boxes
        best_color = (255, 0, 0) # Color of the best boxes
        gt_color = (0, 0, 255) # Color of the ground truth boxes
        
        if prop_boxes:
            for b in prop_boxes:
                xmin, ymin, xmax, ymax = b
                cv2.rectangle(im, (xmin, ymin), (xmax, ymax), prop_color, 1, cv2.LINE_AA)
        
        if qualified_boxes:
            for b in qualified_boxes:
                xmin, ymin, xmax, ymax = b
                cv2.rectangle(im, (xmin, ymin), (xmax, ymax), qualified_color, 2, cv2.LINE_AA)
                
        if best_boxes:
            for b in best_boxes:
                xmin, ymin, xmax, ymax = b
                cv2.rectangle(im, (xmin, ymin), (xmax, ymax), best_color, 2, cv2.LINE_AA)
                
        if gt_boxes:
            for b in gt_boxes:
                xmin, ymin, xmax, ymax = b
                cv2.rectangle(im, (xmin, ymin), (xmax, ymax), gt_color, 3, cv2.LINE_AA)
                
        # Add color legend
        f = plt.figure(figsize=(7, 7))
        plt.plot(0, 0, "-", color=[c/255 - 0.1 for c in prop_color], label="Proposed Objects")
        plt.plot(0, 0, "-", color=[c/255 for c in qualified_color], label="Qualified Objects (IoU > Threshold)")
        plt.plot(0, 0, "-", color=[c/255 for c in best_color], label="Best Matches")
        plt.plot(0, 0, "-", color=[c/255 for c in gt_color], label="Ground Truth Objects")
        f.legend(loc='lower center', ncol=4, fontsize=10, bbox_to_anchor=(0.5, -0.05))
        plt.axis('off')
        plt.tight_layout()
        plt.imshow(im)
    
    def crop_image_to_bbox(self, image, bbox):
        """Crops the image content to the bounding box"""
        xmin, ymin, xmax, ymax = bbox
        return image[ymin:ymax, xmin:xmax]
    
    def get_n_proposals_train(self, image, gt_bbox, n = 50, iou_threshold = 0.5, positive_class_ratio = 0.3):
        """
        Get n proposals for training the object detection model. The proposals are selected based on the IoU threshold and the positive class ratio.
        """
        # Get the bounding box proposals and filter them based on the IoU threshold
        boxes, scores = self.get_proposals(image)
        qualified_boxes, best_boxes, _ = self.filter_by_iou_threshold(boxes, gt_bbox, iou_threshold)
        
        # If there are no qualified boxes, return the top n boxes
        if len(qualified_boxes) == 0:
            return boxes[:n], scores[:n]
        
        # Remove best boxes from the list of qualified boxes
        qualified_boxes = [box for box in qualified_boxes if box not in best_boxes]
        
        # Add the best boxes to the final list
        final_boxes = []
        for box in best_boxes:
            crop = self.crop_image_to_bbox(image, box)
            final_boxes.append((crop, 1))
        
        # Populate list with qualified boxes until the positive class ratio is met
        while len(final_boxes) / n < positive_class_ratio:
            box = random.choice(qualified_boxes)
            crop = self.crop_image_to_bbox(image, box)
            final_boxes.append((crop, 1))
        
        # Populate the rest of the list with random boxes
        while len(final_boxes) < n:
            box = random.choice(boxes)
            crop = self.crop_image_to_bbox(image, box)
            final_boxes.append((crop, 0))

        # Shuffle the final list
        random.shuffle(final_boxes)

        # Extract images and labels
        images = [box[0] for box in final_boxes]
        labels = [box[1] for box in final_boxes]

        return images, labels
    
    def get_n_proposals_test(self, image, n = 50):
        """
        Get n proposals for validating the object detection model, without any filtering.
        """
        # Get the bounding box proposals
        boxes, scores = self.get_proposals(image)
        final_boxes = []
        
        # If there are fewer than n boxes, return all the boxes
        for i, box in enumerate(boxes):
            if i == n:
                break
            crop = self.crop_image_to_bbox(image, box)
            final_boxes.append((crop, 1))
        return final_boxes

if __name__ == '__main__':
    # Load entire dataset
    images_to_load = ['img-' + str(i) for i in range(1, 666)]
    images, gt_bboxes = load_image_and_bboxes(RAW_IMG_DIR, images_to_load)
    
    # Evaluate dataset using params
    # The more boxes, the more likely to have high MABO and recall
    # TODO: Tune the parameters for better results
    # https://docs.opencv.org/3.4/d4/d0d/group__ximgproc__edgeboxes.html
    edgebox_params = {
        'max_boxes': 1000,
    }
    eb = EdgeBoxesProposer(XIMGPROC_MODEL, edgebox_params)
    # eb.evaluate_dataset(images, gt_bboxes)

    # Plot random image with bounding boxes
    index = random.randint(0, len(images) - 1)
    print('-' * 50)
    print(f'Plotting img-{index + 1}.jpg with bounding boxes...')
    img_to_plot = images[index]
    gt_bbox = gt_bboxes[index]
    boxes, scores = eb.get_proposals(img_to_plot)
    qualified_boxes, best_boxes, best_ious = eb.filter_by_iou_threshold(boxes, gt_bbox)
    qualified_pct, recall, mabo = eb.get_metrics(boxes, qualified_boxes, best_boxes, best_ious, gt_bbox, print_output=True)
    eb.plot_bboxes(img_to_plot, boxes[:20], qualified_boxes, best_boxes, gt_bbox) # Plot only the top 20 boxes for visibility
    
    # Get n proposals for training - image, label (50 proposals, 30% positive class ratio)
    n_train = 50
    print('-' * 50)
    print(f'Getting {n_train} proposals for training...')
    image, label = eb.get_n_proposals_train(images[0], gt_bboxes[0], n=n_train, iou_threshold=0.5, positive_class_ratio=0.3)
    print('Number of proposals:', len(image))
    print('Labels:', label)
    
    # Get n proposals for testing - image ONLY (50 proposals)
    n_test = 50
    print('-' * 50)
    print(f'Getting {n_test} proposals for testing...')
    test_images = eb.get_n_proposals_test(images[0], n=n_test)
    print('Number of proposals:', len(test_images))
