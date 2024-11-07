import sys
sys.path.append("..")
import os
import json
import xml.etree.ElementTree as ET
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as transforms
import torch
from .custom_transforms import base_transform
from tqdm import tqdm
import torchvision
from torchvision.utils import draw_bounding_boxes, save_image
import cv2

PROJECT_BASE_DIR = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
)
DATA_DIR = os.path.join(PROJECT_BASE_DIR, "data/raw/Potholes/annotated-images")
SPLITS_FILE = os.path.join(PROJECT_BASE_DIR, "data/raw/Potholes/splits.json")
VISUALIZATION_DIR = os.path.join(PROJECT_BASE_DIR, "data/visualized_bounding_boxes")



class PotholeDataSet(Dataset):
    def __init__(self, train: bool, transform=transforms.ToTensor(), data_path=DATA_DIR, splits_file=SPLITS_FILE):
        self.directory = data_path
        self.transform = transform
        self.splits_file = splits_file
        self.train = train
        # list of XML file paths
        self.xml_paths = self.get_xml_paths()
        # list of image paths
        self.image_paths = self.get_image_paths()
        # dictinary, key: image name, value: bounding boxes- list of lists
        self.all_bounding_boxes = self.process_xml_files()
        self.output_directory = os.path.join(VISUALIZATION_DIR, "train" if train else "test")

    def get_xml_paths(self):
        """
        Gets the list of XML files based on the train/test split.
        """
        with open(self.splits_file, 'r') as f:
            splits = json.load(f)
        split_paths = splits['train'] if self.train else splits['test']
        return [os.path.join(self.directory, path) for path in split_paths]
    

    def process_xml_files(self):
        """
        Processes XML files in the specified directory and extracts bounding boxes.
        """
        all_bounding_boxes = {}
        for file_path in self.xml_paths:
            name, boxes = self.read_xml_file(file_path)
            all_bounding_boxes[name] = boxes

        return all_bounding_boxes
    
    def read_xml_file(self, xml_file: str):
        """
        Reads the content of a single XML file and extracts bounding boxes along with image dimensions.
        """
        tree = ET.parse(xml_file)
        root = tree.getroot()

        list_with_all_boxes = []

        for boxes in root.iter('object'):
            path = root.find('path').text
            filename = os.path.basename(path)

            ymin = int(boxes.find("bndbox/ymin").text)
            xmin = int(boxes.find("bndbox/xmin").text)
            ymax = int(boxes.find("bndbox/ymax").text)
            xmax = int(boxes.find("bndbox/xmax").text)

            list_with_single_boxes = [xmin, ymin, xmax, ymax]
            assert len(list_with_single_boxes) == 4
            list_with_all_boxes.append(list_with_single_boxes)

        boxes_tensor = torch.tensor(list_with_all_boxes, dtype=torch.int32)
        return filename, boxes_tensor
        

    def get_image_paths(self):
        """
        Gets the list of image paths based on the XML files.
        """
        image_paths = []
        for xml_path in self.xml_paths:
            tree = ET.parse(xml_path)
            root = tree.getroot()
            path = root.find('path').text
            image_name = os.path.basename(path)
            image_path = os.path.join(DATA_DIR, image_name)
            image_paths.append(image_path)
        return image_paths

    def draw_bounding_boxes(self, image_path, boxes):
        """
        Visualizes bounding boxes on a single image and saves the result.
        """
        image = torchvision.io.read_image(image_path)
        image = draw_bounding_boxes(image, boxes)
        image_name = os.path.basename(image_path)
        output_path = os.path.join(self.output_directory, image_name)
        torchvision.io.write_png(image, output_path)


    def visualize_selected_images_by_filename(self, image_indexes_in_filenames):
        """
        Visualizes bounding boxes on selected images based on their filenames.
        """
        for image_index in image_indexes_in_filenames:
            filename = f"image-{image_index}.jpg"  # Assuming the filenames follow this pattern
            image_path = os.path.join(self.directory, filename)
            if os.path.exists(image_path):
                boxes = self.all_bounding_boxes[filename]
                self.draw_bounding_boxes(image_path, boxes)

    def visualize_selected_images_by_array_index(self, indexes):
        """
        Visualizes bounding boxes on selected images based on array indexes.
        """
        for index in indexes:
            if index < len(self.image_paths):
                image_path = self.image_paths[index]
                boxes = self.all_bounding_boxes[os.path.basename(image_path)]
                self.draw_bounding_boxes(image_path, boxes)

    def visualize_all_images(self):
        """
        Visualizes bounding boxes on all images.
        """
        for i, image_path in enumerate(self.image_paths):
            boxes = self.all_bounding_boxes[os.path.basename(image_path)]
            self.draw_bounding_boxes(image_path, boxes)

    def remove_all_visualized_images(self):
        """
        Removes all images from the specified directory.
        """
        for filename in os.listdir(self.output_directory):
            file_path = os.path.join(self.output_directory, filename)
            if os.path.isfile(file_path):
                os.remove(file_path)
                print(f"Removed file: {file_path}")

    def get_train_test_indices(self):
        """
        Returns the indices of the train and test splits in increasing order.
        """
        with open(self.splits_file, 'r') as f:
            splits = json.load(f)
        train_indices = sorted([int(os.path.splitext(os.path.basename(path))[0].split('-')[1]) for path in splits['train']])
        test_indices = sorted([int(os.path.splitext(os.path.basename(path))[0].split('-')[1]) for path in splits['test']])
        return train_indices, test_indices
    
    def __len__(self):
        assert len(self.image_paths) == len(self.xml_paths)
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        image = cv2.imread(image_path)
        # image = torch.from_numpy(image).permute(2, 0, 1).float()  # Convert to torch.Tensor
        #image = torchvision.io.read_image(image_path)
        boxes = self.all_bounding_boxes[os.path.basename(image_path)]
        # boxes = torch.tensor(boxes, dtype=torch.float32)

        if self.transform:
            image, boxes = self.transform(image, boxes)
        return (image, boxes)

class PotholeDataModule:
    def __init__(
        self,
        data_path=DATA_DIR,
        batch_size: int = 16,
        train_transform=base_transform(size=256),
        test_transform=base_transform(size=256)
    ):
        self.batch_size = batch_size
        self.data_path = data_path
        self.train_dataset = PotholeDataSet(
            train=True, transform=train_transform, data_path=data_path
        )
        self.test_dataset = PotholeDataSet(
            train=False, transform=test_transform, data_path=data_path
        )
    
    def train_dataloader(self, shuffle=False) -> DataLoader:
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=shuffle,
            num_workers=0,
            collate_fn=self.collate_fn
        )

    def test_dataloader(self, shuffle=False) -> DataLoader:
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=shuffle,
            num_workers=0,
            collate_fn=self.collate_fn
        )
    
    def collate_fn(self,batch):
        images = []
        bounding_boxes = []
        for img, boxes in batch:
            images.append(img)
            bounding_boxes.append(boxes)
        # Stack images as usual; bounding_boxes remains a list of varying-size tensors
        images = torch.stack(images)
        return images, bounding_boxes
    
    def get_training_examples(self):
        images, boxes = next(iter(self.train_dataloader()))
        return images, boxes

    def get_test_examples(self):
        images, boxes = next(iter(self.test_dataloader()))
        return images, boxes

    def visualize_selected_train_images_by_array_index(self, indexes):
        """
        Wrapper for visualizing bounding boxes on selected images based on array indexes in the training dataset.
        """
        self.train_dataset.visualize_selected_images_by_array_index(indexes)

    def visualize_all_train_images(self):
        """
        Wrapper for visualizing bounding boxes on all images in the training dataset.
        """
        self.train_dataset.visualize_all_images()

    def remove_all_visualized_train_images(self):
        """
        Wrapper for removing all visualized images in the training dataset.
        """
        self.train_dataset.remove_all_visualized_images()

    def visualize_selected_test_images_by_array_index(self, indexes):
        """
        Wrapper for visualizing bounding boxes on selected images based on array indexes in the test dataset.
        """
        self.test_dataset.visualize_selected_images_by_array_index(indexes)

    def visualize_all_test_images(self):
        """
        Wrapper for visualizing bounding boxes on all images in the test dataset.
        """
        self.test_dataset.visualize_all_images()

    def remove_all_visualized_test_images(self):
        """
        Wrapper for removing all visualized images in the test dataset.
        """
        self.test_dataset.remove_all_visualized_images()

    def __repr__(self):
        return (
            f"Pothole DataModule with batch size {self.batch_size}\n"
            f" Training dataset: {len(self.train_dataset)} samples\n"
            f" Test dataset: {len(self.test_dataset)} samples"
        )
    
    def get_trainset_size(self):
        return len(self.train_dataset)

    def get_testset_size(self):
        return len(self.test_dataset)
    
def test_data_module(data_module:PotholeDataModule):
    data_module.remove_all_visualized_train_images()
    data_module.remove_all_visualized_test_images()
    train_loader = data_module.train_dataloader()
    test_loader = data_module.test_dataloader()

    # draw bounding boxes on the first minibatch of the train and test loaders
    for minibatch_no, (image, target_boxes) in tqdm(enumerate(train_loader), total=len(train_loader)):
        print(f"Minibatch {minibatch_no}:")
        print(f"Data shape: {image.shape}")
        output_directory=os.path.join(VISUALIZATION_DIR, "train")
        image= draw_bounding_boxes(image[0], target_boxes[0])
        output_path = os.path.join(output_directory, f"minibatch_{minibatch_no}.jpg")
        save_image(image, output_path)

    for minibatch_no, (image, target_boxes) in tqdm(enumerate(test_loader), total=len(test_loader)):
        print(f"Minibatch {minibatch_no}:")
        print(f"Data shape: {image.shape}")
        output_directory=os.path.join(VISUALIZATION_DIR, "test")
        image= draw_bounding_boxes(image[0], target_boxes[0])
        output_path = os.path.join(output_directory, f"minibatch_{minibatch_no}.jpg")
        save_image(image, output_path)

    # Call the visualizer functions
    data_module.visualize_selected_test_images_by_array_index([0, 8])
    data_module.visualize_selected_train_images_by_array_index([0, 8])
    data_module.train_dataset.visualize_selected_images_by_filename([1, 577])

# Example usage
if __name__ == "__main__":
    data_module = PotholeDataModule(batch_size=8)
    test_data_module(data_module)

