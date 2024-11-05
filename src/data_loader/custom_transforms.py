import torch
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF
from PIL import Image
import numpy as np
import cv2
import random

class JointBaseTransform(torch.nn.Module):
    def __init__(self, size: int = 512):
        super().__init__()
        self.size = size
        self.transform = transforms.Compose([
            transforms.Resize((size, size)),
            transforms.ToTensor()
        ])

    def forward(self, img: torch.Tensor, boxes: torch.Tensor):
        """
        Apply the same base transformations (resize, to tensor) to both the image and the bounding boxes.
        """
        # Get original image size
        original_size = img.shape[1], img.shape[2]  # (height, width)

        # Transform the image
        pil_img = TF.to_pil_image(img)
         # Transform the bounding boxes using PIL affine transformation
        img = self.transform(pil_img)
        # Transform the bounding boxes
        out_boxes = self.transform_boxes(boxes, original_size)
        return img, out_boxes

    def transform_boxes(self, boxes, original_size):
        """
        Resize bounding boxes to match the new image size using affine transformation.
        """
        orig_height, orig_width  = original_size
        new_height, new_width,  = self.size, self.size

        # Scale factors
        x_scale = new_width / orig_width
        y_scale = new_height / orig_height
        out_boxes = torch.zeros_like(boxes)
        # Transform the bounding boxes
        out_boxes[:, [0, 2]] = (boxes[:, [0, 2]] * x_scale).round().int()
        out_boxes[:, [1, 3]] = (boxes[:, [1, 3]] * y_scale).round().int()

        return out_boxes

    def __repr__(self):
        return f"{self.__class__.__name__}(size={self.size})"