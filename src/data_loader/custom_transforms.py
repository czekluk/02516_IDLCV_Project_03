import torch
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF
from PIL import Image
import numpy as np
import cv2
import random

class CroppedTransform(torch.nn.Module):
    def __init__(self, size: int = 512):
        super().__init__()
        self.size = size
        self.transform = transforms.Compose([
            transforms.Resize((size, size)),
            transforms.ToTensor()
        ])

    def forward(self, img: torch.Tensor):
        """
        Apply the same base transformations (resize, to tensor) to both the image and the bounding boxes.
        """
        # Transform the image
        pil_img = TF.to_pil_image(img)
         # Transform the bounding boxes using PIL affine transformation
        img = self.transform(pil_img)
        return img


    def __repr__(self):
        return f"{self.__class__.__name__}(size={self.size})"
    
def cropped_transform(size: int = 512):
    return CroppedTransform(size=size)