from torchvision import models
import torch.nn as nn


class ClassifierAlexNet(nn.Module):
    
    def __init__(self):
        """
        The input to the model needs to be [batch_size, 3, 224, 224],
        since AlexNet takes that as the input.
        
        The forward pass returns one logit, so BCE should be used.
        """
        super().__init__()
        
        self.model = models.alexnet(weights=models.AlexNet_Weights.DEFAULT)
        
        # Freeze earlier layers
        for param in self.model.features.parameters():
            param.requires_grad = False
        
        # Adjust the classifier for background and pothole
        self.model.classifier[6] = nn.Linear(4096, 1)
    
    
    def forward(self, x):
        assert x.shape[1:] == (3, 224, 224), f"Expected input shape [batch_size, 3, 224, 224], but got {x.shape}"
        
        return self.model(x)
