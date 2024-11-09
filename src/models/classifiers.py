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


class ClassifierAlexNet64(nn.Module):
    
    def __init__(self):
        """
        The input to the model needs to be [batch_size, 3, 64, 64],
        since AlexNet takes that as the input.
        
        The forward pass returns one logit, so BCE should be used.
        """
        super().__init__()
        
        self.model = models.alexnet(weights=models.AlexNet_Weights.IMAGENET1K_V1)
        
        # Freeze earlier layers
        for param in self.model.features.parameters():
            param.requires_grad = False
        
        # Adjust the classifier for background and pothole
        self.model.classifier = nn.Sequential(
                nn.Dropout(),
                nn.Linear(9216, 4096),
                nn.ReLU(inplace=True),
                nn.Dropout(),
                nn.Linear(4096, 4096),
                nn.ReLU(inplace=True),
                nn.Linear(4096, 1),
            )
    
    
    def forward(self, x):
        assert x.shape[1:] == (3, 64, 64), f"Expected input shape [batch_size, 3, 224, 224], but got {x.shape}"
        
        return self.model(x)