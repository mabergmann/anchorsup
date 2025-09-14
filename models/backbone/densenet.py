"""A DenseNet backbone."""
import torch
import torchvision
from torch import nn
import torch.nn.functional as F
import numpy as np
import math

from upright_anchor.models.backbone.base import IBackbone


class DenseNetBackbone(IBackbone):
    """A DenseNet backbone for the anchor generation service.

    DenseNet is a convolutional neural network architecture that connects each layer to every other layer in a
    feed-forward fashion.
    """

    def __init__(self, image_size: tuple[int, int], *, pretrained: bool = True) -> None:
        """Initialize a new instance of DenseNetBackbone.

        Args:
            image_size (tuple[int, int]): The size of the input image.
            pretrained (bool, optional): Whether to use a pretrained model. Defaults to True.
        """
        super().__init__(image_size, pretrained=pretrained)

        if pretrained:
            densenet = torchvision.models.densenet121(weights=torchvision.models.DenseNet121_Weights.IMAGENET1K_V1)
        else:
            densenet = torchvision.models.densenet121(weights=None)

        self.features = densenet.features
        self.n_features = densenet.classifier.in_features
        self.image_size = image_size
        

        torch.randn(1, 3, self.image_size[0], self.image_size[1])
        self.output_size = self.forward(torch.randn(1, 3, self.image_size[0], self.image_size[1])).shape[1]
        
        

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass of the DenseNet backbone.

        Args:
            x (torch.Tensor): The input tensor.

        Returns:
            torch.Tensor: The output tensor.
        """
        x = self.features(x)
       

        return x.view(x.size(0), -1)

    def get_output_features(self) -> int:
        """Get the output size of the DenseNet backbone.

        Calculates based on the number of features in the classifier and the input size.

        Returns:
            int: The output size.
        """
        return self.output_size
