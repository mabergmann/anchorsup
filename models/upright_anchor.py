"""The upright anchor Neural Network."""
import torch
import torch.nn as nn
import torchvision.models as models
import torch.nn.functional as F

from upright_anchor.models.backbone.base import IBackbone


class UprightAnchor(torch.nn.Module):
    """The upright anchor Neural Network."""

    def __init__(self, anchors: torch.Tensor, backbone: IBackbone, angles_activation: torch.nn.Module) -> None:
        """Initialize a new instance of UprightAnchor."""
        super().__init__()
        self.anchors = anchors
        self.backbone = backbone
        self.regressor = torch.nn.Linear(backbone.get_output_features(), 3 * self.anchors.shape[0])
        self.angles_activation = angles_activation

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass of the UprightAnchor network.

        Args:
            x (torch.Tensor): The input tensor.

        Returns:
            torch.Tensor: The output tensor.
        """
        features = self.backbone(x)
        x = self.regressor(features)
        
        x = x.view(-1, self.anchors.shape[0], 3)
        x[:, :, 1:] = self.angles_activation(x[:, :, 1:])
        return x
        
