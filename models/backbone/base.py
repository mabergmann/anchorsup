"""Abstract class for backbone networks."""
from abc import ABC, abstractmethod

import torch


class IBackbone(ABC, torch.nn.Module):
    """Abstract class for backbone networks."""

    @abstractmethod
    def __init__(self, image_size: tuple[int, int], *, pretrained: bool = True) -> None:
        """Initialize a new instance of the backbone network."""
        super().__init__()

    @abstractmethod
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass."""

    @abstractmethod
    def get_output_features(self) -> int:
        """Return the number of output features."""
