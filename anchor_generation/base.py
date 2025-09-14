"""This file contains the abstract class for the Anchor Generation Service."""
from abc import ABC, abstractmethod

import torch


class IAnchorGenerationService(ABC):
    """Abstract class for the Anchor Generation Service."""

    @abstractmethod
    def __init__(self, n_anchors: int) -> None:
        """Initialize a new instance of IAnchorGenerationService.

        Args:
            n_anchors (int): The number of anchors to generate.
        """

    @abstractmethod
    def generate(self) -> torch.Tensor:
        """Generate the anchors."""
