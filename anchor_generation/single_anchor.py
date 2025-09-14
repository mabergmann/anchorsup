"""Single anchor generation service.

This service generates a single anchor point at [0, 0, 1].
"""
import torch

from upright_anchor.anchor_generation.base import IAnchorGenerationService


class SingleAnchor(IAnchorGenerationService):
    """Generates a single anchor point at [0, 0, 1]."""

    def __init__(self, n_anchors: int) -> None:
        """Initialize a new instance of SingleAnchor.

        Args:
            n_anchors (int): This parameter is ignored as this service always generates a single anchor.
        """
        super().__init__(n_anchors)

    def generate(self) -> torch.Tensor:
        """Generate a single anchor point at [0, 0, 1].

        Returns:
            torch.Tensor: A tensor containing a single anchor point [0, 0, 1].
        """
        return torch.tensor([[0.0, 0.0, 1.0]]) 