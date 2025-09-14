"""Abstract class for metrics."""

from abc import ABC, abstractmethod

import torch


class IMetric(ABC):
    """Abstract class for metrics."""

    @abstractmethod
    def __init__(self) -> None:
        """Initialize the metric."""

    @abstractmethod
    def __call__(self, y_true: torch.Tensor, y_pred: torch.Tensor) -> float:
        """Compute the metric."""
        raise NotImplementedError

    @abstractmethod
    def __str__(self) -> str:
        """Return the name of the metric."""
        raise NotImplementedError

    @abstractmethod
    def reset(self) -> None:
        """Reset the metric."""
        raise NotImplementedError

    @abstractmethod
    def value(self) -> float:
        """Return the current value of the metric."""
        raise NotImplementedError
