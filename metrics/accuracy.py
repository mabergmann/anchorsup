"""Implementation of the anchor estimation accuracy metric."""

import torch

from upright_anchor.metrics.base import IMetric


class AnchorAccuracy(IMetric):
    """Implementation of the anchor estimation accuracy metric."""

    def __init__(self, anchors: torch.Tensor) -> None:
        """Initialize the metric."""
        self.reset()
        self.anchors = anchors
        self.cos_similarity = torch.nn.CosineSimilarity(dim=-1)

    def __call__(self, y_true: torch.Tensor, y_pred: torch.Tensor) -> float:
        """Compute the metric.

        Args:
            y_true (torch.Tensor): The ground truth tensor.
            y_pred (torch.Tensor): The predicted tensor.

        Returns:
            float: The value of the metric.
        """
        self.anchors = self.anchors.to(y_pred.device)
        anchors_stacked = torch.stack([self.anchors] * len(y_true), dim=0)
        y_true_stacked = torch.stack([y_true] * len(self.anchors), dim=1)
        cos_similarity = self.cos_similarity(anchors_stacked, y_true_stacked)
        true_anchor = torch.argmax(cos_similarity, dim=-1)
        best_anchor = torch.argmax(y_pred[:, :, 0], dim=-1)

        correct_predictions = (best_anchor == true_anchor).sum().item()
        total_predictions = len(y_true)

        accuracy = correct_predictions / total_predictions
        self._values.append(accuracy)
        return self._values[-1]

    def __str__(self) -> str:
        """Return the name of the metric."""
        return "anchor_accuracy"

    def reset(self) -> None:
        """Reset the metric."""
        self._values = []

    def value(self) -> float:
        """Return the current value of the metric."""
        values_torch = torch.Tensor(self._values)
        return torch.mean(values_torch).item()
