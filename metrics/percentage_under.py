"""Implementation of the angular error metric."""

import math

import numpy as np
import torch
from scipy.spatial.transform import Rotation

from upright_anchor.metrics.base import IMetric


class PercentageUnder(IMetric):
    """Implementation of the angular error metric."""

    PI = math.pi

    def __init__(self, anchors: torch.Tensor, threshold: float) -> None:
        """Initialize the metric."""
        self.reset()
        self.anchors = anchors
        self.threshold = threshold

    def __call__(self, y_true: torch.Tensor, y_pred: torch.Tensor) -> float:
        """Compute the metric.

        Args:
            y_true (torch.Tensor): The ground truth tensor.
            y_pred (torch.Tensor): The predicted tensor.

        Returns:
            float: The value of the metric.
        """
        best_anchor = torch.argmax(y_pred[:, :, 0], dim=-1)
        pred_angles = y_pred[range(len(y_true)), best_anchor, 1:]
        predictions = []
        for i in range(len(pred_angles)):
            ry, rx = pred_angles[i].detach().cpu().numpy()
            rot = Rotation.from_euler("yx", [ry, rx], degrees=False)
            v = rot.apply(self.anchors[best_anchor[i]].detach().cpu().numpy())
            predictions.append(v)
        predicted = torch.Tensor(np.array(predictions)).to(y_true.device)

        cossines = torch.nn.functional.cosine_similarity(predicted, y_true, dim=-1)

        angles = torch.acos(cossines)
        angles = angles * 180 / self.PI

        total = len(angles)
        under_threshold = torch.sum(angles <= self.threshold).item()
        self._total += total
        self._under_threshold += under_threshold
        return under_threshold / total

    def __str__(self) -> str:
        """Return the name of the metric."""
        return f"percentage_under_{self.threshold}"

    def reset(self) -> None:
        """Reset the metric."""
        self._total = 0
        self._under_threshold = 0

    def value(self) -> float:
        """Return the current value of the metric."""
        return self._under_threshold / self._total
