"""Implementation of the angular error metric."""

import math

import numpy as np
import torch
from scipy.spatial.transform import Rotation

from upright_anchor.metrics.base import IMetric


class AngularError(IMetric):
    """Implementation of the angular error metric."""

    PI = math.pi

    def __init__(self, anchors: torch.Tensor) -> None:
        """Initialize the metric."""
        self.reset()
        self.anchors = anchors

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

        mean_angles = torch.mean(angles)
        self._values.append(mean_angles.item())
        return self._values[-1]

    def __str__(self) -> str:
        """Return the name of the metric."""
        return "angular_error"

    def reset(self) -> None:
        """Reset the metric."""
        self._values = []

    def value(self) -> float:
        """Return the current value of the metric."""
        values_torch = torch.Tensor(self._values)
        return torch.mean(values_torch).item()
