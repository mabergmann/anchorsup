import numpy as np
import torch
from torch import nn


class AnglesLoss(nn.Module):
    def __init__(self, alpha: float, anchors: torch.Tensor) -> None:
        super().__init__()
        self.cos_similarity = nn.CosineSimilarity(dim=-1)
        self.anchors = anchors
        self.ce = nn.CrossEntropyLoss()
        self.mse = nn.MSELoss()
        self.alpha = alpha

    def forward(self, pred: torch.Tensor, gt: torch.Tensor) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        if self.anchors.device != pred.device:
            self.anchors = self.anchors.to(pred.device)
        batch_size = gt.size(0)

        # Compute cosine similarities between anchors and ground truth
        cos_similarity = self.cos_similarity(
            self.anchors.unsqueeze(0).expand(batch_size, -1, -1),
            gt.unsqueeze(1).expand(-1, self.anchors.size(0), -1),
        )

        # Get the target anchor indices
        correct_anchor_index = torch.argmax(cos_similarity, dim=1)

        # Classification loss using raw logits and target indices
        classification_loss = self.ce(pred[:, :, 0], correct_anchor_index)

        # Vectorized computation of angles
        best_anchors = self.anchors[correct_anchor_index]
        pred_angles = pred[torch.arange(batch_size), correct_anchor_index, 1:]
        expected_angles = torch.stack(
            [align_vectors(best_anchors[i], gt[i]) for i in range(batch_size)]
        ).to(pred.device)

        regression_loss = self.mse(pred_angles, expected_angles)

        loss = self.alpha * classification_loss + (1 - self.alpha) * regression_loss
        return (
            loss.type(torch.float32),
            {"classification_loss": classification_loss, "regression_loss": regression_loss},
        )


def align_vectors(v1: torch.Tensor, v2: torch.Tensor) -> torch.Tensor:
    theta_v1 = torch.atan2(v1[0], v1[2])
    theta_v2 = torch.atan2(v2[0], v2[2])
    ry = theta_v2 - theta_v1

    cos_ry = torch.cos(ry)
    sin_ry = torch.sin(ry)
    v1x_prime = cos_ry * v1[0] + sin_ry * v1[2]
    v1y_prime = v1[1]
    v1z_prime = -sin_ry * v1[0] + cos_ry * v1[2]
    v1_rotated = torch.stack([v1x_prime, v1y_prime, v1z_prime])

    numerator = v1_rotated[1] * v2[2] - v1_rotated[2] * v2[1]
    denominator = v1_rotated[1] * v2[1] + v1_rotated[2] * v2[2]
    rx = torch.atan2(numerator, denominator)

    return torch.tensor([ry, rx])
