"""Utility functions for model setup and configuration."""

import torch
import torch.nn as nn
import torch.optim as optim
from omegaconf import DictConfig
from typing import List, Tuple, Union, Literal

from upright_anchor.anchor_generation import create_anchor_generator
from upright_anchor.models.backbone.base import IBackbone
from upright_anchor.models.backbone.convnext import ConvNeXtBackbone
from upright_anchor.models.backbone.convnext_swhdc import ConvNeXtBackboneSWHDC
# from upright_anchor.models.backbone.convnext_attention import ConvNextAttentionBackbone
from upright_anchor.models.backbone.densenet import DenseNetBackbone
# from upright_anchor.models.backbone.densenet_gcn import DenseNetBackboneGCN
from upright_anchor.models.backbone.densenet_swhdc import DenseNetBackboneSWHDC
from upright_anchor.metrics.accuracy import AnchorAccuracy
from upright_anchor.metrics.angular_error import AngularError
from upright_anchor.metrics.median_error import MedianError
from upright_anchor.metrics.percentage_under import PercentageUnder
from upright_anchor.models.upright_anchor import UprightAnchor

def get_backbone(backbone_name: str, image_size: tuple[int, int]) -> IBackbone:
    """Get the backbone model.

    Args:
        backbone_name (str): The name of the backbone model.
        image_size (tuple[int, int]): The size of the input image.

    Returns:
        IBackbone: The backbone model.

    Raises:
        ValueError: If the backbone name is not supported.
    """
    if backbone_name == "densenet":
        return DenseNetBackbone(image_size)
    elif backbone_name == "densenet_gcn":
        return DenseNetBackboneGCN(image_size)
    elif backbone_name == "densenet_swhdc":
        return DenseNetBackboneSWHDC(image_size)
    elif backbone_name == "convnext":
        return ConvNeXtBackbone(image_size)
    elif backbone_name == "convnext_attention":
        return ConvNextAttentionBackbone(image_size)
    elif backbone_name == "convnext_swhdc":
        return ConvNeXtBackboneSWHDC(image_size)
    else:
        raise ValueError(f"Unsupported backbone: {backbone_name}")

def get_angles_activation(activation_name: str) -> nn.Module:
    """Get the angles activation function.

    Args:
        activation_name (str): The name of the activation function.

    Returns:
        nn.Module: The activation function.

    Raises:
        ValueError: If the activation name is not supported.
    """
    if activation_name == "linear":
        return nn.Identity()
    elif activation_name == "tanh":
        return nn.Tanh()
    else:
        raise ValueError(f"Unsupported activation: {activation_name}")

def setup_model(cfg: DictConfig) -> UprightAnchor:
    """Set up the model.

    Args:
        cfg (DictConfig): The configuration.

    Returns:
        UprightAnchor: The model.
    """
    image_size = (cfg.data.height, cfg.data.width)
    backbone = get_backbone(cfg.model.backbone, image_size)
    angles_activation = get_angles_activation(cfg.model.angles_activation)
    anchors = create_anchor_generator(cfg).generate()
    return UprightAnchor(anchors, backbone, angles_activation), anchors

def setup_optimizer(model: nn.Module, cfg: DictConfig) -> optim.Optimizer:
    """Set up the optimizer.

    Args:
        model (nn.Module): The model.
        cfg (DictConfig): The configuration.

    Returns:
        optim.Optimizer: The optimizer.

    Raises:
        ValueError: If the optimizer name is not supported.
    """
    if cfg.optimizer.name == "adam":
        return optim.Adam(model.parameters(), lr=cfg.optimizer.lr, weight_decay=cfg.optimizer.weight_decay)
    else:
        raise ValueError(f"Unsupported optimizer: {cfg.optimizer.name}")

def setup_lr_scheduler(optimizer: optim.Optimizer, cfg: DictConfig) -> optim.lr_scheduler._LRScheduler:
    """Set up the learning rate scheduler.

    Args:
        optimizer (optim.Optimizer): The optimizer.
        cfg (DictConfig): The configuration.

    Returns:
        optim.lr_scheduler._LRScheduler: The learning rate scheduler.

    Raises:
        ValueError: If the scheduler name is not supported.
    """
    if cfg.lr_scheduler.name == "step":
        return optim.lr_scheduler.StepLR(optimizer, step_size=cfg.lr_scheduler.step_size, gamma=cfg.lr_scheduler.gamma)
    elif cfg.lr_scheduler.name == "cosine":
        return optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg.lr_scheduler.T_max)
    elif cfg.lr_scheduler.name == "reduce_on_plateau":
        return optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=cfg.lr_scheduler.factor, patience=cfg.lr_scheduler.patience
        )
    else:
        raise ValueError(f"Unsupported scheduler: {cfg.lr_scheduler.name}")

def setup_metrics(
    metric_names: List[str],
    anchors: torch.Tensor,
    is_train: bool
) -> List:
    """Set up evaluation metrics.
    
    Args:
        metric_names: List of metric names to set up
        anchors: Generated anchor points
        is_train: Whether the metrics are being set up for training or testing
        
    Returns:
        List of metric instances
        
    Raises:
        ValueError: If invalid metric specified
    """
    metrics = []
    for metric in metric_names:
        if metric == "angular_error":
            metrics.append(AngularError(anchors=anchors))
        elif metric == "anchor_accuracy":
            metrics.append(AnchorAccuracy(anchors=anchors))
        elif metric == "median_error":
            if not is_train:
                metrics.append(MedianError(anchors=anchors))
        elif metric.startswith("under_threshold"):
            if not is_train:
                threshold = float(metric.split("_")[-1])
                metrics.append(PercentageUnder(anchors=anchors, threshold=threshold))
        else:
            raise ValueError(f"Unknown metric: {metric}")
    return metrics

# For backward compatibility with existing code
def setup_model_from_cfg(cfg: DictConfig) -> Tuple[UprightAnchor, torch.Tensor]:
    """Wrapper for backward compatibility with config-based setup."""
    return setup_model(cfg), create_anchor_generator(cfg).generate()

def setup_metrics_from_cfg(cfg: DictConfig, anchors: torch.Tensor, is_train: bool) -> List:
    """Wrapper for backward compatibility with config-based setup."""
    return setup_metrics(cfg.metrics, anchors, is_train) 
