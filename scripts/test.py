"""Test script for model evaluation and inference.

This script loads a trained model checkpoint and runs evaluation on the test dataset.
It supports different backbones, loss functions, and metrics configurations through
a hydra config file.
"""
import logging
import pathlib as pl

import hydra
import lightning
import torch
from omegaconf import DictConfig

from upright_anchor.datasets.sun360 import SUN360
from upright_anchor.loss.angles_loss import AnglesLoss
from upright_anchor.utils.model_setup import setup_model, setup_metrics
from upright_anchor.upright_anchor_lightning_module import UprightAnchorLightningModule

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@hydra.main(version_base="1.3", config_path="../config", config_name="config")
def main(cfg: DictConfig) -> None:
    """Main function for testing the model.
    
    Args:
        cfg: The configuration object
    """
    # Load model checkpoint
    if not hasattr(cfg, "checkpoint_path"):
        raise ValueError("Please provide a checkpoint path in the config using checkpoint_path")
    checkpoint_path = pl.Path(cfg.checkpoint_path)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found at {checkpoint_path}")
    
    # Set up model and metrics
    model, anchors = setup_model(
        cfg
    )
    metrics = setup_metrics(
        metric_names=cfg.metrics,
        anchors=anchors,
        is_train=False
    )
    
    # Set up loss
    if cfg.loss.name == "angles":
        criterion = AnglesLoss(alpha=cfg.loss.alpha, anchors=anchors)
    else:
        raise ValueError(f"Unknown loss: {cfg.loss.name}")
    
    # Set up optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.optimizer.lr)

    # Set up scheduler
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=cfg.lr_scheduler.step_size, gamma=cfg.lr_scheduler.gamma)

    # Set up lightning module
    lightning_module = UprightAnchorLightningModule(
        model=model,
        criterion=criterion,
        metrics=metrics,
        cfg=cfg,
        optimizer=optimizer,
        scheduler=scheduler,
    )

    # Load the checkpoint
    logger.info(f"Loading checkpoint from {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path)
    lightning_module.load_state_dict(checkpoint["state_dict"])
    lightning_module.eval()

    # Set up data
    test_dataset = SUN360(cfg.data.test_path, cfg.data.lut_path)
    test_loader = torch.utils.data.DataLoader(
        test_dataset, 
        batch_size=cfg.data.batch_size, 
        shuffle=False, 
        num_workers=12,
    )

    # Set up trainer and test
    trainer = lightning.Trainer()
    trainer.test(model=lightning_module, dataloaders=test_loader)

if __name__ == "__main__":
    main()
