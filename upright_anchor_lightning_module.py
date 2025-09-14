"""Lightning module for training the Upright Anchor model."""

import lightning
import torch
from lightning.pytorch.utilities.types import LRSchedulerTypeUnion
from omegaconf import DictConfig
from torch import nn, optim

from upright_anchor.metrics.base import IMetric


class UprightAnchorLightningModule(lightning.LightningModule):
    """Lightning module for training the Upright Anchor model."""

    def __init__(  # noqa: PLR0913
        self,
        model: nn.Module,
        criterion: nn.Module,
        optimizer: optim.Optimizer,
        scheduler: LRSchedulerTypeUnion,
        metrics: list[IMetric],
        cfg: DictConfig,
    ) -> None:
        """Initializes the UprightAnchorLightningModule.

        Args:
            model: The model to train.
            criterion: The loss function to use.
            optimizer: The optimizer to use.
            scheduler: The scheduler to use.
            metrics: The metrics to calculate.
            cfg: The configuration object.
        """
        super().__init__()
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.metrics = metrics
        self.scheduler = scheduler
        self.save_hyperparameters(cfg)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the model.

        Args:
            x: The input tensor.

        Returns:
            The output tensor.
        """
        return self.model(x)

    def training_step(self, batch: tuple[torch.Tensor, torch.Tensor]) -> torch.Tensor | None:
        """Runs a training step.

        Args:
            batch: The batch of data.
            batch_idx: The index of the batch.

        Returns:
            The loss tensor.
        """
        x, y = batch
        y_hat = self.model(x)
        loss, losses_parts = self.criterion(y_hat, y)
        self.log("train_loss", loss, prog_bar=True)
        for key, value in losses_parts.items():
            self.log(f"train_{key}", value, prog_bar=False)
        for metric in self.metrics:
            metric(y, y_hat)
            self.log(f"train_{metric}", metric.value(), prog_bar=False)
        if torch.isnan(loss).item():
            return None  # ignore the batch
        return loss

    def validation_step(self, batch: tuple[torch.Tensor, torch.Tensor]) -> torch.Tensor | None:
        """Runs a validation step.

        Args:
            batch: The batch of data.
            batch_idx: The index of the batch.

        Returns:
            The loss tensor.
        """
        x, y = batch
        y_hat = self.model(x)
        loss, losses_parts = self.criterion(y_hat, y)
        self.log("val_loss", loss, prog_bar=False)
        for key, value in losses_parts.items():
            self.log(f"val_{key}", value, prog_bar=False)
        for metric in self.metrics:
            metric(y, y_hat)
            self.log(f"val_{metric}", metric.value(), prog_bar=False)
        if torch.isnan(loss).item():
            return None
        return loss

    def test_step(self, batch: tuple[torch.Tensor, torch.Tensor]) -> torch.Tensor | None:
        """Runs a test step.

        Args:
            batch: The batch of data.
            batch_idx: The index of the batch.

        Returns:
            The loss tensor.
        """
        x, y = batch
        y_hat = self.model(x)
        loss, losses_parts = self.criterion(y_hat, y)
        self.log("test_loss", loss, prog_bar=False)
        for key, value in losses_parts.items():
            self.log(f"test_{key}", value, prog_bar=False)
        for metric in self.metrics:
            metric(y, y_hat)
            self.log(f"test_{metric}", metric.value(), prog_bar=False)
        return loss

    def configure_optimizers(self) -> dict[str, object]:
        """Configures the optimizer and scheduler.

        Returns:
            The optimizer.
        """
        return {"optimizer": self.optimizer, "lr_scheduler": self.scheduler, "monitor": "val_loss"}

    def on_train_epoch_start(self) -> None:
        """Runs at the start of the training epoch."""
        for metric in self.metrics:
            metric.reset()

    def on_validation_epoch_start(self) -> None:
        """Runs at the start of the validation epoch."""
        for metric in self.metrics:
            metric.reset()
    
    def on_test_epoch_start(self) -> None:
        """Runs at the start of the test epoch."""
        for metric in self.metrics:
            metric.reset()
