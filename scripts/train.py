"""Entry point for training the model."""

import hydra
import lightning
from lightning.pytorch.callbacks import checkpoint
import torch
from omegaconf import DictConfig

from upright_anchor.datasets.sun360 import SUN360
from upright_anchor.loss.angles_loss import AnglesLoss
from upright_anchor.utils.model_setup import setup_model, setup_metrics
from upright_anchor.upright_anchor_lightning_module import UprightAnchorLightningModule


@hydra.main(version_base="1.3", config_path="../config", config_name="config")
def main(cfg: DictConfig) -> None:
    """Main function for training the model.

    Args:
        cfg: The configuration object

    Raises:
        ValueError: If an unknown configuration is provided.
    """
    lightning.pytorch.seed_everything(cfg.seed, workers=True)

    # Set up model and metrics
    model, anchors = setup_model(
        # backbone_name=cfg.model.backbone,
        # input_size=(cfg.data.width, cfg.data.height),
        # n_anchors=cfg.model.n_anchors,
        # angles_activation=cfg.model.angles_activation
        cfg=cfg
    )
    metrics = setup_metrics(
        metric_names=cfg.metrics,
        anchors=anchors,
        is_train=True
    )
    
    

    #print("-- ANCHORS --", anchors.shape)

    # Set up loss
    if cfg.loss.name == "angles":
        criterion = AnglesLoss(alpha=cfg.loss.alpha, anchors=anchors)
    else:
        raise ValueError(f"Unknown loss: {cfg.loss.name}")

    # Set up optimizer
    if cfg.optimizer.name == "adam":
        # Split parameters into two groups - with and without weight decay
        decay = []
        no_decay = []
        for name, param in model.named_parameters():
            if not param.requires_grad:
                continue
            # Apply weight decay to weights, but not to biases and batch norm parameters
            if len(param.shape) == 1 or name.endswith(".bias") or 'bn' in name or 'norm' in name:
                no_decay.append(param)
            else:
                decay.append(param)
        optimizer = torch.optim.Adam([
            {'params': decay, 'weight_decay': cfg.optimizer.weight_decay},
            {'params': no_decay, 'weight_decay': 0.0}
        ], lr=cfg.optimizer.lr)
    elif cfg.optimizer.name == "sgd":
        # Split parameters into two groups - with and without weight decay
        decay = []
        no_decay = []
        for name, param in model.named_parameters():
            if not param.requires_grad:
                continue
            # Apply weight decay to weights, but not to biases and batch norm parameters
            if len(param.shape) == 1 or name.endswith(".bias") or 'bn' in name or 'norm' in name:
                no_decay.append(param)
            else:
                decay.append(param)
        optimizer = torch.optim.SGD([
            {'params': decay, 'weight_decay': cfg.optimizer.weight_decay},
            {'params': no_decay, 'weight_decay': 0.0}
        ], lr=cfg.optimizer.lr)
    else:
        raise ValueError(f"Unknown optimizer: {cfg.optimizer.name}")

    # Set up learning rate scheduler
    if cfg.lr_scheduler.name == "step":
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer,
            step_size=cfg.lr_scheduler.step_size,
            gamma=cfg.lr_scheduler.gamma,
        )
    elif cfg.lr_scheduler.name == "cosine":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg.lr_scheduler.T_max)
    elif cfg.lr_scheduler.name == "plateau":
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=cfg.lr_scheduler.factor,
            patience=cfg.lr_scheduler.patience,
        )
    else:
        raise ValueError(f"Unknown scheduler: {cfg.lr_scheduler.name}")

    # Set up callbacks
    callbacks = [
        lightning.pytorch.callbacks.ModelCheckpoint(
            monitor="val_angular_error",
            filename="best_model",
            save_top_k=1,
            mode="min",
        ),
        lightning.pytorch.callbacks.ModelCheckpoint(
            filename="last_model",
            save_last=True,
        ),
        # lightning.pytorch.callbacks.EarlyStopping(
        #     monitor="val_angular_error",
        #     patience=500, # 50
        #     mode="min",
        #     check_finite=False,
        # ),
        lightning.pytorch.callbacks.LearningRateMonitor(logging_interval="epoch"),
        lightning.pytorch.callbacks.DeviceStatsMonitor(),
    ]

    # Set up lightning module
    lightning_module = UprightAnchorLightningModule(
        model=model,
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler,
        metrics=metrics,
        cfg=cfg,
    )
    lightning_module.compile()

    # Set up data loaders
    train_dataset = SUN360(cfg.data.train_path, cfg.data.lut_path)
    val_dataset = SUN360(cfg.data.val_path, cfg.data.lut_path)
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=cfg.data.batch_size, shuffle=True, num_workers=12,
    )
    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=cfg.data.batch_size, shuffle=False, num_workers=12
    )

    # Set up logger and trainer
    wandb_logger = lightning.pytorch.loggers.WandbLogger(project="upright_anchors", log_model=True)
    trainer = lightning.Trainer(
        max_epochs=cfg.trainer.max_epochs,
        gradient_clip_val=cfg.trainer.gradient_clip_val,
        callbacks=callbacks,
        logger=wandb_logger,
    )

    # Train model
    trainer.fit(
        model=lightning_module,
        train_dataloaders=train_loader,
        val_dataloaders=val_loader,
        ckpt_path=cfg.get('checkpoint_path', None),
    )

    # Test on test set
    test_dataset = SUN360(cfg.data.test_path, cfg.data.lut_path)
    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=cfg.data.batch_size, shuffle=False, num_workers=12
    )
    metrics = setup_metrics(
        metric_names=cfg.metrics,
        anchors=anchors,
        is_train=False
    )
    lightning_module.metrics = metrics
    lightning_module.eval()
    # Load best model and test
    lightning_module.load_state_dict(torch.load(trainer.checkpoint_callback.best_model_path)["state_dict"])
    trainer.test(model=lightning_module, dataloaders=test_loader)

if __name__ == "__main__":
    main()
