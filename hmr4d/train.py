import hydra
import pytorch_lightning as pl
import rich
import rich.syntax
import rich.tree

from omegaconf import DictConfig, OmegaConf
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.utilities import rank_zero_only

from hmr4d.utils.pylogger import Log, monitor_process_wrapper
from hmr4d.utils.net_utils import load_pretrained_model, find_last_ckpt_path

import hmr4d.configs.config_register  # noqa: F401

# ================================ #
#      utils & get components      #
# ================================ #


@monitor_process_wrapper
def get_data(cfg: DictConfig) -> pl.LightningDataModule:
    datamodule = hydra.utils.instantiate(cfg.data, _recursive_=False)
    return datamodule


@monitor_process_wrapper
def get_model(cfg: DictConfig) -> pl.LightningModule:
    model = hydra.utils.instantiate(cfg.model, _recursive_=False)
    if hasattr(cfg, "exp_name"):
        # TODO: Maybe not elegant!
        model.exp_name = cfg.exp_name
    return model


@monitor_process_wrapper
def get_callbacks(cfg: DictConfig) -> list:
    if not hasattr(cfg, "callbacks") or cfg.callbacks is None:
        return None
    callbacks = []
    for callback in cfg.callbacks.values():
        if callback is not None:
            callbacks.append(hydra.utils.instantiate(callback, _recursive_=False))
    return callbacks


@rank_zero_only
def print_cfg(cfg: DictConfig, use_rich: bool = False):
    if use_rich:
        print_order = ("data", "model", "callbacks", "logger", "pl_trainer")
        style = "dim"
        tree = rich.tree.Tree("CONFIG", style=style, guide_style=style)

        # add fields from `print_order` to queue
        # add all the other fields to queue (not specified in `print_order`)
        queue = []
        for field in print_order:
            queue.append(field) if field in cfg else Log.warn(f"Field '{field}' not found in config. Skipping.")
        for field in cfg:
            if field not in queue:
                queue.append(field)

        # generate config tree from queue
        for field in queue:
            branch = tree.add(field, style=style, guide_style=style)
            config_group = cfg[field]
            if isinstance(config_group, DictConfig):
                branch_content = OmegaConf.to_yaml(config_group, resolve=False)
            else:
                branch_content = str(config_group)
            branch.add(rich.syntax.Syntax(branch_content, "yaml"))
        rich.print(tree)
    else:
        Log.info(OmegaConf.to_yaml(cfg, resolve=False))


# ================================ #
#           train & test           #
# ================================ #


def train(cfg: DictConfig) -> None:
    """
    Instantiate the trainer, and then train the model.
    """
    
    print_cfg(cfg, use_rich=True)
    pl.seed_everything(cfg.seed)

    # preparations
    datamodule: pl.LightningDataModule = get_data(cfg)
    model: pl.LightningModule = get_model(cfg)
    if cfg.ckpt_path is not None:
        load_pretrained_model(model, cfg.ckpt_path, cfg.ckpt_type)

    # PL callbacks and logger
    Log.info('PL callbacks and logger')
    callbacks = get_callbacks(cfg)
    logger = hydra.utils.instantiate(cfg.logger, _recursive_=False)

    # PL-Trainer
    trainer = pl.Trainer(
        accelerator="gpu",
        logger=logger if logger is not None else False,
        callbacks=callbacks,
        #strategy='ddp_find_unused_parameters_true',
        **cfg.pl_trainer,
    )

    if cfg.task == "fit":
        ckpt_path = None
        if cfg.resume_training:
            if cfg.resume_model_path is None:
                ckpt_path = find_last_ckpt_path(cfg.callbacks.model_checkpoint.dirpath)
            else:
                ckpt_path = cfg.resume_model_path
            Log.info(f"Resume training from {ckpt_path}")
        trainer.fit(model, datamodule.train_dataloader(), datamodule.val_dataloader(), ckpt_path=ckpt_path)
    elif cfg.task == "test":
        trainer.test(model, datamodule.test_dataloader())
    elif cfg.task == "predict":
        trainer.predict(model, datamodule.val_dataloader())
