import torch
import pickle
import pytorch_lightning as pl
from pathlib import Path
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf
from time import time

from hmr4d.utils.pylogger import Log


class MDMModel(pl.LightningModule):
    def __init__(self, network, optimizer=None, scheduler_cfg=None, seed=42, args={}):
        super().__init__()
        self.network = instantiate(network, _recursive_=False)
        self.optimizer = instantiate(optimizer)
        self.scheduler_cfg = scheduler_cfg
        self.seed = seed
        self.test_step = self.predict_step = self.validation_step  # The test step is the same as validation

        generate_func = {"motion": "forward_sample"}
        self.generate_target = generate_func[args.generate_target]

        self.is_random_seed = args.get("random_seed", False)
        if self.is_random_seed:
            print("is_random_seed: True. Seed of each batch is set with batch_idx for diffusion.")

    def training_step(self, batch, batch_idx):
        # forward and compute loss
        outputs = self.network.forward_train(batch)
        loss = outputs["loss"]

        # log metrics
        B = batch["length"].shape[0]
        self.log("train/loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True, batch_size=B)
        return outputs

    def validation_step(self, batch, batch_idx):
        batch["generator"] = self.get_generatror(batch_idx)
        outputs = getattr(self.network, self.generate_target)(batch)
        return outputs

    def configure_optimizers(self):
        params = []
        for k, v in self.network.named_parameters():
            if v.requires_grad:
                params.append(v)
        optimizer = self.optimizer(params=params)

        if self.scheduler_cfg is None:
            return optimizer

        scheduler_cfg = self.scheduler_cfg
        scheduler_cfg["scheduler"] = instantiate(scheduler_cfg["scheduler"], optimizer=optimizer)

        return [optimizer], [scheduler_cfg]

    # ============== Utils ================= #

    def get_generatror(self, batch_idx):
        generator = torch.Generator(self.device)
        if self.is_random_seed:
            generator.manual_seed(batch_idx)
        else:
            generator.manual_seed(self.seed)
        return generator

    def load_pretrained_model(self, ckpt_path, ckpt_type):
        """
        Load pretrained checkpoint, and assign each weight to the corresponding part.
        """
        Log.info(f"Loading ckpt type `{ckpt_type}': {ckpt_path}")
        if ckpt_type == "pl":
            state_dict = torch.load(ckpt_path, "cpu")["state_dict"]
            try:
                self.load_state_dict(state_dict, strict=True)
            except Exception as e:
                Log.warn(f"Loading ckpt failed: {e}")
                self.load_state_dict(state_dict, strict=False)
        else:
            raise NotImplementedError
