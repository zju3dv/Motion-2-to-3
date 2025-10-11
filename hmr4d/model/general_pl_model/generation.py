import torch
import pytorch_lightning as pl
from hydra.utils import instantiate
from hmr4d.utils.pylogger import Log


class GenerationPL(pl.LightningModule):
    def __init__(self, network, optimizer=None, scheduler_cfg=None, seed=42):
        super().__init__()
        self.network = instantiate(network, _recursive_=False)
        self.optimizer = instantiate(optimizer)
        self.scheduler_cfg = scheduler_cfg
        self.seed = seed
        self.test_step = self.predict_step = self.validation_step  # The test step is the same as validation

    def training_step(self, batch, batch_idx):
        # forward and compute loss
        outputs = self.network.forward_train(batch)
        loss = outputs["loss"]
        self.log("train/loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return outputs

    def validation_step(self, batch, batch_idx):
        batch["generator"] = self.get_generatror()
        # sample
        outputs = self.network.forward_sample(batch)
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

    def get_generatror(self):
        generator = torch.Generator(self.device)
        generator.manual_seed(self.seed)
        return generator

    def load_pretrained_model(self, ckpt_path, ckpt_type):
        """
        Load pretrained checkpoint, and assign each weight to the corresponding part.
        """
        Log.info(f"Loading ckpt type `{ckpt_type}': {ckpt_path}")
        if ckpt_type == "pl":
            state_dict = torch.load(ckpt_path, "cpu")["state_dict"]
            self.load_state_dict(state_dict, strict=True)
        else:
            raise NotImplementedError
