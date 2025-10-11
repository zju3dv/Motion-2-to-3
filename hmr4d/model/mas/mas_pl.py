from typing import Any, Dict
import torch
import pytorch_lightning as pl
from hydra.utils import instantiate
from hmr4d.utils.pylogger import Log
from einops import rearrange

from hmr4d.utils.check_utils import check_equal_get_one
from hmr4d.utils.geo_transform import compute_T_ayfz2ay, apply_T_on_points, compute_T_ayf2az
from hmr4d.utils.wis3d_utils import make_wis3d, add_motion_as_lines
from hmr4d.utils.smplx_utils import make_smplx
from hmr4d.utils.eval.sliding_windows import (
    get_window_startends,
    split_pad_batch,
    slide_merge,
    slide_merge_root_aa_ayfz,
)
from pytorch3d.transforms import axis_angle_to_matrix


class MASPL(pl.LightningModule):
    def __init__(
        self,
        pipeline,
        optimizer=None,
        scheduler_cfg=None,
        args=None,
        ignored_weights_prefix=["pipeline.clip"],
    ):
        super().__init__()
        self.pipeline = instantiate(pipeline, _recursive_=False)

        self.optimizer = instantiate(optimizer)

        #self.fake_optimizer = torch.optim.Adam(self.pipeline.parameters(), lr=1e-5) #!

        self.scheduler_cfg = scheduler_cfg

        # Options
        self.seed = args.seed
        self.rng_type = args.rng_type  # [const, random, seed_plus_idx]
        self.ignored_weights_prefix = ignored_weights_prefix

        # The test step is the same as validation
        self.test_step = self.predict_step = self.validation_step

    def training_step(self, batch, batch_idx):

        #self.fake_optimizer.zero_grad(set_to_none=True) #!
        
        # forward and compute loss
        B = self.trainer.train_dataloader.batch_size
        outputs = self.pipeline.forward_train(batch)
        loss = outputs["loss"]
        self.log("train/loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True, batch_size=B)
        for k, v in outputs.items():
            if "_loss" in k:
                self.log(f"train/{k}", v, on_step=True, on_epoch=True, prog_bar=True, logger=True, batch_size=B)
        """
        loss.backward()

        for n, p in dict(self.pipeline.named_parameters()).items():
            if p.requires_grad is True and p.grad is None:
                Log.warn(f"Parameter `{n}' has no grad.")
            if p.requires_grad is True and p.grad is not None:
                Log.info(f"Parameter `{n}' has grad.")
            #if p.grad is None:
            #    Log.warn(f"Parameter `{n}' has no grad.")

        breakpoint()
        """
        return outputs

    def validation_step(self, batch, batch_idx):
        """Task-specific validation step. 2D or 3D."""
        task = check_equal_get_one(batch["task"], "task")

        # Add generator to batch
        if "generator" not in batch:
            batch["generator"] = self.get_generatror(batch_idx)

        if task == "2D":
            outputs = self.pipeline.forward_sample(batch)
        elif task == "3D":
            outputs = self.pipeline.forward_sample3d(batch)
        else:
            raise NotImplementedError

        return outputs

    def configure_optimizers(self):
        params = []
        for k, v in self.pipeline.named_parameters():
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
        """Fix the random seed for each batch at sampling stage."""
        generator = torch.Generator(self.device)
        if self.rng_type == "const":
            generator.manual_seed(self.seed)
        elif self.rng_type == "random":
            pass
        elif self.rng_type == "seed_plus_idx":
            generator.manual_seed(self.seed + batch_idx)
        else:
            raise ValueError(f"rng_type `{self.rng_type}' is not supported.")
        generator.manual_seed(self.seed)
        return generator

    def on_save_checkpoint(self, checkpoint: Dict[str, Any]) -> None:
        for ig_keys in self.ignored_weights_prefix:
            flag = False
            for k in list(checkpoint["state_dict"].keys()):
                if k.startswith(ig_keys):
                    flag = True
                    checkpoint["state_dict"].pop(k)
            if flag:
                Log.debug(f"Remove key `{ig_keys}' from checkpoint.")

        super().on_save_checkpoint(checkpoint)

    def load_pretrained_model(self, ckpt_path, ckpt_type):
        """Load pretrained checkpoint, and assign each weight to the corresponding part."""
        Log.info(f"Loading ckpt type `{ckpt_type}': {ckpt_path}")

        if ckpt_type == "pl_2d_mv":
            assert len(ckpt_path) == 2
            state_dict = {
                **torch.load(ckpt_path[0], "cpu")["state_dict"],
                **torch.load(ckpt_path[1], "cpu")["state_dict"],
            }
            missing, unexpected = self.load_state_dict(state_dict, strict=False)
            real_missing = []
            for ig_keys in self.ignored_weights_prefix:
                for k in missing:
                    if not k.startswith(ig_keys):
                        real_missing.append(k)

            if len(real_missing) > 0:
                Log.warn(f"Missing keys: {real_missing}")
            if len(unexpected) > 0:
                Log.warn(f"Unexpected keys: {unexpected}")
        elif ckpt_type == "pl_2d_control":
            state_dict = torch.load(ckpt_path, "cpu")["state_dict"]
            new_state_dict = {}
            for k in state_dict.keys():
                new_state_dict[k] = state_dict[k]
                if "seqTransEncoder" in k:
                    prefix = k[:45]
                    postfix = k[45:]
                    new_state_dict[prefix + "c_" + postfix] = state_dict[k]
                if "seqTransDecoder" in k:
                    prefix = k[:45]
                    postfix = k[45:]
                    new_state_dict[prefix + "c_" + postfix] = state_dict[k]

            missing, unexpected = self.load_state_dict(new_state_dict, strict=False)
            real_missing = []
            for ig_keys in self.ignored_weights_prefix:
                for k in missing:
                    if not k.startswith(ig_keys):
                        real_missing.append(k)

            if len(real_missing) > 0:
                Log.warn(f"Missing keys: {real_missing}")
            if len(unexpected) > 0:
                Log.warn(f"Unexpected keys: {unexpected}")

        else:
            assert ckpt_type == "pl"
            state_dict = torch.load(ckpt_path, "cpu")["state_dict"]
            # for mv finetune, modify denoiser2d -> denoisermv
            if hasattr(self.pipeline, "denoisermv"):
                state_dict_ = {}
                for k in state_dict.keys():
                    state_dict_[k.replace("denoiser2d", "denoisermv")] = state_dict[k]
                state_dict = state_dict_
            missing, unexpected = self.load_state_dict(state_dict, strict=False)
            real_missing = []
            for ig_keys in self.ignored_weights_prefix:
                for k in missing:
                    if not k.startswith(ig_keys):
                        real_missing.append(k)

            if len(real_missing) > 0:
                Log.warn(f"Missing keys: {real_missing}")
            if len(unexpected) > 0:
                Log.warn(f"Unexpected keys: {unexpected}")
