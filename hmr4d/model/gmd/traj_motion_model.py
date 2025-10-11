import torch
import pickle
import pytorch_lightning as pl
from pathlib import Path
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf

import torch.nn.functional as F


class TrajMotionModel(pl.LightningModule):
    def __init__(self, traj_network, motion_network, seed=42):
        """This is the 2-stage implementation of GMD"""
        super().__init__()
        self.traj_network = instantiate(traj_network, _recursive_=False)
        self.motion_network = instantiate(motion_network, _recursive_=False)
        self.seed = seed

    @torch.inference_mode()
    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        batch["generator"] = self.get_generatror()

        # Stage1: Condition on root trajectory
        outputs_traj = self.traj_network.forward_sample(batch)
        pred_hmlvec4 = outputs_traj["pred_hmlvec4"]  # (B, 4, L)
        pred_traj = outputs_traj["pred_traj"]  # (B, L, 3)

        # Stage2: Condition on root trajectory
        # batch.update({"cond_hmlvec": pred_hmlvec4, "cond_hmlvec4_mask": torch.ones_like(pred_hmlvec4).bool()})
        cond_motion = F.pad(pred_traj.unsqueeze(2), (0, 0, 0, 21), value=0)  # (B, L, 22, 3)
        cond_motion_mask = torch.zeros_like(cond_motion).bool()  # (B, L, 22, 3)
        cond_motion_mask[:, :, 0, [0, 2]] = True  # Condition on root trajectory (x,z)
        batch.update({"cond_motion": cond_motion, "cond_motion_mask": cond_motion_mask})
        outputs_motion = self.motion_network.forward_sample(batch)

        outputs = {**outputs_traj, **outputs_motion}
        return outputs

    # ============== Utils ================= #

    def get_generatror(self):
        generator = torch.Generator(self.device)
        generator.manual_seed(self.seed)
        return generator

    def load_pretrained_model(self, ckpt_path, ckpt_type):
        """
        Load pretrained checkpoint, and assign each weight to the corresponding part.
        """
        if ckpt_type[0] == "gmd_release_folder@traj":
            ckpt_path_ = Path(ckpt_path) / "traj_unet_adazero_swxs_eps_abs_fp16_clipwd_224/model000062500.pt"
            state_dict = torch.load(ckpt_path_, "cpu")["model"]

            # Condition-latents
            state_clip_proj = {"weight": state_dict["embed_text.weight"], "bias": state_dict["embed_text.bias"]}
            self.traj_network.clip_proj.load_state_dict(state_clip_proj, strict=True)

            # UNet MDM
            # Remove some keys that start with prefix:
            keys_to_rm = ["embed_text", "sequence_pos_encoder", "embed_timestep.sequence_pos_encoder.pe"]
            state_unet = {}
            for k, v in state_dict.items():
                if not any([k.startswith(prefix) for prefix in keys_to_rm]):
                    state_unet[k] = v
            self.traj_network.unet.load_state_dict(state_unet, strict=True)
        else:
            raise NotImplementedError

        if ckpt_type[1] == "gmd_release_folder@motion":
            ckpt_path_ = Path(ckpt_path) / "unet_adazero_xl_x0_abs_proj10_fp16_clipwd_224/model000500000.pt"
            state_dict = torch.load(ckpt_path_, "cpu")["model"]

            # Condition-latents
            state_clip_proj = {"weight": state_dict["embed_text.weight"], "bias": state_dict["embed_text.bias"]}
            self.motion_network.clip_proj.load_state_dict(state_clip_proj, strict=True)

            # UNet MDM
            # Remove some keys that start with prefix:
            keys_to_rm = ["embed_text", "sequence_pos_encoder", "embed_timestep.sequence_pos_encoder.pe"]
            state_unet = {}
            for k, v in state_dict.items():
                if not any([k.startswith(prefix) for prefix in keys_to_rm]):
                    state_unet[k] = v
            self.motion_network.unet.load_state_dict(state_unet, strict=True)

        else:
            raise NotImplementedError
