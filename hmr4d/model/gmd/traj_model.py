import torch
import pickle
import pytorch_lightning as pl
from pathlib import Path
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf


class TrajModel(pl.LightningModule):
    def __init__(self, network, optimizer=None, scheduler_cfg=None, seed=42):
        super().__init__()
        self.network = instantiate(network, _recursive_=False)
        self.optimizer = instantiate(optimizer)
        self.scheduler_cfg = scheduler_cfg
        self.seed = seed

    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        """
        Return the predictions and some inputs of the network.
        The PredWriter will save the outputs to disk.
        """
        self.network = self.network.to(self.device)
        batch["generator"] = self.get_generatror()
        outputs = self.network.forward_sample(batch)
        return outputs

    def get_generatror(self):
        """
        Get a random generator with the same seed as the model.
        """
        generator = torch.Generator(self.device)
        generator.manual_seed(self.seed)
        return generator

    def load_pretrained_model(self, ckpt_path, ckpt_type):
        """
        Load pretrained checkpoint, and assign each weight to the corresponding part.
        """
        assert ckpt_type == "gmd_release_folder@traj"
        ckpt_path = Path(ckpt_path) / "traj_unet_adazero_swxs_eps_abs_fp16_clipwd_224/model000062500.pt"
        # 'unet', 'sequence_pos_encoder', 'embed_timestep', 'embed_text'
        state_dict = torch.load(ckpt_path, "cpu")["model"]

        # Condition-latents
        state_clip_proj = {"weight": state_dict["embed_text.weight"], "bias": state_dict["embed_text.bias"]}
        self.network.clip_proj.load_state_dict(state_clip_proj, strict=True)

        # UNet MDM
        # Remove some keys that start with prefix:
        keys_to_rm = ["embed_text", "sequence_pos_encoder", "embed_timestep.sequence_pos_encoder.pe"]
        state_unet = {}
        for k, v in state_dict.items():
            if not any([k.startswith(prefix) for prefix in keys_to_rm]):
                state_unet[k] = v
        self.network.unet.load_state_dict(state_unet, strict=True)
