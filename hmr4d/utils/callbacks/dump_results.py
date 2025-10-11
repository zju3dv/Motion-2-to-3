import torch
import torch.nn.functional as F
from pathlib import Path
import pytorch_lightning as pl
from pytorch_lightning.utilities import rank_zero_only
from hmr4d.utils.wis3d_utils import make_wis3d, add_motion_as_lines
from hmr4d.utils.geo_transform import apply_T_on_points, compute_T_ayfz2ay
import hmr4d.utils.matrix as matrix
from hmr4d.dataset.supermotion.collate import pad_to_max_len
from einops import einsum, rearrange, repeat

from hmr4d.utils.net_utils import detach_to_cpu


class ResultDumper(pl.Callback):
    def __init__(self, saved_dir="./outputs/dumped_newtext_ours"):
        super().__init__()
        # self.name = name
        self.on_test_batch_end = self.on_predict_batch_end

        self.saved_dir = Path(saved_dir)
        self.saved_dir.mkdir(parents=True, exist_ok=True)
        self.i = 0

    @rank_zero_only
    def on_predict_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx=0):
        # input
        length = batch["length"]
        text = batch["text"]

        if "pred_global_motion" in outputs.keys():
            # generated motion (AY)
            pred_motion = outputs["pred_global_motion"]  # (B, L, 22, 3)
            B, L, J, _ = pred_motion.shape
            # AY -> AYFZ
            T_ay2ayfz = compute_T_ayfz2ay(pred_motion[:, 0], inverse=True)  # (B, 4, 4)
            pred_ayfz_motion = apply_T_on_points(rearrange(pred_motion, "b l j c -> b (l j) c"), T_ay2ayfz)  # (B, L*22, 3)
            pred_ayfz_motion = rearrange(pred_ayfz_motion, "b (l j) c -> b l j c", j=J)  # (B, L, 22, 3)
        
            pred_ayfz_motion_ = torch.zeros((B, L, J, 3), device=pred_ayfz_motion.device)
            for i, l in enumerate(length):
                pred_ayfz_motion_[i, :l] = pred_ayfz_motion[i, :l]  # pad with last frame
                pred_ayfz_motion_[i, l:] = pred_ayfz_motion[i, l - 1]  # pad with last frame
            pred_ayfz_motion_floor = pred_ayfz_motion.reshape(B, -1, 3)[:, :, 1].min(dim=1)[0]  # B
            pred_ayfz_motion_[..., 1] = pred_ayfz_motion_[..., 1] - pred_ayfz_motion_floor[:, None, None]

            for b in range(B):
                saved_data = {
                    "pred": pred_ayfz_motion_[b].clone(),
                    "text": text[b],
                    "length": length[b],
                }
                name_txt = text[b][11:40].replace(" ", "_")
                saved_data = detach_to_cpu(saved_data)
                global_rank = trainer.global_rank
                # torch.save(saved_data, self.saved_dir / f"{global_rank}_{self.i:05d}.pth")
                torch.save(saved_data, self.saved_dir / f"{global_rank}_{self.i:05d}_{name_txt}.pth")
                self.i += 1
        else:
            pred_motion = outputs["pred_motion2d"]  # (B, L, 22, 2)
            B, L, J, _ = pred_motion.shape
            pred_motion_ = torch.zeros((B, L, J, 2), device=pred_motion.device)
            for i, l in enumerate(length):
                pred_motion_[i, :l] = pred_motion[i, :l]  # pad with last frame
                pred_motion_[i, l:] = pred_motion[i, l - 1]  # pad with last frame

            for b in range(B):
                saved_data = {
                    "pred": pred_motion_[b].clone(),
                    "text": text[b],
                    "length": length[b],
                }
                name_txt = text[b].replace(" ", "_")[10:40]
                saved_data = detach_to_cpu(saved_data)
                global_rank = trainer.global_rank
                torch.save(saved_data, self.saved_dir / f"{global_rank}{name_txt}_{self.i:05d}.pth")
                self.i += 1
