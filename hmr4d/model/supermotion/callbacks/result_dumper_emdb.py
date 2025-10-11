import torch
import torch.nn.functional as F
import pytorch_lightning as pl
from pytorch_lightning.utilities import rank_zero_only
from hmr4d.utils.metric_utils import SumAggregator
from hmr4d.utils.pylogger import Log
from hmr4d.utils.check_utils import check_equal_get_one

from hmr4d.utils.eval.wham.eval_utils import first_align_joints, global_align_joints, compute_jpe


def is_target_task(batch):
    task = check_equal_get_one(batch["task"], "task")
    return task == "CAP"


class Dumper(pl.Callback):
    def __init__(self):
        super().__init__()

        # The metrics are calculated similarly for val/test/predict
        self.on_test_batch_end = self.on_validation_batch_end = self.on_predict_batch_end
        self.on_test_epoch_end = self.on_validation_epoch_end = self.on_predict_epoch_end

        self.outputs = []

    # ================== Batch-based Computation  ================== #
    def on_predict_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx=0):
        """
        The behaviour is the same for val/test/predict
        """
        if not is_target_task(batch):
            return

        B = batch["length"].shape[0]
        lengths = batch["length"]  # (B,)

        pred_ayfz_motion = outputs["pred_ayfz_motion"]  # (B, L, 22, 3)
        meta = batch["meta"]

        gt_w_motion3d = batch["gt_w_motion3d"]  # (B, L, 24, 3)
        gt_T_w2c = batch["gt_T_w2c"]
        gt_c_motion3d = batch["gt_c_motion3d"]  # (B, L, 24, 3)

        # Check if pred_motion has nan, if so, set it to zero and print a warning
        if torch.isnan(pred_ayfz_motion).any():
            for b in range(B):
                if torch.isnan(pred_ayfz_motion[b, : lengths[b]]).any():
                    print(meta[b])

            nan_mask = torch.isnan(pred_ayfz_motion)
            num_nan_item = (nan_mask.view(B, -1).any(dim=-1)).sum()
            Log.warn(f"{num_nan_item} pred_motion has nan")
            pred_ayfz_motion[nan_mask] = 0

        # Accumluate the predictions to dump
        for b in range(B):
            one_data = {
                "meta": meta[b],
                "pred_ayfz_motion": pred_ayfz_motion[b, : lengths[b], :].detach().clone().cpu(),
                "pred_T_ayfz2c": outputs["T_ayfz2c"][b, : lengths[b]].detach().clone().cpu(),  # (4, 4)

                "gt_w_motion": gt_w_motion3d[b, : lengths[b], :].detach().clone().cpu(),
                "gt_T_w2c":gt_T_w2c[b].detach().clone().cpu(),  # (4, 4)
                "gt_c_motion": gt_c_motion3d[b, : lengths[b], :].detach().clone().cpu(),

                "wham_w_motion": batch["pred_w_motion3d"][b, : lengths[b], :].detach().clone().cpu(),
                "wham_cr_motion": batch["pred_cr_motion3d"][b, : lengths[b], :].detach().clone().cpu(),
            }

            self.outputs.append(one_data)

    # ================== Epoch Summary  ================== #
    def on_predict_epoch_end(self, trainer, pl_module):
        """Without logger"""

        total = len(self.outputs)
        Log.info(f"{total} samples will be saved in {self.__class__.__name__}")
        if total == 0:
            return

        # Save the predictions
        torch.save(self.outputs, f"dump_emdb.pt")
