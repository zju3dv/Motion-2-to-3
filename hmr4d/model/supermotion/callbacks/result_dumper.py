import torch
import torch.nn.functional as F
import pytorch_lightning as pl
from hmr4d.utils.pylogger import Log
from hmr4d.configs import MainStore, builds
from hmr4d.utils.net_utils import detach_to_cpu


class Dumper(pl.Callback):
    def __init__(self):
        super().__init__()

        # Behaviours are the same for val/test/predict
        self.on_test_batch_end = self.on_validation_batch_end = self.on_predict_batch_end
        self.on_test_epoch_end = self.on_validation_epoch_end = self.on_predict_epoch_end
        self.dumps = []

    # ================== Batch-based Computation  ================== #
    def on_predict_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx=0):
        """The behaviour is the same for val/test/predict"""

        # Now, let's focus on incam simplify
        # Save gt so that I can compute metrics during experiment
        gender = batch["gender"][0]
        T_w2c = batch["T_w2c"][0]
        gt_smpl_params_world = {k: v[0] for k, v in batch["gt_smpl_params"].items()}

        dump = {
            "pred_smpl_params_incam": outputs["pred_smpl_params_incam"],
            "pred_cam": outputs["pred_cam"],
            "K_fullimg": batch["K_fullimg"][0],
            "bbx_xys": batch["bbx_xys"][0],
            "kp2d": batch["kp2d"][0],  # (L, J, 3), last dimension is confidence
            # Ground truth
            "gt_smpl_params_world": gt_smpl_params_world,
            "gender": gender,
            "T_w2c": T_w2c,
            # Render
            "meta_render": batch["meta_render"][0],
        }

        dump = detach_to_cpu(dump)
        self.dumps.append(dump)

    # ================== Epoch Summary  ================== #
    def on_predict_epoch_end(self, trainer, pl_module):
        out_fn = f"outputs/dump/{pl_module.exp_name}.pt"
        Log.info(f"Saving {len(self.dumps)} samples to {out_fn}")
        torch.save(self.dumps, out_fn)


cfg_dumper = builds(Dumper, populate_full_signature=True)
MainStore.store(name="supermotion", node=cfg_dumper, group=f"callbacks/dump_result")
