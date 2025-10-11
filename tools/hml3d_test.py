import os
import json
import numpy as np
import torch
from hmr4d.utils.hml3d.metric import get_metric_statistics, print_table


REPLICATION_TIMES = 20
EXPORT_PATH = "./outputs/TEST_Hml3d/motion3d_pretrainditlikev2_l8_b128_n8_g4"
CONFIG_PATH = "motion3d_prior/pretrain_ditlikev2/bigv1"
CHECKPOINTS_PATH = "inputs/checkpoints_motion3d/bigv0_e249.ckpt"
for i in range(REPLICATION_TIMES):
    os.system(
        f"YDRA_FULL_ERROR=1 python tools/train.py exp={CONFIG_PATH} global/task=supermotion/test_generate_prior3d ckpt_path={CHECKPOINTS_PATH} model.pipeline.args.guidance_scale=2.5 seed={i}"
    )
    os.system(
        f"YDRA_FULL_ERROR=1 python tools/train.py exp={CONFIG_PATH} global/task=supermotion/test_mm_generate_prior3d ckpt_path={CHECKPOINTS_PATH} model.pipeline.args.guidance_scale=2.5 seed={i}"
    )
EVAL_PATH = os.path.join(EXPORT_PATH, "evaluation")
os.makedirs(EVAL_PATH, exist_ok=True)
all_metrics = {}
for p in os.listdir(EVAL_PATH):
    if p.endswith(".pt"):
        metrics = torch.load(os.path.join(EVAL_PATH, p))
        for k, v in metrics.items():
            if k not in all_metrics:
                all_metrics[k] = []
            if isinstance(v, (torch.Tensor, np.ndarray)):
                v = v.item()
            all_metrics[k].append(float(v))
all_metrics_new = {}
for k, v in all_metrics.items():
    if len(v) != REPLICATION_TIMES:
        print(f"Loaded metrics - {k} ({len(v)}) does not equal to replication times ({REPLICATION_TIMES})!")
    mean, conf_interval = get_metric_statistics(np.array(v), REPLICATION_TIMES)
    all_metrics_new[k + "/mean"] = mean
    all_metrics_new[k + "/conf_interval"] = conf_interval
print_table(f"Mean Metrics", all_metrics_new)
all_metrics_new.update(all_metrics)
metric_file = os.path.join(EVAL_PATH, "metrics.json")
with open(metric_file, "w", encoding="utf-8") as f:
    json.dump(all_metrics_new, f, indent=4)
