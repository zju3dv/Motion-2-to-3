import os
import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

from hmr4d.dataset.HumanML3D.utils import resample_motion_fps
from hmr4d.model.mas.utils.motion3d_endecoder import Hmlvec263OriginalEnDecoder
import hmr4d.network.evaluator.t2m_motionenc as t2m_motionenc
import hmr4d.network.evaluator.t2m_textenc as t2m_textenc
from hmr4d.network.evaluator.word_vectorizer import POS_enumerator
from hmr4d.utils.hml3d.metric import (
    calculate_activation_statistics_np,
    calculate_frechet_distance_np,
)

torch.multiprocessing.set_sharing_strategy("file_system")

# NOTE: Follow HumanML3D uses the whole dataset instead of only the training set for mean and std
data_endecoder = Hmlvec263OriginalEnDecoder("hmr4d.network.evaluator.statistics", "T2M_VEC").cuda()

opt = {
    "dim_word": 300,
    "max_motion_length": 196,
    "dim_pos_ohot": len(POS_enumerator),
    "dim_motion_hidden": 1024,
    "max_text_len": 20,
    "dim_text_hidden": 512,
    "dim_coemb_hidden": 512,
    "dim_pose": 263,
    "dim_movement_enc_hidden": 512,
    "dim_movement_latent": 512,
    "checkpoints_dir": "./inputs/checkpoints/t2m",
    "unit_length": 4,
}
# init module
t2m_textencoder = t2m_textenc.TextEncoderBiGRUCo(
    word_size=opt["dim_word"],
    pos_size=opt["dim_pos_ohot"],
    hidden_size=opt["dim_text_hidden"],
    output_size=opt["dim_coemb_hidden"],
)

t2m_moveencoder = t2m_motionenc.MovementConvEncoder(
    input_size=opt["dim_pose"] - 4,
    hidden_size=opt["dim_movement_enc_hidden"],
    output_size=opt["dim_movement_latent"],
)

t2m_motionencoder = t2m_motionenc.MotionEncoderBiGRUCo(
    input_size=opt["dim_movement_latent"],
    hidden_size=opt["dim_motion_hidden"],
    output_size=opt["dim_coemb_hidden"],
)
# load pretrianed
t2m_checkpoint = torch.load(
    os.path.join(opt["checkpoints_dir"], "text_mot_match/model/finest.tar"),
    map_location="cpu",
)
t2m_textencoder.load_state_dict(t2m_checkpoint["text_encoder"])
t2m_moveencoder.load_state_dict(t2m_checkpoint["movement_encoder"])
t2m_motionencoder.load_state_dict(t2m_checkpoint["motion_encoder"])

# freeze params
t2m_textencoder.eval()
t2m_moveencoder.eval()
t2m_motionencoder.eval()
for p in t2m_textencoder.parameters():
    p.requires_grad = False
for p in t2m_moveencoder.parameters():
    p.requires_grad = False
for p in t2m_motionencoder.parameters():
    p.requires_grad = False

t2m_textencoder = t2m_textencoder.cuda()
t2m_moveencoder = t2m_moveencoder.cuda()
t2m_motionencoder = t2m_motionencoder.cuda()

all_emb = []

saved_path = "./outputs/saved_generation"
# saved_path = "./outputs/egoexo_gen"
all_pth = os.listdir(saved_path)
all_pth = [p for p in all_pth if p.endswith(".pth")]
all_pth = sorted(all_pth)  # ["00000.pth", ...]
# saved_pred is {"00000": {'pred', 'gt','text','length'}, ...}
saved_pred = {}
for i in range(len(all_pth)):
    k = all_pth[i]
    saved_pred[f"{i:05d}"] = torch.load(os.path.join(saved_path, k))

for k, v in tqdm(saved_pred.items()):
    pred_ayfz_motion_ = torch.zeros((1, 200, 22, 3)).cuda()
    gt_ayfz_motion_ = torch.zeros((1, 200, 22, 3)).cuda()
    now_len = (v["gt"].sum(dim=-1).sum(dim=-1) == 0).nonzero()
    if len(now_len) == 0:
        now_len = 300
    else:
        now_len = now_len[0]
    ori_len = int(now_len / 3.0 * 2)
    pred_ayfz_motion_[0, :ori_len] = resample_motion_fps(v["pred"][:now_len], ori_len)
    gt_ayfz_motion_[0, :ori_len] = resample_motion_fps(v["gt"][:now_len], ori_len)
    pred_ayfz_motion_1 = gt_ayfz_motion_.clone()
    pred_ayfz_motion_2 = gt_ayfz_motion_.clone()
    pred_ayfz_motion_3 = gt_ayfz_motion_.clone()
    pred_ayfz_motion_4 = gt_ayfz_motion_.clone()

    pred_ayfz_motion_1 = pred_ayfz_motion_1 - pred_ayfz_motion_1[..., :1, :] # this is correct
    pred_ayfz_motion_2 = pred_ayfz_motion_2 - pred_ayfz_motion_2[:, :, [0]] # this is correct
    pred_ayfz_motion_3 -= pred_ayfz_motion_3[..., :1, :] # do not use this
    pred_ayfz_motion_4 -= pred_ayfz_motion_4[:, :, [0]] # this is correct
    print(pred_ayfz_motion_1.std())
    print(pred_ayfz_motion_2.std())
    print(pred_ayfz_motion_3.std())
    print(pred_ayfz_motion_4.std())
    import ipdb;ipdb.set_trace()
    # pred_ayfz_motion_floor = pred_ayfz_motion_.reshape(1, -1, 3)[:, :, 1].min(dim=1)[0]  # B
    # pred_ayfz_motion_[..., 1] = pred_ayfz_motion_[..., 1] - pred_ayfz_motion_floor[:, None, None]

    ori_len = torch.tensor([ori_len], dtype=torch.long).cuda()
    hmlvec = data_endecoder.encode(pred_ayfz_motion_, ori_len)
    hmlvec = hmlvec.transpose(1, 2)
    # t2m motion encoder
    m_lens = torch.div(ori_len, 4, rounding_mode="floor")

    gt_motion_mov = t2m_moveencoder(hmlvec[..., :-4].cuda()).detach()
    gt_motion_emb = t2m_motionencoder(gt_motion_mov, m_lens)
    saved_pred[k]["emb"] = gt_motion_emb
    all_emb.append(gt_motion_emb)


all_emb = torch.cat(all_emb, dim=0)
mu, cov = calculate_activation_statistics_np(all_emb.detach().cpu().numpy())
gt = torch.load("./hmr4d/utils/hml3d/gt_stat.pth")
fid = calculate_frechet_distance_np(gt["mu"], gt["cov"], mu, cov)
print(fid)
torch.save(saved_pred, "./outputs/tmp_dump_fid.pt")  # Huaijin's debugging file
import ipdb

ipdb.set_trace()
if False:
    torch.save("tmp_dump_fid.pt")  # Zehong's debugging file
