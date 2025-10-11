# This file is a run-script to examine inidividual files
import torch
from omegaconf import OmegaConf
from hydra.utils import instantiate
from hmr4d.utils.wis3d_utils import make_wis3d, add_motion_as_lines
from tqdm import tqdm
from pytorch3d.transforms import axis_angle_to_matrix


def wis3d_debug(ayfz_joints, name, start_i, prefix=""):
    for i in range(ayfz_joints.shape[0]):
        wis3d = make_wis3d(name=name)
        add_motion_as_lines(ayfz_joints[i], wis3d, name=prefix + str(i + start_i))


def batch_mean_var(x, batch_size):
    # x: List (Tensor (1, C))
    n_samples = len(x)
    n_batches = (n_samples + batch_size - 1) // batch_size  # 确保覆盖所有数据

    mean_sum = torch.zeros(x[0].size(1)).cuda()
    sq_sum = torch.zeros(x[0].size(1)).cuda()
    total_samples = 0

    for i in tqdm(range(n_batches)):
        batch = x[i * batch_size : min((i + 1) * batch_size, n_samples)]
        batch = torch.cat(batch, dim=0).cuda()
        batch_mean = batch.mean(0)
        batch_sq_mean = batch.pow(2).mean(0)

        mean_sum += batch_mean * batch.size(0)
        sq_sum += batch_sq_mean * batch.size(0)
        total_samples += batch.size(0)

    mean = mean_sum / total_samples
    var = (sq_sum / total_samples) - (mean**2)
    return mean, var


def run_dataset():
    def get_dataset(TYPE):
        if TYPE == "AMASS_SMPL":
            from hmr4d.dataset.amass.amass_smpl import SMDataset

            dataset = SMDataset()
            return dataset
        elif TYPE == "BEDLAM_SMPL":
            from hmr4d.dataset.bedlam.bedlam_smpl import SMDataset

            dataset = SMDataset(max_motion_time=4)
            return dataset

    TYPE = "BEDLAM_SMPL"
    dataset = get_dataset(TYPE)
    print(f"Dataset size: {len(dataset)}")
    data = dataset[0]
    from torch.utils.data import DataLoader
    from tqdm import tqdm
    from hmr4d.dataset.supermotion.collate import collate_fn
    from hmr4d.model.supermotion.utils.motion3d_endecoder import (
        SMPLEnDecoder,
        SMPLRelVecV50EnDecoder,
        SMPLRelVecV51EnDecoder,
    )
    from hmr4d.model.supermotion.utils.motionsmpl_endecoder import (
        SMPLRelVecV6EnDecoder,
    )

    data_endecoder = SMPLRelVecV51EnDecoder(
        "hmr4d.network.supermotion.statisticssmpl",
        "SMPL",
        "SMPL",  # stats_incam_name
        forward_func="fk",
    )
    # data_endecoder = SMPLRelVecV6EnDecoder(
    #     "hmr4d.network.supermotion.statisticssmpl",
    #     "SMPL",
    #     "SMPL",  # stats_incam_name
    #     forward_func="fk",
    # )
    data_endecoder = data_endecoder.cuda()

    loader = DataLoader(
        dataset,
        shuffle=False,
        num_workers=0,
        persistent_workers=False,
        pin_memory=False,
        batch_size=1,
        collate_fn=collate_fn,
    )
    i = 0
    error_N = 0
    all_x = []
    for batch in tqdm(loader):
        i += 1
        for k in batch.keys():
            if isinstance(batch[k], torch.Tensor):
                batch[k] = batch[k].cuda()

        ##### Get statistics, work for amass dataset ##########
        # beta = batch["beta"] # (B, 10)
        # skeleton = batch["skeleton"] # (B, J, 3)
        # gender = batch["gender"]
        # modeltype = batch["model"]

        # x = data_endecoder.encode(batch["smplpose"].cuda())  # B, C, L
        # x_ = x.permute(0, 2, 1)  # B, L, C
        # l = batch["length"]
        # for j in range(x.shape[0]):
        #     all_x.append(x_[j, : l[j]].detach().cpu())  # L, C
        #################

        ##### Get statistics for incam, work for bedlam dataset ##########
        # beta = batch["beta"]  # (B, 10)
        # skeleton = batch["skeleton"]  # (B, J, 3)
        # gender = batch["gender"]
        # modeltype = batch["model"]

        # transl_incam = batch["transl_incam"]  # (B, L, 3)
        # global_orient_incam = batch["global_orient_incam"]  # (B, L, 3)
        # param_incam = torch.cat([transl_incam, global_orient_incam], dim=-1)  # (B, L, 6)

        # smpl_endecoder.set_beta(beta)
        # smpl_endecoder.set_skeleton(skeleton)
        # smpl_endecoder.set_gender(gender)
        # smpl_endecoder.set_modeltype(modeltype)

        # data_endecoder.set_beta(beta)
        # data_endecoder.set_skeleton(skeleton)

        # # x = data_endecoder.encode(batch["smplpose"].cuda())  # B, C, L # For debug nay to ay, do not use this during calculating

        # x = data_endecoder.encode_pose_incam(param_incam.cuda())  # B, C, L
        # x_ = x.permute(0, 2, 1)  # B, L, C
        # l = batch["length"]
        # for j in range(x.shape[0]):
        #     all_x.append(x_[j, : l[j]].detach().cpu())  # L, C
        #################

        ####### Debug get fk joints from smplrelvec #####
        data_endecoder.set_cfg(batch)
        ayfz_batch = data_endecoder.convert2ayfzdata(batch)
        # x = data_endecoder.encode(batch)
        x = data_endecoder.encode(ayfz_batch)
        out = data_endecoder.decode(x)
        comp_data = ayfz_batch
        for k in out.keys():
            if isinstance(out[k], torch.Tensor) and k in comp_data.keys():
                if "global_orient" in k:
                    # axis-angle is not 1to1 mapping, so we need to compare the rotation matrix
                    out_mat = axis_angle_to_matrix(out[k])
                    comp_mat = axis_angle_to_matrix(comp_data[k])
                    e = (out_mat - comp_mat).abs().max()
                elif "body_pose" in k:
                    out_mat = axis_angle_to_matrix(out[k].reshape(*out[k].shape[:-1], -1, 3))
                    comp_mat = axis_angle_to_matrix(comp_data[k].reshape(*comp_data[k].shape[:-1], -1, 3))
                    e = (out_mat - comp_mat).abs().max()
                else:
                    e = (out[k] - comp_data[k]).abs().max()
                if e > 1e-4:
                    import ipdb

                    ipdb.set_trace()

        ayfz_data = data_endecoder.get_ayfz_data()
        ayfz_joints = ayfz_data["joints"]

        out_joints1, _ = data_endecoder.fk_forward(**out)
        # out_joints2, _ = data_endecoder.localjoints_forward(**out)
        out_joints3, _ = data_endecoder.smpl_forward(**out)

        ayfz_joints1, _ = data_endecoder.fk_forward(**ayfz_data)
        # ayfz_joints2, _ = data_endecoder.localjoints_forward(**ayfz_data)
        ayfz_joints3, _ = data_endecoder.smpl_forward(**ayfz_data)

        error1 = (ayfz_joints - out_joints1).abs().max()
        # error2 = (ayfz_joints - out_joints2).abs().max()
        error2 = 0
        error3 = (ayfz_joints - out_joints3).abs().max()
        if error1 > 1e-3 or error2 > 1e-3 or error3 > 1e-3:  # error > 1mm
            import ipdb

            ipdb.set_trace()

        wis3d_debug(ayfz_joints, TYPE, i)
        wis3d_debug(ayfz_joints1, TYPE, i, prefix="fk")
        # wis3d_debug(ayfz_joints2, TYPE, i, prefix="localjoint")
        wis3d_debug(ayfz_joints3, TYPE, i, prefix="smpl")

        wis3d_debug(ayfz_joints1, TYPE, i, prefix="ayfz-fk")
        # wis3d_debug(ayfz_joints2, TYPE, i, prefix="ayfz-localjoint")
        wis3d_debug(ayfz_joints3, TYPE, i, prefix="ayfz-smpl")
        #################

        # if i == 20:
        #     raise AssertionError
        #################
    mean, var = batch_mean_var(all_x, 10000)
    # find var might be < 0 but very close to 0 due to numerical
    var = var.abs()
    std = var**0.5
    std[std < 1e-4] = 1.0
    statistics = {"mean": mean.cpu(), "std": std.cpu()}
    import ipdb

    ipdb.set_trace()


if __name__ == "__main__":
    run_dataset()
