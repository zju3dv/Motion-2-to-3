import torch
from pprint import pprint

from hmr4d.utils.hml3d import (
    convert_motion_to_hmlvec263,
    convert_hmlvec263_to_motion,
    REPEAT_LAST_FRAME,
    ZERO_FRAME_AHEAD,
    readable_hml263_vec,
)
from hmr4d.utils.pylogger import Log
from hmr4d.utils.skeleton_motion_visualization import SkeletonAnimationGenerator

if __name__ == "__main__":
    abs_hml263 = False
    velocity_padding_strategy = ZERO_FRAME_AHEAD
    velocity_padding_strategy = REPEAT_LAST_FRAME

    data_raw = torch.load("inputs/debug/motion_joints.pt")
    motion = data_raw["data"]
    motion_len = data_raw["length"]

    hml263 = convert_motion_to_hmlvec263(
        motion.clone(),
        seq_len=motion_len,
        return_abs=abs_hml263,
        smooth_motion=False,
        velocity_padding_strategy=velocity_padding_strategy,
    )

    # hml263_0 = convert_motion_to_hmlvec263(
    #     motion.clone(),
    #     return_abs=abs_hml263,
    #     smooth_motion=False,
    #     velocity_padding_strategy=ZERO_FRAME_AHEAD,
    # )

    # hml263_f = convert_motion_to_hmlvec263(
    #     motion.clone(),
    #     return_abs=abs_hml263,
    #     smooth_motion=False,
    #     velocity_padding_strategy=REPEAT_LAST_FRAME,
    # )

    # r_pos_i = hml263[..., 0]
    motion_rec = convert_hmlvec263_to_motion(
        hml263,
        seq_len=motion_len,
        abs_3d=abs_hml263,
        from_velocity_padding_strategy=velocity_padding_strategy,
    )

    # delta = hml263_0 - hml263_f
    # pprint(readable_hml263_vec(delta[0, :, 1], abs=False))
    delta = motion_rec - motion

    # Log.info(motion[0, 100:105])
    # Log.info(motion_rec[0, 100:105])
    # Log.info(hml263.shape)
    Log.info(hml263.transpose(1, 2)[0, 101:104])

    # Log.info(f"scale: {motion.abs().max()}")
    # Log.info(f"scale: {motion.abs().mean()}")
    # Log.info(f"scale: {motion.abs().std()}")
    Log.info(f"max delta: {delta.abs().max()}")
    Log.info(f"mean delta: {delta.abs().mean()}")
    Log.info(f"std delta: {delta.abs().std()}")
    # pprint(readable_hml263_vec(hml263_f[0, 26], abs=False))
