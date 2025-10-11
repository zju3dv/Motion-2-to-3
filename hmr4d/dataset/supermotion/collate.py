import torch
import numpy as np
import torch.nn.functional as F
from torch.utils.data import default_collate
from hmr4d.utils.pylogger import Log
from einops import repeat

# Training
MOTIONSMPL_KEYS = [
    "transl",
    "global_orient",
    "body_pose",
    "text",
    "f_imgseq",
    "skeleton",
    "transl_incam",
    "global_orient_incam",
    "cam_angvel",
]

# Testing
MOTION3D_GEN_ADDITIONAL_KEYS = ["task", "length", "ori_length", "ori_motion", "word_embs", "pos_onehot", "text_len"]


def pad_to_max_len(x, max_len, dim=0):
    """Pad the data to max_len along dim, always repeat last frame
    Repeating the last frame is necessary for motion data to compute a valid speed
    """
    if isinstance(x, np.ndarray):
        x = torch.from_numpy(x).float()
    assert isinstance(x, torch.Tensor)
    if x.shape[dim] == max_len:
        return x
    elif x.shape[dim] < max_len:
        x = x.clone()
        x = x.transpose(0, dim)
        x = torch.cat([x, repeat(x[-1:], "b ... -> (b r) ...", r=max_len - x.shape[0])])
        x = x.transpose(0, dim)
        return x
    else:
        raise ValueError(f"Unexpected length v.s. max_len: {x.shape[0]} v.s. {max_len}")


def collate_fn(batch):
    """
    Args:
        batch: list of dict, each dict is a data point
    """
    # Assume all keys in the batch are the same
    return_dict = {}
    for k in batch[0].keys():
        if k.startswith("meta"):  # data information, do not batch
            return_dict[k] = [d[k] for d in batch]
        else:
            return_dict[k] = default_collate([d[k] for d in batch])
    return_dict["B"] = len(batch)
    return return_dict


def set_default_values_motionsmpl(data, target_keys, max_len=300, d_imgseq=1024):
    return_data = {}
    for k in target_keys:
        if k == "transl":  # Set up "transl" and "length"
            L = data[k].shape[0]
            return_data[k] = pad_to_max_len(data[k], max_len, dim=0)
            return_data["length"] = L if "length" not in data else data["length"]
        elif k == "global_orient":  # Set up "global_orient"
            return_data[k] = pad_to_max_len(data[k], max_len, dim=0)
        elif k == "body_pose":  # Set up "body_pose"
            return_data[k] = pad_to_max_len(data[k], max_len, dim=0)
        elif k == "f_imgseq":  # Set up "f_imgseq" and "has_imgseq"
            if "f_imgseq" in data:
                return_data[k] = pad_to_max_len(data[k], max_len, dim=0)
                return_data["has_imgseq"] = True
            else:
                return_data[k] = torch.zeros((max_len, d_imgseq))
                return_data["has_imgseq"] = False
        elif k == "cam_angvel":
            if "cam_angvel" in data:
                return_data[k] = pad_to_max_len(data[k], max_len, dim=0)
                return_data["has_cam_angvel"] = True
            else:
                return_data[k] = torch.zeros((max_len, 6))
                return_data["has_cam_angvel"] = False
        elif k == "transl_incam":
            if "transl_incam" in data:
                return_data[k] = pad_to_max_len(data[k], max_len, dim=0)
                return_data["has_incam"] = True
            else:
                return_data[k] = torch.zeros((max_len, 3))
                return_data["has_incam"] = False
        elif k == "global_orient_incam":
            if "global_orient_incam" in data:
                return_data[k] = pad_to_max_len(data[k], max_len, dim=0)
                return_data["has_incam"] = True
            else:
                return_data[k] = torch.zeros((max_len, 3))
                return_data["has_incam"] = False

    # Add the rest of the keys
    keys_part_1 = list(return_data.keys())
    for k in data.keys():
        if k not in keys_part_1:
            return_data[k] = data[k]

    return return_data


def before_collate_motionsmpl(data, max_len=300, d_imgseq=1024):
    """Sanity check and make sure all keys are in data.
    Possible k-v pair in data:
        transl: (F, 3), in the align-y coordinate
        global_orient: (F, 3), in the align-y coordinate
        body_pose: (F, 63), in the align-y coordinate
        length: (1,) the length of the motion
        text: str
        f_imgseq: (F, D)
    """
    # Must have smplpose
    assert "transl" in data
    # Set default values
    target_keys = MOTIONSMPL_KEYS
    return_data = set_default_values_motionsmpl(data, target_keys, max_len=max_len, d_imgseq=d_imgseq)
    return return_data


def before_collate_motionsmpl_test(data, max_len=300):
    """Sanity check and make sure all keys are in data.
    Possible k-v pair in data:
        smplpose: (F, 69), in the align-y coordinate
        length: (1,) the length of the motion
    """
    # Must have smplpose
    assert "smplpose" in data
    assert data["task"] in ["GEN"], f"task should be GEN, but got {data['task']}"

    # Set default values
    target_keys = MOTIONSMPL_KEYS
    return_data = set_default_values_motionsmpl(data, target_keys, max_len=max_len)

    return return_data
