import torch
import torch.nn.functional as F
from torch.utils.data import default_collate
from hmr4d.utils.pylogger import Log
from einops import repeat


def densify_batch(batch):
    B = batch["B"]
    keys_to_dense = [k for k in batch.keys() if k.endswith("_flatten")]

    padding_value = {"img_seq": 0.0, "img_seq_fid": -1}
    # TODO: We don't need this anymore in the future, we will load image features from disk
    I_max = 20  # (B, I_max, 3, 224, 224), this is an estimate to prevent OOM, but in practice this will always be overriden by the max size in the batch
    batch_shape = {"img_seq": (B, I_max, 3, 224, 224), "img_seq_fid": (B, I_max)}

    for k_flatten in keys_to_dense:
        # Real keys: k_, k_flatten, k_bid
        k_ = k_flatten[:-8]
        k_bid = k_ + "_bid"
        bid_counts = torch.bincount(batch[k_bid], minlength=B)  # size of item
        bid_max = bid_counts.max().item()
        # Support non-image data, align with previous for testing.
        if bid_max == 0:
            bid_max = 12

        # Create the dense-value
        device = batch[k_flatten].device
        dtype = batch[k_flatten].dtype
        padding_v = padding_value[k_]
        shape = list(batch_shape[k_])
        shape[1] = bid_max  # Pad to the max size
        if I_max < bid_max:
            Log.warn(f"Max size of {k_} in batch is {bid_max}, which is larger than pre-set {I_max}")

        v = torch.full(shape, padding_v, dtype=dtype, device=device)

        # Put the sparse-value in the right place
        indices = torch.split(batch[k_flatten], bid_counts.tolist())
        for b, idx in enumerate(indices):
            v[b, : len(idx)] = idx

        batch[k_] = v

    return batch


def collate_fn(batch):
    """
    Args:
        batch: list of dict, each dict is a data point
    """
    # Assume all keys in the batch are the same
    # The keys that are registered as FLATTEN_COLLATE_KEYS will be flattened and create a corresponding _bid key,
    # and the rest will be default_collated
    return_dict = {}
    for k in batch[0].keys():
        # if k in FLATTEN_COLLATE_KEYS:
        #     # check if the value is already batchable
        #     # shapes = [d[k].shape for d in batch]
        #     # if all([s == shapes[0] for s in shapes]):
        #     #     return_dict[k] = default_collate([d[k] for d in batch])
        #     # else:
        #     # flatten (make batch on GPU)
        #     return_dict[k + "_flatten"] = torch.cat([d[k] for d in batch], dim=0)
        #     return_dict[k + "_bid"] = torch.cat([torch.full((d[k].shape[0],), i) for i, d in enumerate(batch)], dim=0)
        if k.startswith("meta"):  # data information, do not batch
            return_dict[k] = [d[k] for d in batch]
        else:
            return_dict[k] = default_collate([d[k] for d in batch])

    return_dict["B"] = len(batch)
    return return_dict
