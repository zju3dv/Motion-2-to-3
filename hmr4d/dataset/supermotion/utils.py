import torch


def fid_to_imgseq_fid(fid, max_length):
    """
    Returns:
        imgseq_fid : (max_length, ), -1 indicates the invalid frame
    """
    imgseq_fid = torch.ones((max_length,), dtype=torch.long) * -1
    imgseq_fid[: len(fid)] = torch.tensor(fid, dtype=torch.long)
    return imgseq_fid
