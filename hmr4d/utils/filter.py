import math
import numpy as np
import torch

import scipy.signal  # ubuntu needs this
import scipy.ndimage  # ubuntu needs this


def smoothing_factor(t_e, cutoff):
    r = 2 * math.pi * cutoff * t_e
    return r / (r + 1)


def exponential_smoothing(a, x, x_prev):
    return a * x + (1 - a) * x_prev


class OneEuroFilter:
    def __init__(self, t0, x0, dx0=0.0, min_cutoff=1.0, beta=0.0, d_cutoff=1.0):
        # https://github.com/mkocabas/VIBE/blob/master/lib/utils/one_euro_filter.py#L14
        """Initialize the one euro filter."""
        # decrease min_cutoff will remove more noise and jitter
        # decrease beta will be less sensitive to signal delta
        # decrease d_cutoff will remove more noise and jitter
        # The parameters.
        self.min_cutoff = float(min_cutoff)
        self.beta = float(beta)
        self.d_cutoff = float(d_cutoff)
        # Previous values.
        self.x_prev = x0
        self.dx_prev = dx0
        self.t_prev = t0

    def __call__(self, t, x):
        """Compute the filtered signal."""
        t_e = t - self.t_prev

        # The filtered derivative of the signal.
        a_d = smoothing_factor(t_e, self.d_cutoff)
        dx = (x - self.x_prev) / t_e
        dx_hat = exponential_smoothing(a_d, dx, self.dx_prev)

        # The filtered signal.
        cutoff = self.min_cutoff + self.beta * torch.abs(dx_hat)
        a = smoothing_factor(t_e, cutoff)
        x_hat = exponential_smoothing(a, x, self.x_prev)

        # Memorize the previous values.
        self.x_prev = x_hat
        self.dx_prev = dx_hat
        self.t_prev = t

        return x_hat


def smooth_pose_oneeurofilter(pose, min_cutof=0.004, beta=1.5, d_cutof=0.004):
    """Smooth pose using one euro filter."""
    # pose: N, T, J, C
    smoothed_pose = []
    filter = OneEuroFilter(0, pose[:, 0], min_cutoff=min_cutof, beta=beta, d_cutoff=d_cutof)
    smoothed_pose.append(pose[:, 0])
    for i in range(1, pose.shape[1]):
        smoothed_pose.append(filter(i, pose[:, i]))
    smoothed_pose = torch.stack(smoothed_pose, dim=1)

    return smoothed_pose


def smooth_pose_savgol(pose, window_length=15, polyorder=5):
    """Smooth pose using savgol filter."""
    # pose: N, T, J, C

    smoothed_pose = []
    for i in range(pose.shape[0]):
        smoothed_pose_T = []
        for j in range(pose.shape[2]):
            smoothed_pose_j = []
            for k in range(pose.shape[3]):
                traj = scipy.signal.savgol_filter(pose[i, :, j, k].detach().cpu().numpy(), window_length, polyorder)
                smoothed_pose_j.append(torch.from_numpy(traj).to(pose.device))
            smoothed_pose_j = torch.stack(smoothed_pose_j, dim=-1)
            smoothed_pose_T.append(smoothed_pose_j)
        smoothed_pose_T = torch.stack(smoothed_pose_T, dim=1)
        smoothed_pose.append(smoothed_pose_T)
    smoothed_pose = torch.stack(smoothed_pose, dim=0)

    return smoothed_pose


def smooth_pose_gaussian(pose, sigma=3):
    """Smooth pose using gaussian filter."""
    # pose: N, T, J, C

    smoothed_pose = []
    for i in range(pose.shape[0]):
        smoothed_pose_T = []
        for j in range(pose.shape[2]):
            smoothed_pose_j = []
            for k in range(pose.shape[3]):
                traj = scipy.ndimage.gaussian_filter1d(pose[i, :, j, k].detach().cpu().numpy(), sigma=sigma)
                smoothed_pose_j.append(torch.from_numpy(traj).to(pose.device))
            smoothed_pose_j = torch.stack(smoothed_pose_j, dim=-1)
            smoothed_pose_T.append(smoothed_pose_j)
        smoothed_pose_T = torch.stack(smoothed_pose_T, dim=1)
        smoothed_pose.append(smoothed_pose_T)
    smoothed_pose = torch.stack(smoothed_pose, dim=0)

    return smoothed_pose
