import torch
import numpy as np
from hmr4d.utils.geo_transform import project_p2d, convert_bbx_xys_to_lurb, cvt_to_bi01_p2d


def estimate_focal_length(img_w, img_h):
    return (img_w**2 + img_h**2) ** 0.5  # Diagonal FOV = 2*arctan(0.5) * 180/pi = 53


def estimate_K(img_w, img_h):
    focal_length = estimate_focal_length(img_w, img_h)
    K = torch.eye(3).float()
    K[0, 0] = focal_length
    K[1, 1] = focal_length
    K[0, 2] = img_w / 2.0
    K[1, 2] = img_h / 2.0
    return K


def convert_K_to_K4(K):
    K4 = torch.stack([K[0, 0], K[1, 1], K[0, 2], K[1, 2]]).float()
    return K4


def convert_f_to_K(focal_length, img_w, img_h):
    K = torch.eye(3).float()
    K[0, 0] = focal_length
    K[1, 1] = focal_length
    K[0, 2] = img_w / 2.0
    K[1, 2] = img_h / 2.0
    return K


def resize_K(K, f=0.5):
    K = K.clone() * f
    K[..., 2, 2] = 1.0
    return K


def create_camera_sensor(width=None, height=None, f_fullframe=None):
    if width is None or height is None:
        # The 4:3 aspect ratio is widely adopted by image sensors in mobile phones.
        if np.random.rand() < 0.5:
            width, height = 1200, 1600
        else:
            width, height = 1600, 1200

    # Sample FOV from common options:
    # 1. wide-angle lenses are common in mobile phones,
    # 2. telephoto lenses has less perspective effect, which should makes it easy to learn
    if f_fullframe is None:
        f_fullframe_options = [24, 26, 28, 30, 35, 40, 50, 60, 70]
        f_fullframe = np.random.choice(f_fullframe_options)

    # We use diag to map focal-length: https://www.nikonians.org/reviews/fov-tables
    diag_fullframe = (24**2 + 36**2) ** 0.5
    diag_img = (width**2 + height**2) ** 0.5
    focal_length = diag_img / diag_fullframe * f_fullframe

    K_fullimg = torch.eye(3)
    K_fullimg[0, 0] = focal_length
    K_fullimg[1, 1] = focal_length
    K_fullimg[0, 2] = width / 2
    K_fullimg[1, 2] = height / 2

    return width, height, K_fullimg


# ====== Compute cliffcam ===== #


def convert_xys_to_cliff_cam_wham(xys, res):
    """
    Args:
        xys: (N, 3) in pixel. Note s should not be touched by 200
        res: (2), e.g. [4112., 3008.]  (w,h)
    Returns:
        cliff_cam: (N, 3), normalized representation
    """

    def normalize_keypoints_to_image(x, res):
        """
        Args:
            x: (N, 2), centers
            res: (2), e.g. [4112., 3008.]
        Returns:
            x_normalized: (N, 2)
        """
        res = res.to(x.device)
        scale = res.max(-1)[0].reshape(-1)
        mean = torch.stack([res[..., 0] / scale, res[..., 1] / scale], dim=-1).to(x.device)
        x = 2 * x / scale.reshape(*[1 for i in range(len(x.shape[1:]))]) - mean.reshape(
            *[1 for i in range(len(x.shape[1:-1]))], -1
        )
        return x

    centers = normalize_keypoints_to_image(xys[:, :2], res)  # (N, 2)
    scale = xys[:, 2:] / res.max()
    location = torch.cat((centers, scale), dim=-1)
    return location


def compute_bbox_info_bedlam(bbx_xys, K_fullimg):
    """impl as in BEDLAM
    Args:
        bbx_xys: ((B), N, 3), in pixel space described by K_fullimg
        K_fullimg: ((B), (N), 3, 3)
    Returns:
        bbox_info: ((B), N, 3)
    """
    fl = K_fullimg[..., 0, 0].unsqueeze(-1)
    icx = K_fullimg[..., 0, 2]
    icy = K_fullimg[..., 1, 2]

    cx, cy, b = bbx_xys[..., 0], bbx_xys[..., 1], bbx_xys[..., 2]
    bbox_info = torch.stack([cx - icx, cy - icy, b], dim=-1)
    bbox_info = bbox_info / fl
    return bbox_info


# ====== Convert Prediction to Cam-t ===== #


def compute_transl_full_cam(pred_cam, bbx_xys, K_fullimg):
    s, tx, ty = pred_cam[..., 0], pred_cam[..., 1], pred_cam[..., 2]
    focal_length = K_fullimg[..., 0, 0]

    icx = K_fullimg[..., 0, 2]
    icy = K_fullimg[..., 1, 2]
    sb = s * bbx_xys[..., 2]
    cx = 2 * (bbx_xys[..., 0] - icx) / (sb + 1e-9)
    cy = 2 * (bbx_xys[..., 1] - icy) / (sb + 1e-9)
    tz = 2 * focal_length / (sb + 1e-9)

    cam_t = torch.stack([tx + cx, ty + cy, tz], dim=-1)
    return cam_t


def project_to_bi01(points, bbx_xys, K_fullimg):
    """
    points: (B, L, J, 3)
    bbx_xys: (B, L, 3)
    K_fullimg: (B, L, 3, 3)
    """
    # p2d = project_p2d(points, K_fullimg)
    p2d = perspective_projection(points, K_fullimg)
    bbx_lurb = convert_bbx_xys_to_lurb(bbx_xys)
    p2d_bi01 = cvt_to_bi01_p2d(p2d, bbx_lurb)
    return p2d_bi01


def perspective_projection(points, K):
    # points: (B, L, J, 3)
    # K: (B, L, 3, 3)
    projected_points = points / points[..., -1].unsqueeze(-1)
    projected_points = torch.einsum("...ij,...kj->...ki", K, projected_points.float())
    return projected_points[..., :-1]
