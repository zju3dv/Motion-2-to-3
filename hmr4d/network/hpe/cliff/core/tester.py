import os
import cv2
import torch
from einops import rearrange

from . import constants
from torchvision.transforms import Normalize
from ..utils.train_utils import load_pretrained_model

from .config import update_hparams
from ..models.hmr import HMR
from ..models.head.smplx_cam_head import SMPLXCamHead, perspective_projection

from ..path import proj_root
from pathlib import Path

from hmr4d.utils.geo.hmr_cam import compute_bbox_info_bedlam

cfg_path = f"{Path(__file__).parent}/config.yaml"  # path to config


class CLIFF_Wrapper:
    def __init__(self, ckpt, cfg=cfg_path):
        print("Setup CLIFF Wrapper: ->cuda ->no_grad()")

        self.ckpt = ckpt
        self.model_cfg = update_hparams(cfg)
        self.normalize_img = Normalize(mean=constants.IMG_NORM_MEAN, std=constants.IMG_NORM_STD)
        self.bboxes_dict = {}

        # Loading Model and weights
        self.model = self._build_model()
        self.smplx_cam_head = SMPLXCamHead(img_res=self.model_cfg.DATASET.IMG_RES).cuda()
        self._load_pretrained_model()
        self.model.eval()

        # Utils
        self.smplx2smpl = torch.load(proj_root / "hmr4d/utils/body_model/smplx2smpl_sparse.pt").cuda()
        self.smpl_J_regressor = torch.load(proj_root / "hmr4d/utils/body_model/smpl_neutral_J_regressor.pt").cuda()

    def _build_model(self):
        self.hparams = self.model_cfg
        model = HMR(
            backbone=self.hparams.MODEL.BACKBONE,
            img_res=self.hparams.DATASET.IMG_RES,
            pretrained_ckpt=self.hparams.TRAINING.PRETRAINED_CKPT,
            hparams=self.hparams,
        ).cuda()
        return model

    def _load_pretrained_model(self):
        ckpt = torch.load(self.ckpt)["state_dict"]
        load_pretrained_model(self.model, ckpt, overwrite_shape_mismatch=True, remove_lightning=True)
        print(f'Loaded pretrained weights from "{self.ckpt}"')

    @torch.no_grad()
    def run_on_images(self, batch):
        # Parse batch
        img_crops_ts = batch["img_crops_ts"]  # (B, C, 224, 224) tensor in range [0, 1], RGB
        K_fullimg = batch["K_fullimg"]  # (B, 3, 3), camera intrinsics for the original full img
        bbx_xys = batch["bbx_xys"]  # (B, 3), [cx, cy, s], bbox information in the original full img

        # Assersion
        assert img_crops_ts.shape[2] == img_crops_ts.shape[3] == 224, "The height and width of the image should be 224."

        # Forward
        imgs = self.normalize_img(img_crops_ts).cuda()  # (B, C, H, W)
        bbox_info = compute_bbox_info_bedlam(bbx_xys, K_fullimg).cuda()  # (B, 3)
        cliff_output = self.model.forward_minimal(imgs, bbox_info=bbox_info)

        return cliff_output

        """  Previous code for reference
            Rreturn:
                motion3d: (B, 22, 3) in camera coordinate
                motion2d: (B, 22, 2) in image coordinate
                smpl_verts_converted: (B, 6890, 3) in camera coordinate
                pred_cam_t: (B, 3) in camera coordinate (for rendering)
        """

        # This part is to get the joint 3d of smpl from smplx vertices.
        # device = pred["vertices"].device
        # smplx_verts = pred["vertices"].detach()  # (N, V_smplx, 3)
        # smpl_verts_converted = torch.stack([torch.matmul(self.smplx2smpl, v) for v in smplx_verts])  # (N, 6890, 3)
        # motion3d = torch.matmul(self.smpl_J_regressor[None], smpl_verts_converted)[:, :22]  # (N, 22, 3)
        # pred_cam_t = pred["pred_cam_t"]
        # torch.save(motion3d, os.path.join("test3d.pt"))

        # This part is to get the joint 2d
        # motion2d = perspective_projection(
        #     motion3d,
        #     rotation=torch.eye(3, device=device).unsqueeze(0).expand(B, -1, -1),
        #     translation=pred["pred_cam_t"],
        #     cam_intrinsics=pred["cam_intrinsics"],
        # )
        # torch.save(motion2d, os.path.join("test2d.pt"))

        # #!! ====== visualization ======
        # focal_length = (img_w * img_w + img_h * img_h) ** 0.5
        # pred_vertices_array = (pred['vertices'] + pred['pred_cam_t'].unsqueeze(1)).detach().cpu().numpy()
        # renderer = Renderer(focal_length=focal_length[0], img_w=W, img_h=H,
        #                     faces=self.smplx_cam_head.smplx.faces,
        #                     same_mesh_color=False)
        # front_view = renderer.render_front_view(pred_vertices_array,
        #                                         bg_img_rgb=ori_imgs[0].cpu().numpy().copy())

        # # save rendering results
        # front_view_path = os.path.join(".", "vis.jpg")
        # cv2.imwrite(front_view_path, front_view[:, :, ::-1])
        # renderer.delete()

        # return motion3d, motion2d, smpl_verts_converted, pred_cam_t
