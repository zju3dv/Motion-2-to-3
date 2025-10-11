from .hmr2 import HMR2

HMR2A_CKPT = f"inputs/checkpoints/hmr2/epoch=10-step=25000.ckpt"  # this is HMR2.0a, follow WHAM


def load_hmr2(checkpoint_path=HMR2A_CKPT):
    from pathlib import Path
    from .hmr2 import HMR2
    from .configs import get_config

    model_cfg = str((Path(__file__).parent / "configs/model_config.yaml").resolve())
    model_cfg = get_config(model_cfg)

    # Override some config values, to crop bbox correctly
    if (model_cfg.MODEL.BACKBONE.TYPE == "vit") and ("BBOX_SHAPE" not in model_cfg.MODEL):
        model_cfg.defrost()
        assert (
            model_cfg.MODEL.IMAGE_SIZE == 256
        ), f"MODEL.IMAGE_SIZE ({model_cfg.MODEL.IMAGE_SIZE}) should be 256 for ViT backbone"
        model_cfg.MODEL.BBOX_SHAPE = [192, 256]  # (W, H)
        model_cfg.freeze()

    model = HMR2.load_from_checkpoint(checkpoint_path, strict=False, cfg=model_cfg)
    return model
