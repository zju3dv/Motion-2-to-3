# This file is a run-script to examine inidividual files
import torch
from omegaconf import OmegaConf
from hydra.utils import instantiate


def run_dataset():
    def get_dataset(TYPE):
        if TYPE == "RICH_MOTION":
            from hmr4d.dataset.rich.rich_motion import Dataset

            configs = [
                {"split": "val", "return_text": True},
                {"split": "val", "cond_img_source": 0},  # ~0.6 it/s
                {"split": "val", "cond_img_source": 1},  # ~3.1 it/s
                {"split": "val", "cond_img_source": 2},  # ~12.3 it/s; x3 when not reading images
                {"split": "test", "cond_img_source": 2, "start_frame_interval": 60},
            ]
            cfg = configs[4]
            return Dataset(**cfg)
        elif TYPE == "RICH_MOTION_V2":
            from hmr4d.dataset.rich.rich_motion_v2 import RichMotionV2Dataset

            return RichMotionV2Dataset()
        elif TYPE == "RICH_MOTION_V2_VAL":
            from hmr4d.dataset.rich.rich_motion_v2 import RichMotionV2Dataset

            return RichMotionV2Dataset(split="val")
        elif TYPE == "RICH_MOTION2D":
            from hmr4d.dataset.rich.rich_motion2d import Dataset

            configs = [{"split": "test"}]
            cfg = configs[0]
            dataset = Dataset(**cfg)
            return dataset
        elif TYPE == "RICH_MOTION_TEST":
            from hmr4d.dataset.rich.rich_motion_test import Dataset, FullSeqDataset

            # dataset = Dataset()
            dataset = FullSeqDataset()
            return dataset
        elif TYPE == "RICH_SMPL_TEST":
            from hmr4d.dataset.rich.rich_smpl_test import RichSmplFullSeqDataset

            dataset = RichSmplFullSeqDataset()
            return dataset
        elif TYPE == "AMASS_MOTION":
            from hmr4d.dataset.amass.amass_sm import SMDataset

            dataset = SMDataset(root="inputs/amass/smpl22_joints3d_neutral.pth", debug_mode=True)
            return dataset
        elif TYPE == "AMASS_SMPL":
            from hmr4d.dataset.amass.amass_smpl import SMDataset

            dataset = SMDataset()
            return dataset
        elif TYPE == "BEDLAM_MOTION":
            from hmr4d.dataset.bedlam.bedlam_motion import BedlamMotionDataset

            dataset = BedlamMotionDataset()
            return dataset
        elif TYPE == "SAMP_MOTION":
            from hmr4d.dataset.samp.samp_sm import SAMPDataset

            dataset = SAMPDataset(debug_mode=True)
            return dataset
        elif TYPE == "COUCH_MOTION":
            from hmr4d.dataset.couch.couch_sm import COUCHDataset

            dataset = COUCHDataset(debug_mode=True)
            return dataset
        elif TYPE == "AISTPP_MOTION":
            from hmr4d.dataset.aistpp.aistpp_sm import AISTPPDataset

            dataset = AISTPPDataset(debug_mode=True)
            return dataset
        elif TYPE == "BEHAVE_MOTION":
            from hmr4d.dataset.behave.behave_sm import BEHAVEDataset

            dataset = BEHAVEDataset(debug_mode=True)
            return dataset
        elif TYPE == "H36M_MOTION":
            from hmr4d.dataset.human36m.human36m_sm import Human36MDataset

            dataset = Human36MDataset(debug_mode=True)
            return dataset
        elif TYPE == "CIRCLE_MOTION":
            from hmr4d.dataset.circle.circle_sm import CIRCLEDataset

            dataset = CIRCLEDataset(debug_mode=True)
            return dataset
        elif TYPE == "Fit3D_MOTION":
            from hmr4d.dataset.fit3d.fit3d_sm import Fit3DDataset

            dataset = Fit3DDataset(debug_mode=True)
            return dataset
        elif TYPE == "EgoBody_MOTION":
            from hmr4d.dataset.egobody.egobody_sm import EgoBodyDataset

            dataset = EgoBodyDataset(debug_mode=True)
            return dataset
        elif TYPE == "GroundLink_MOTION":
            from hmr4d.dataset.groundlink.groundlink_sm import GroundLinkDataset

            dataset = GroundLinkDataset(debug_mode=True)
            return dataset
        elif TYPE == "CHI3D_MOTION":
            from hmr4d.dataset.chi3d.chi3d_sm import CHI3DDataset

            dataset = CHI3DDataset(debug_mode=True)
            return dataset
        elif TYPE == "OMOMO_MOTION":
            from hmr4d.dataset.omomo.omomo_sm import OMOMODataset

            dataset = OMOMODataset(debug_mode=True)
            return dataset
        elif TYPE == "InterHuman_MOTION":
            from hmr4d.dataset.interhuman.interhuman_sm import InterHumanDataset

            dataset = InterHumanDataset(debug_mode=True)
            return dataset
        elif TYPE == "EMDB":
            from hmr4d.dataset.emdb.emdb_motion_test import EmdbTestDataset

            dataset = EmdbTestDataset()
            return dataset
        elif TYPE == "BEDLAM_SMPL":
            from hmr4d.dataset.bedlam.bedlam_smpl import SMDataset

            return SMDataset(root="inputs/bedlam/sm_support", limit_size=None, max_motion_time=10)

    TYPE = "BEDLAM_SMPL"
    dataset = get_dataset(TYPE)
    print(len(dataset))
    data = dataset[0]
    from torch.utils.data import DataLoader
    from tqdm import tqdm
    from hmr4d.dataset.supermotion.collate import collate_fn

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
    for batch in tqdm(loader):
        i += 1
        # if i == 20:
        #     raise AssertionError
        # time.sleep(0.2)
        pass


def run_datamodule():
    cfg = OmegaConf.load("hmr4d/configs/data/supermotion3d/AmassHml3dRich.yaml")
    datamodule = instantiate(cfg, _recursive_=False)

    trianloader = datamodule.train_dataloader()
    for batch in trianloader:
        print(batch.keys())
        break


def run_network():
    # cfg = OmegaConf.load("hmr4d/configs/network/mdm/mdm_2d.yaml")
    # cfg = OmegaConf.load("hmr4d/configs/network/supermotion/prior3d_mca.yaml")
    cfg = OmegaConf.load("hmr4d/configs/network/supermotion/prior3d_bertlike.yaml")
    network = instantiate(cfg, _recursive_=False)
    B, L = 3, 120
    I = 15

    # fmt: off
    from hmr4d.dataset.supermotion.utils import fid_to_imgseq_fid
    fids = [[0, 3, 6], [0,], []]
    f_imgseq_fid = torch.stack([fid_to_imgseq_fid(fid, I) for fid in fids], dim=0)  # (B, I)
    # fmt: on

    model_kwargs = {
        "x": torch.randn((B, 263, L)),
        "timesteps": torch.randint(0, 1000, (B,)),
        "length": torch.tensor([40, 120, 30], dtype=torch.long),
        "f_text": torch.randn((B, 77, 512)),
        "f_text_length": torch.tensor([10, 77, 20], dtype=torch.long),
        "f_imgseq": torch.randn((B, I, 512)),
        "f_imgseq_fid": f_imgseq_fid,
    }
    network(**model_kwargs)

    print("Hello")


def run_pipeline():
    # cfg = OmegaConf.load("hmr4d/configs/model/sahmr4d/network/hmr4d_prior2d3d_pieline.yaml")
    # print(cfg)
    # pipeline = instantiate(cfg, _recursive_=False)

    from pathlib import Path
    from hydra import initialize_config_dir, compose

    file_path = Path(__file__).absolute()
    proj_folder = file_path.parent.parent
    with initialize_config_dir(version_base=None, config_dir=f"{proj_folder}/hmr4d/configs"):
        overrides = ["exp=motion3d_prior/baseline/baseline"]
        cfg = compose(config_name="train", overrides=overrides)

    print(cfg.model.pipeline)
    pipeline = instantiate(cfg.model.pipeline, _recursive_=False)
    pass


def run_clip():
    cfg = OmegaConf.load("hmr4d/configs/network/clip/clip_text_vision.yaml")
    clip_encoder = instantiate(cfg, _recursive_=False)
    B, I = 2, 3
    imgseq = torch.rand((B, I, 3, 224, 224))
    imfseq_fid = torch.tensor([[0, 4, 5], [0, 1, -1]])
    clip_encoder.encode_imgseq(imgseq, imfseq_fid)
    prompt = ["hello world!", ""]
    clip_encoder.encode_text(prompt, True)
    print("Hello")


if __name__ == "__main__":
    run_dataset()
    # run_datamodule()
    # run_network()
    # run_pipeline()
    # run_clip()
