import pytorch_lightning as pl
from torch.utils.data import DataLoader
from torch.utils.data.dataloader import default_collate
from hydra.utils import instantiate

from hmr4d.utils.pylogger import Log


class SAHMR_PoseDM(pl.LightningDataModule):
    """Lighting data module which maintain the dataset and dataloader."""

    def __init__(self, opt, **kwargs):
        super().__init__()
        self.opt = opt
        for split in ["train", "val", "test"]:
            # instantiate split if exists
            if split not in opt:
                Log.warn(f"{split} dataset not founded in config files")
                continue
            dataset = instantiate(opt.get(split))
            setattr(self, f"{split}set", dataset)

    def train_dataloader(self):
        if hasattr(self, "trainset"):
            return DataLoader(
                self.trainset,
                shuffle=True,
                num_workers=self.opt.train_loader.num_workers,
                persistent_workers=True and self.opt.train_loader.num_workers > 0,
                pin_memory=True,
                batch_size=self.opt.train_loader.batch_size,
                collate_fn=collate_fn_wrapper,
            )
        else:
            return super().train_dataloader()

    def val_dataloader(self):
        if hasattr(self, "valset"):
            return DataLoader(
                self.valset,
                shuffle=False,
                num_workers=self.opt.val_loader.num_workers,
                persistent_workers=True and self.opt.val_loader.num_workers > 0,
                pin_memory=True,
                batch_size=self.opt.val_loader.batch_size,
                collate_fn=collate_fn_wrapper,
            )
        else:
            return super().val_dataloader()

    def test_dataloader(self):
        if hasattr(self, "testset"):
            return DataLoader(
                self.testset,
                shuffle=False,
                num_workers=self.opt.test_loader.num_workers,
                persistent_workers=True and self.opt.test_loader.num_workers > 0,
                pin_memory=True,
                batch_size=self.opt.test_loader.batch_size,
                collate_fn=collate_fn_wrapper,
            )
        else:
            return super().test_dataloader()


def collate_fn_wrapper(batch):
    """Collate function for dataloader."""
    keys_to_collate_as_list = ["meta"]
    list_in_batch = {}
    for k in keys_to_collate_as_list:
        if k in batch[0]:
            list_in_batch[k] = [data[k] for data in batch]
    # use default collate for the rest of batch
    batch = default_collate(batch)
    batch.update({k: v for k, v in list_in_batch.items()})
    return batch
