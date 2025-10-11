import pytorch_lightning as pl
from hydra.utils import instantiate
from torch.utils.data import DataLoader, ConcatDataset
from omegaconf import ListConfig
from hmr4d.utils.pylogger import Log

class GeneralDataModule(pl.LightningDataModule):
    def __init__(self, name, dataset_target, opt):
        """This is a general datamodule that can be used for any dataset.
        Args:
            name: used by other module
            dataset_target: the target of the dataset
            opt: the options for the dataset
        """
        super().__init__()
        self.opt = opt
        self.name = name
        for split in ("train", "val", "test"):  # Fit stage uses train and val. Test stage uses test.
            if split not in opt:
                continue
            split_opt = opt.get(split)
            # multiple datasets into one
            if isinstance(split_opt, ListConfig):
                dataset = []
                for split_opt_i in split_opt:
                    dataset_i = instantiate({"_target_": dataset_target, **dict(split_opt_i)})
                    dataset.append(dataset_i)
                dataset = ConcatDataset(dataset)
            else:  # single dataset
                dataset = instantiate({"_target_": dataset_target, **dict(split_opt)})
            setattr(self, f"{split}set", dataset)
            Log.info(f"[Dataset]: Split={split}, Dataset size={len(dataset)}")

    def train_dataloader(self):
        if hasattr(self, "trainset"):
            return DataLoader(
                self.trainset,
                shuffle=True,
                num_workers=self.opt.train_loader.num_workers,
                persistent_workers=True and self.opt.train_loader.num_workers > 0,
                batch_size=self.opt.train_loader.batch_size,
                drop_last=True,
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
                batch_size=self.opt.val_loader.batch_size,
            )
        else:
            return super().val_dataloader()

    def test_dataloader(self):
        if hasattr(self, "testset"):
            return DataLoader(
                self.testset,
                shuffle=False,
                num_workers=self.opt.test_loader.num_workers,
                persistent_workers=False,
                batch_size=self.opt.test_loader.batch_size,
            )
        else:
            return super().test_dataloader()
