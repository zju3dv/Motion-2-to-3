import pytorch_lightning as pl
from pytorch_lightning.utilities.combined_loader import CombinedLoader
from hydra.utils import instantiate, get_method
from torch.utils.data import DataLoader, ConcatDataset, Subset
from omegaconf import ListConfig, DictConfig
from hmr4d.utils.pylogger import Log
from numpy.random import choice


class GeneralDataModule(pl.LightningDataModule):
    def __init__(
        self, name, dataset_opts: DictConfig, loader_opts: DictConfig, collate_fn=None, limit_each_trainset=None
    ):
        """This is a general datamodule that can be used for any dataset.
        Train uses ConcatDataset
        Val and Test use CombinedLoader, sequential, completely consumes ecah iterable sequentially, and returns a triplet (data, idx, iterable_idx)

        Args:
            name: used by other module
            dataset_opts: the target of the dataset. e.g. dataset_opts.train = {_target_: ..., limit_size: None}
            loader_opts: the options for the dataset
            limit_each_trainset: limit the size of each dataset, None means no limit, useful for debugging
        """
        super().__init__()
        self.name = name
        self.loader_opts = loader_opts
        self.collate_fn = get_method(collate_fn) if collate_fn else None
        self.limit_each_trainset = limit_each_trainset

        # Train uses concat dataset
        if "train" in dataset_opts:
            assert "train" in self.loader_opts, "train not in loader_opts"
            split = "train"
            split_opts = dataset_opts.get(split)
            if isinstance(split_opts, ListConfig):
                # multiple datasets into one
                dataset = []
                for split_opts_i in split_opts:
                    dataset_i = instantiate(split_opts_i)
                    if self.limit_each_trainset:
                        dataset_i = Subset(dataset_i, choice(len(dataset_i), self.limit_each_trainset))
                    dataset.append(dataset_i)
                dataset = ConcatDataset(dataset)
            else:
                # single dataset
                dataset = instantiate(split_opts)
                if self.limit_each_trainset:
                    dataset = Subset(dataset, choice(len(dataset_i), self.limit_each_dataset))

            setattr(self, f"{split}set", dataset)
            Log.info(f"[Dataset]: Split={split}, Dataset size={len(dataset)}")

        # Val and Test use sequential dataset
        for split in ("val", "test"):
            if split not in dataset_opts:
                continue
            assert split in self.loader_opts, f"split={split} not in loader_opts"
            split_opts = dataset_opts.get(split)
            if isinstance(split_opts, ListConfig):
                # multiple datasets into one
                dataset = []
                for split_opts_i in split_opts:
                    dataset_i = instantiate(split_opts_i)
                    dataset.append(dataset_i)
                    Log.info(f"[Dataset]: Split={split}, Dataset size={len(dataset_i)}, {split_opts_i._target_}")
            else:
                # single dataset
                dataset = [instantiate(split_opts)]
                Log.info(f"[Dataset]: Split={split}, Dataset size={len(dataset[0])}, {split_opts._target_}")
            setattr(self, f"{split}sets", dataset)

    def train_dataloader(self):
        Log.info(f"train_dataloader: {len(self.trainset)}")
        if hasattr(self, "trainset"):
            return DataLoader(
                self.trainset,
                shuffle=True,
                num_workers=self.loader_opts.train.num_workers,
                persistent_workers=True and self.loader_opts.train.num_workers > 0,
                batch_size=self.loader_opts.train.batch_size,
                drop_last=True,
                collate_fn=self.collate_fn,
            )
        else:
            return super().train_dataloader()

    def val_dataloader(self):
        if hasattr(self, "valsets"):
            loaders = []
            for valset in self.valsets:
                loaders.append(
                    DataLoader(
                        valset,
                        shuffle=False,
                        num_workers=self.loader_opts.val.num_workers,
                        persistent_workers=True and self.loader_opts.val.num_workers > 0,
                        batch_size=self.loader_opts.val.batch_size,
                        collate_fn=self.collate_fn,
                    )
                )
            return CombinedLoader(loaders, mode="sequential")
        else:
            return None

    def test_dataloader(self):
        if hasattr(self, "testsets"):
            loaders = []
            for testset in self.testsets:
                loaders.append(
                    DataLoader(
                        testset,
                        shuffle=False,
                        num_workers=self.loader_opts.test.num_workers,
                        persistent_workers=False,
                        batch_size=self.loader_opts.test.batch_size,
                        collate_fn=self.collate_fn,
                    )
                )
            return CombinedLoader(loaders, mode="sequential")
        else:
            return super().test_dataloader()
