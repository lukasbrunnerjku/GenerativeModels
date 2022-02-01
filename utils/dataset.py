from path import Path
from glob import glob
from typing import Optional, List, Callable
import numpy as np
import cv2 as cv
import torch
import torchvision.transforms.functional as TF
import torchvision.transforms as transforms
import pytorch_lightning as pl
from torch.utils.data import DataLoader, Dataset

# https://www.kaggle.com/kvpratama/pokemon-images-dataset


class PokeDataset(Dataset):
    def __init__(
        self,
        data_dir: str = ".",
        image_size: int = 32,
        transform: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
    ) -> None:
        super().__init__()
        data_dir = Path(data_dir) / "pokemon_jpg" / "*.jpg"
        self.files = glob(data_dir)
        self.images: Optional[List[torch.Tensor]] = None
        self.preprocess = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Resize(size=image_size),
                transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
            ]
        )
        self.transform = transform

    @property
    def preloaded(self) -> bool:
        return self.images is not None

    def imread(self, file) -> torch.Tensor:
        img: np.ndarray = cv.cvtColor(cv.imread(file), cv.COLOR_BGR2RGB)
        return self.preprocess(img)

    def preload(self) -> None:
        self.images = [self.imread(file) for file in self.files]

    def __getitem__(self, idx) -> torch.Tensor:
        if self.preloaded:
            img = self.images[idx]
        else:
            img = self.imread(self.files[idx])

        # add pseudo class "label"
        if self.transform is not None:
            img = self.transform(img)

        # we need to create additional data that does not serverly alter
        # the original distribution thus light rotations are added
        angle = np.random.uniform(-10, +10)
        img = TF.rotate(img, angle, fill=1.0)

        return img, 0

    def __len__(self):
        return len(self.files)


class PokeDataModule(pl.LightningDataModule):
    def __init__(
        self,
        data_dir: str = ".",
        batch_size: int = 64,
        num_workers: int = 0,
        size: int = 32,
        preload_data: bool = False,
        flip_probability: float = 0.0,
        pin_memory: bool = False,
    ):
        super().__init__()

        self.data_dir = data_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.size = size
        self.preload_data = preload_data
        self.flip_probability = flip_probability
        self.pin_memory = pin_memory

    def setup(self, stage=None):
        transform = None
        if self.flip_probability > 0.0:
            transform = transforms.RandomVerticalFlip(p=self.flip_probability)
        self.dataset = PokeDataset(
            data_dir=self.data_dir, image_size=self.size, transform=transform
        )

    def train_dataloader(self):
        return DataLoader(
            self.dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            shuffle=True,
        )
