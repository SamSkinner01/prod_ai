import torch
from torch.utils.data import Dataset
from torchvision import datasets
from typing import List, Tuple
import numpy as np


class CIFAR10Dataset(Dataset):
    """
    CIFAR-10 Dataset wrapper with support for batch reading via __getitems__.

    This dataset downloads and prepares CIFAR-10 data, and supports both single
    and batch item retrieval for efficient data loading.

    CIFAR-10 consists of 60000 32x32 color images in 10 classes.
    """

    def __init__(self, root: str = "./data", train: bool = True, download: bool = True):
        """
        Initialize CIFAR-10 dataset.

        Args:
            root: Root directory for storing dataset
            train: If True, creates dataset from training set, otherwise test set
            download: If True, downloads the dataset if not already present
        """
        # Load CIFAR-10 dataset without transforms (we'll apply transforms on GPU)
        self.cifar = datasets.CIFAR10(
            root=root,
            train=train,
            download=download,
            transform=None  # No CPU transforms, we'll do GPU transforms later
        )

    def __len__(self) -> int:
        """Return the total number of samples."""
        return len(self.cifar)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        """
        Get a single item from the dataset.

        Args:
            idx: Index of the item to retrieve

        Returns:
            Tuple of (image, label) where image is a tensor and label is an integer
        """
        image, label = self.cifar[idx]

        # Convert PIL Image to tensor and normalize to [0, 1]
        if not isinstance(image, torch.Tensor):
            image = torch.from_numpy(np.array(image)).float() / 255.0
            # Convert from (H, W, C) to (C, H, W)
            image = image.permute(2, 0, 1)

        return image, label

    def __getitems__(self, indices: List[int]) -> List[Tuple[torch.Tensor, int]]:
        """
        Get multiple items from the dataset in batch for efficient reading.

        This method is called by DataLoader when using batched reading,
        which can be more efficient than calling __getitem__ multiple times.

        Args:
            indices: List of indices to retrieve

        Returns:
            List of (image, label) tuples
        """
        batch = []
        for idx in indices:
            image, label = self.__getitem__(idx)
            batch.append((image, label))

        return batch
