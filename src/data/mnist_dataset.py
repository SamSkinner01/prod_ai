import torch
from torch.utils.data import Dataset
from torchvision import datasets
from typing import List, Tuple, Union
import numpy as np


class MNISTDataset(Dataset):
    """
    MNIST Dataset wrapper with support for batch reading via __getitems__.

    This dataset downloads and prepares MNIST data, and supports both single
    and batch item retrieval for efficient data loading.
    """

    def __init__(self, root: str = "./data", train: bool = True, download: bool = True):
        """
        Initialize MNIST dataset.

        Args:
            root: Root directory for storing dataset
            train: If True, creates dataset from training set, otherwise test set
            download: If True, downloads the dataset if not already present
        """
        # Load MNIST dataset without transforms (we'll apply transforms on GPU)
        self.mnist = datasets.MNIST(
            root=root,
            train=train,
            download=download,
            transform=None  # No CPU transforms, we'll do GPU transforms later
        )

    def __len__(self) -> int:
        """Return the total number of samples."""
        return len(self.mnist)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        """
        Get a single item from the dataset.

        Args:
            idx: Index of the item to retrieve

        Returns:
            Tuple of (image, label) where image is a tensor and label is an integer
        """
        image, label = self.mnist[idx]

        # Convert PIL Image to tensor and normalize to [0, 1]
        if not isinstance(image, torch.Tensor):
            image = torch.from_numpy(np.array(image)).float() / 255.0
            # Add channel dimension: (H, W) -> (C, H, W)
            image = image.unsqueeze(0)

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
