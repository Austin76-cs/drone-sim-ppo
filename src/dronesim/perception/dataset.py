"""PyTorch dataset for drone gate perception training data (HDF5 format)."""
from __future__ import annotations

from pathlib import Path

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset


class GatePerceptionDataset(Dataset):
    """Loads (image, heatmap) pairs from an HDF5 file produced by generate_data.py.

    Images are returned as float32 tensors in [0, 1] with shape (3, H, W).
    Heatmaps are float32 tensors in [0, 1] with shape (1, H, W).

    Args:
        h5_path: Path to the HDF5 dataset file.
        augment: Whether to apply random augmentations.
        in_memory: If True, load the entire dataset into RAM at init time.
                   Eliminates all disk I/O during training. Recommended when
                   you have sufficient RAM (dataset is ~14 GB).
    """

    def __init__(
        self,
        h5_path: str | Path,
        augment: bool = True,
        in_memory: bool = False,
    ) -> None:
        self.h5_path = Path(h5_path)
        self.augment = augment

        if in_memory:
            print(f"Loading dataset into RAM from {h5_path} ...")
            with h5py.File(self.h5_path, "r") as f:
                # Store images as uint8 to save RAM; convert to float in __getitem__.
                self._images   = f["images"][:]    # (N, H, W, 3) uint8
                self._heatmaps = f["heatmaps"][:]  # (N, H, W) float32
            self._len = self._images.shape[0]
            print(f"Loaded {self._len:,} samples into RAM.")
            # Mark as in-memory so __getitem__ skips HDF5.
            self._file = None
            self._in_memory = True
        else:
            with h5py.File(self.h5_path, "r") as f:
                self._len = f["images"].shape[0]
            # Keep the file closed between __getitem__ calls to support multiple workers.
            # The file is opened lazily per worker via _open().
            self._file: h5py.File | None = None
            self._in_memory = False

    def _open(self) -> h5py.File:
        if self._file is None:
            self._file = h5py.File(self.h5_path, "r")
        return self._file

    def __len__(self) -> int:
        return self._len

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        if self._in_memory:
            img = self._images[idx].astype(np.float32) / 255.0  # (H, W, 3)
            hm  = self._heatmaps[idx].astype(np.float32)         # (H, W)
        else:
            f = self._open()
            img = f["images"][idx].astype(np.float32) / 255.0   # (H, W, 3)
            hm  = f["heatmaps"][idx].astype(np.float32)          # (H, W)

        # HWC → CHW
        img = torch.from_numpy(img).permute(2, 0, 1)          # (3, H, W)
        hm  = torch.from_numpy(hm).unsqueeze(0)               # (1, H, W)

        if self.augment:
            img, hm = self._augment(img, hm)

        return img, hm

    @staticmethod
    def _augment(
        img: torch.Tensor, hm: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Horizontal flip + brightness/contrast jitter."""
        if torch.rand(1).item() < 0.5:
            img = torch.flip(img, dims=[2])
            hm  = torch.flip(hm,  dims=[2])

        # Brightness / contrast
        factor = 0.8 + torch.rand(1).item() * 0.4  # [0.8, 1.2]
        img = torch.clamp(img * factor, 0.0, 1.0)
        bias = (torch.rand(1).item() - 0.5) * 0.1  # [-0.05, 0.05]
        img = torch.clamp(img + bias, 0.0, 1.0)

        return img, hm

    def __del__(self) -> None:
        if self._file is not None:
            try:
                self._file.close()
            except Exception:
                pass


def split_dataset(
    h5_path: str | Path,
    val_fraction: float = 0.1,
    in_memory: bool = False,
) -> tuple[GatePerceptionDataset, GatePerceptionDataset]:
    """Return (train_dataset, val_dataset) with a random split."""
    from torch.utils.data import Subset

    full = GatePerceptionDataset(h5_path, augment=True, in_memory=in_memory)
    n = len(full)
    n_val = max(1, int(n * val_fraction))
    indices = torch.randperm(n).tolist()
    val_ds = Subset(GatePerceptionDataset(h5_path, augment=False, in_memory=in_memory), indices[:n_val])
    train_ds = Subset(full, indices[n_val:])
    return train_ds, val_ds  # type: ignore[return-value]
