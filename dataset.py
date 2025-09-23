from typing import Tuple
import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader

def load_partition_npz(npz_path: str) -> TensorDataset:
    """Load local NPZ shard with arrays X (float32) and y (int64, labels 0..C-1)."""
    part = np.load(npz_path)
    X = torch.from_numpy(part["X"]).float()
    y = torch.from_numpy(part["y"]).long()
    return TensorDataset(X, y)

def make_train_loader(ds: TensorDataset, batch: int, workers: int = 0) -> DataLoader:
    """Single-loader setup for local training; keep workers=0 on Raspberry Pi."""
    return DataLoader(ds, batch_size=batch, shuffle=True, num_workers=workers, pin_memory=False)

def infer_dims(ds: TensorDataset) -> Tuple[int, int]:
    x0, _ = ds[0]
    d_in = int(x0.numel())
    ys = torch.stack([y for _, y in ds])
    d_out = int(torch.max(ys).item() + 1)  # assumes labels are 0..C-1
    return d_in, d_out