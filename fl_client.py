from typing import Dict, List, Tuple
import time, csv, os

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import flwr as fl

from utils import append_csv_row

class TorchNumPyClient(fl.client.NumPyClient):
    """Flower NumPyClient that trains a PyTorch model on a local TensorDataset."""
    def __init__(self, model: torch.nn.Module, train_loader: DataLoader, device: torch.device, csv_log_path: str):
        self.model = model
        self.train_loader = train_loader
        self.device = device
        self.criterion = nn.CrossEntropyLoss()
        self.csv_log_path = csv_log_path

    # ---- parameter serialization ----
    def get_parameters(self, config: Dict) -> List[np.ndarray]:
        return [p.detach().cpu().numpy() for p in self.model.state_dict().values()]

    def set_parameters(self, parameters: List[np.ndarray]) -> None:
        state_dict = self.model.state_dict()
        for (k, _), v in zip(state_dict.items(), parameters):
            state_dict[k] = torch.tensor(v)
        self.model.load_state_dict(state_dict, strict=True)

    # ---- training ----
    def fit(self, parameters: List[np.ndarray], config: Dict) -> Tuple[List[np.ndarray], int, Dict]:
        self.set_parameters(parameters)

        # server-pushed (with local fallbacks)
        epochs     = int(config.get("local_epochs", 2))
        lr         = float(config.get("lr", 1e-3))
        batch_size = int(config.get("batch_size", self.train_loader.batch_size or 64))

        # If the server changes batch size, rebuild the loader (optional)
        if batch_size != (self.train_loader.batch_size or 64):
            ds = self.train_loader.dataset
            from dataset import make_train_loader
            self.train_loader = make_train_loader(ds, batch=batch_size, workers=0)

        opt = optim.Adam(self.model.parameters(), lr=lr)
        self.model.train()

        start = time.time()
        for ep in range(1, epochs + 1):
            total_loss, total_n = 0.0, 0
            for xb, yb in self.train_loader:
                xb, yb = xb.to(self.device), yb.to(self.device)
                opt.zero_grad(set_to_none=True)
                loss = self.criterion(self.model(xb), yb)
                loss.backward()
                opt.step()
                bs = xb.size(0)
                total_loss += loss.item() * bs
                total_n    += bs
            avg_loss = total_loss / max(1, total_n)
            append_csv_row(self.csv_log_path, dict(
                epoch=ep, batch=batch_size, lr=lr, loss=round(avg_loss, 6)
            ))
            print(f"[client] epoch {ep}/{epochs}  loss={avg_loss:.4f}", flush=True)

        wall = time.time() - start
        print(f"[client] local_fit: epochs={epochs} wall={wall:.1f}s", flush=True)
        return self.get_parameters(config), len(self.train_loader.dataset), {}

    # ---- optional local evaluation (skipped for speed) ----
    def evaluate(self, parameters: List[np.ndarray], config: Dict) -> Tuple[float, int, Dict]:
        return 0.0, 0, {}