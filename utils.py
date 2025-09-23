import os, sys, csv, random
from pathlib import Path
from typing import Dict
import numpy as np
import torch
import signal

def set_seeds(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

def set_cpu_threads(n: int = 2):
    os.environ.setdefault("OMP_NUM_THREADS", str(n))
    try:
        torch.set_num_threads(int(os.environ["OMP_NUM_THREADS"]))
    except Exception:
        pass

def install_sigterm():
    def _handler(signum, frame):
        print("[client] SIGTERM received; exiting.", flush=True)
        sys.exit(0)
    signal.signal(signal.SIGTERM, _handler)

def ensure_dir(path: str):
    Path(path).mkdir(parents=True, exist_ok=True)

def append_csv_row(path: str, row: Dict):
    new_file = not Path(path).exists()
    with open(path, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(row.keys()))
        if new_file:
            w.writeheader()
        w.writerow(row)