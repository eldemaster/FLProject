#!/usr/bin/env python3
"""
client_main.py — Entry point for a Flower client on Raspberry Pi.
Assumptions:
- Local shard at --datapath as NPZ with arrays: X [N, d] float32, y [N] int64 in 0..C-1
- Vector input model (MLP). Dataset prep/feature extraction is done upstream.
"""

import argparse, os, sys, time
from pathlib import Path

import torch
import flwr as fl

from models import build_mlp
from dataset import load_partition_npz, make_train_loader, infer_dims
from fl_client import TorchNumPyClient
from utils import set_seeds, set_cpu_threads, install_sigterm, ensure_dir

def parse_args():
    p = argparse.ArgumentParser(description="DFL Client")
    p.add_argument("--server",   required=True, help="Flower server host:port (e.g., 192.168.1.10:8081)")
    p.add_argument("--datapath", required=True, help="Path to partition.npz (with X and y)")
    p.add_argument("--logdir",   default=str(Path.home() / "dfl" / "logs"), help="Directory for local logs")
    p.add_argument("--task",     default="generic", choices=["mnist","har","wisdm","generic"],
                   help="Used only for naming logs; does not change the model")
    p.add_argument("--hsize",    type=int, default=128, help="Hidden units per MLP layer")
    p.add_argument("--batch",    type=int, default=64,  help="Initial batch size (server may override)")
    p.add_argument("--epochs",   type=int, default=2,   help="Initial local epochs (server may override)")
    p.add_argument("--lr",       type=float, default=1e-3, help="Initial LR (server may override)")
    p.add_argument("--workers",  type=int, default=0,   help="DataLoader workers; keep 0 on Pi")
    p.add_argument("--threads",  type=int, default=2,   help="OMP/torch threads; Pi3 may use 1")
    p.add_argument("--seed",     type=int, default=42,  help="Deterministic-ish seeding")
    return p.parse_args()

def main():
    args = parse_args()
    install_sigterm()
    set_seeds(args.seed)
    set_cpu_threads(args.threads)

    # Ensure dirs
    ensure_dir(args.logdir)
    task_logdir = Path(args.logdir) / args.task
    ensure_dir(task_logdir)
    csv_log = task_logdir / "train.csv"

    # Load local partition and create loader
    ds = load_partition_npz(args.datapath)
    d_in, d_out = infer_dims(ds)
    train_loader = make_train_loader(ds, batch=args.batch, workers=args.workers)

    # Build compact MLP
    device = torch.device("cpu")
    model = build_mlp(d_in=d_in, d_hidden=args.hsize, d_out=d_out).to(device)

    # Print a short banner (helps when many services run)
    print(f"[client] task={args.task} d_in={d_in} d_out={d_out} "
          f"batch={args.batch} epochs={args.epochs} lr={args.lr} "
          f"threads={args.threads} workers={args.workers}", flush=True)

    # Flower client
    client = TorchNumPyClient(model=model,
                              train_loader=train_loader,
                              device=device,
                              csv_log_path=str(csv_log))

    # Start
    fl.client.start_numpy_client(server_address=args.server, client=client)

if __name__ == "__main__":
    sys.exit(main())