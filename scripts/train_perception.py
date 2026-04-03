"""Train the GateUNet on data collected by generate_data.py.

Usage:
    python scripts/train_perception.py \\
        --data data/perception_train.h5 \\
        --run-name unet_v1 \\
        --epochs 50 --batch-size 64
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from dronesim.perception.dataset import split_dataset
from dronesim.perception.unet import GateUNet


def train(args: argparse.Namespace) -> None:
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"Training on {device}")

    # Dataset
    train_ds, val_ds = split_dataset(args.data, val_fraction=0.1)
    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True,
        num_workers=args.workers, pin_memory=True, drop_last=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size * 2, shuffle=False,
        num_workers=args.workers, pin_memory=True,
    )
    print(f"Train: {len(train_ds):,}  Val: {len(val_ds):,}  "
          f"Batches/epoch: {len(train_loader)}")

    # Model
    model = GateUNet(base_ch=32).to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"GateUNet: {n_params:,} parameters")

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=1e-5
    )
    loss_fn = nn.MSELoss()

    ckpt_dir = Path("checkpoints") / args.run_name
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    writer = SummaryWriter(log_dir=f"runs/{args.run_name}")

    best_val_loss = float("inf")
    global_step = 0

    for epoch in range(1, args.epochs + 1):
        # --- Train ---
        model.train()
        train_loss = 0.0
        t0 = time.time()
        for imgs, heatmaps in train_loader:
            imgs     = imgs.to(device, non_blocking=True)
            heatmaps = heatmaps.to(device, non_blocking=True)

            preds = model(imgs)
            loss  = loss_fn(preds, heatmaps)

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            train_loss += loss.item()
            global_step += 1

        train_loss /= len(train_loader)
        scheduler.step()

        # --- Validate ---
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for imgs, heatmaps in val_loader:
                imgs     = imgs.to(device, non_blocking=True)
                heatmaps = heatmaps.to(device, non_blocking=True)
                preds    = model(imgs)
                val_loss += loss_fn(preds, heatmaps).item()
        val_loss /= len(val_loader)

        dt = time.time() - t0
        lr = optimizer.param_groups[0]["lr"]
        print(
            f"Epoch {epoch:3d}/{args.epochs}  "
            f"train={train_loss:.5f}  val={val_loss:.5f}  "
            f"lr={lr:.2e}  {dt:.1f}s"
        )
        writer.add_scalar("loss/train", train_loss, epoch)
        writer.add_scalar("loss/val",   val_loss,   epoch)
        writer.add_scalar("lr", lr, epoch)

        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(
                {"epoch": epoch, "model": model.state_dict(), "val_loss": val_loss},
                ckpt_dir / "best_model.pt",
            )

        # Periodic checkpoint
        if epoch % 10 == 0:
            torch.save(
                {"epoch": epoch, "model": model.state_dict(), "val_loss": val_loss},
                ckpt_dir / f"epoch_{epoch:03d}.pt",
            )

    writer.close()
    print(f"\nBest val loss: {best_val_loss:.5f}  → {ckpt_dir}/best_model.pt")


def main() -> None:
    parser = argparse.ArgumentParser(description="Train GateUNet perception model")
    parser.add_argument("--data",       required=True, help="Path to .h5 dataset")
    parser.add_argument("--run-name",   default="unet_v1")
    parser.add_argument("--epochs",     type=int,   default=50)
    parser.add_argument("--batch-size", type=int,   default=64)
    parser.add_argument("--lr",         type=float, default=3e-4)
    parser.add_argument("--workers",    type=int,   default=0)
    parser.add_argument("--device",     default="cuda")
    args = parser.parse_args()
    train(args)


if __name__ == "__main__":
    main()
