"""
train.py — WGAN-GP training loop for CelebA 64×64.

All hyper-parameters live in the `Config` dataclass at the top.
Change values there — nothing is buried inside functions.

Outputs (written to Config.output_dir)
---------------------------------------
  checkpoints/   G_epoch{N}.pt / D_epoch{N}.pt
  samples/       grid_{step:06d}.png   (periodic)  +  samples_final_50.png
  samples/       interpolation.png
  training_curves.png
  losses.json

Run
---
    python train.py                       # uses defaults
    python train.py --data_root /path     # override data path
"""

from __future__ import annotations

import argparse
import os
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from dataset import get_celeba_loader
from models  import Generator, Critic, weights_init
from utils   import (
    gradient_penalty,
    LossTracker,
    plot_training_curves,
    save_image_grid,
    denormalize,
)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class Config:
    # Data
    data_root:   str  = "./data"
    max_samples: Optional[int] = None   # set e.g. 50_000 to cap on free Colab

    # Architecture
    latent_dim:   int = 128
    feature_maps: int = 64    # ngf = ndf = 64

    # Training
    num_epochs:   int   = 20
    batch_size:   int   = 64
    n_critic:     int   = 5      # critic updates per generator update
    lambda_gp:    float = 10.0   # WGAN-GP penalty coefficient
    lr:           float = 1e-4   # learning rate for both G and D
    adam_b1:      float = 0.0    # WGAN-GP paper uses beta1=0 (not 0.9)
    adam_b2:      float = 0.9

    # Logging / saving
    output_dir:        str = "./outputs"
    sample_interval:   int = 200   # save grid every N generator steps
    ckpt_interval:     int = 5     # save checkpoint every N epochs
    num_workers:       int = 2

    # Reproducibility
    seed: int = 42


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def save_checkpoint(
    G: nn.Module,
    D: nn.Module,
    opt_G: torch.optim.Optimizer,
    opt_D: torch.optim.Optimizer,
    epoch: int,
    out_dir: str | Path,
) -> None:
    ckpt_dir = Path(out_dir) / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "epoch":     epoch,
            "G":         G.state_dict(),
            "D":         D.state_dict(),
            "opt_G":     opt_G.state_dict(),
            "opt_D":     opt_D.state_dict(),
        },
        ckpt_dir / f"ckpt_epoch{epoch:03d}.pt",
    )
    print(f"  [ckpt] saved epoch {epoch}")


def load_checkpoint(
    path: str | Path,
    G: nn.Module,
    D: nn.Module,
    opt_G: torch.optim.Optimizer,
    opt_D: torch.optim.Optimizer,
    device: torch.device,
) -> int:
    """Load a checkpoint; returns the epoch number to resume from."""
    ckpt = torch.load(path, map_location=device)
    G.load_state_dict(ckpt["G"])
    D.load_state_dict(ckpt["D"])
    opt_G.load_state_dict(ckpt["opt_G"])
    opt_D.load_state_dict(ckpt["opt_D"])
    print(f"  [ckpt] resumed from epoch {ckpt['epoch']}")
    return ckpt["epoch"]


# ---------------------------------------------------------------------------
# Latent-space interpolation (for Task 1 report)
# ---------------------------------------------------------------------------

@torch.no_grad()
def latent_interpolation(
    G: nn.Module,
    device: torch.device,
    latent_dim: int,
    n_steps: int = 10,
    save_path: str | Path = "interpolation.png",
    seed: int = 0,
) -> None:
    """
    Linearly interpolate between two random z vectors.
    Generates n_steps + 2 images (endpoints included) and saves a 1-row grid.

    The assignment requires ≥ 8 intermediate steps; default n_steps=10 gives
    10 intermediate + 2 endpoints = 12 total images in the row.
    """
    G.eval()
    rng = torch.Generator(device=device)
    rng.manual_seed(seed)

    z1 = torch.randn(1, latent_dim, device=device, generator=rng)
    z2 = torch.randn(1, latent_dim, device=device, generator=rng)

    # Include both endpoints: t=0 → z1, t=1 → z2
    alphas = torch.linspace(0.0, 1.0, n_steps + 2, device=device)
    frames = []
    for a in alphas:
        z_interp = (1.0 - a) * z1 + a * z2         # linear interpolation
        img      = G(z_interp)                       # (1, 3, 64, 64)
        frames.append(img)

    grid_tensor = torch.cat(frames, dim=0)           # (n_steps+2, 3, 64, 64)
    save_image_grid(
        grid_tensor,
        path=save_path,
        nrow=len(alphas),   # all in one row
    )
    G.train()
    print(f"  [interp] saved {len(alphas)}-frame interpolation → {save_path}")


# ---------------------------------------------------------------------------
# Main training loop
# ---------------------------------------------------------------------------

def train(cfg: Config, resume_from: Optional[str] = None) -> None:
    set_seed(cfg.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device  : {device}")
    print(f"Epochs  : {cfg.num_epochs}  |  Batch : {cfg.batch_size}  |  n_critic : {cfg.n_critic}")

    # --- Output directories ---
    out       = Path(cfg.output_dir)
    samp_dir  = out / "samples"
    samp_dir.mkdir(parents=True, exist_ok=True)

    # --- Data ---
    loader: DataLoader = get_celeba_loader(
        root=cfg.data_root,
        split="train",
        batch_size=cfg.batch_size,
        num_workers=cfg.num_workers,
        attr_name=None,                  # Task 1: no conditioning
        max_samples=cfg.max_samples,
    )
    print(f"Dataset : {len(loader.dataset):,} images  →  {len(loader):,} batches/epoch")

    # --- Models ---
    G = Generator(latent_dim=cfg.latent_dim, feature_maps=cfg.feature_maps).to(device)
    D = Critic(feature_maps=cfg.feature_maps).to(device)
    G.apply(weights_init)
    D.apply(weights_init)
    print(f"G params: {sum(p.numel() for p in G.parameters()):,}")
    print(f"D params: {sum(p.numel() for p in D.parameters()):,}")

    # --- Optimisers (WGAN-GP paper: Adam, β1=0, β2=0.9) ---
    opt_G = torch.optim.Adam(G.parameters(), lr=cfg.lr, betas=(cfg.adam_b1, cfg.adam_b2))
    opt_D = torch.optim.Adam(D.parameters(), lr=cfg.lr, betas=(cfg.adam_b1, cfg.adam_b2))

    # --- Optional resume ---
    start_epoch = 0
    if resume_from is not None:
        start_epoch = load_checkpoint(resume_from, G, D, opt_G, opt_D, device)

    # --- Fixed noise for consistent sample grids across training ---
    fixed_z = torch.randn(64, cfg.latent_dim, device=device)

    # --- Loss tracker ---
    tracker   = LossTracker()
    g_step    = 0    # counts generator updates (x-axis for tracker)
    d_step    = 0    # counts all critic updates (for n_critic bookkeeping)

    # -----------------------------------------------------------------------
    # Epoch loop
    # -----------------------------------------------------------------------
    for epoch in range(start_epoch, cfg.num_epochs):
        G.train()
        D.train()

        for batch_idx, real in enumerate(loader):
            # real is a plain image tensor (no labels in Task 1)
            real = real.to(device)                   # (B, 3, 64, 64)
            B    = real.size(0)

            # ==============================================================
            # Step 1: Update CRITIC  (n_critic times per G update)
            # ==============================================================
            opt_D.zero_grad()

            # Sample noise and generate fakes (detach — critic doesn't update G)
            z    = torch.randn(B, cfg.latent_dim, device=device)
            fake = G(z).detach()

            # Wasserstein distance estimate (negative = better separation)
            d_real = D(real).mean()
            d_fake = D(fake).mean()

            # Gradient penalty
            gp = gradient_penalty(D, real, fake, device, lambda_gp=cfg.lambda_gp)

            # Critic loss: minimise → maximise D(real) - D(fake)
            # Full objective: -(D(real) - D(fake)) + λ·GP
            d_loss = -d_real + d_fake + gp

            d_loss.backward()
            opt_D.step()
            d_step += 1

            # ==============================================================
            # Step 2: Update GENERATOR  (every n_critic critic steps)
            # ==============================================================
            if d_step % cfg.n_critic == 0:
                opt_G.zero_grad()

                z    = torch.randn(B, cfg.latent_dim, device=device)
                fake = G(z)                          # fresh forward — not detached

                # Generator loss: maximise D(fake)  ↔  minimise -D(fake)
                g_loss = -D(fake).mean()

                g_loss.backward()
                opt_G.step()

                # ----------------------------------------------------------
                # Record losses (one entry per G step)
                # gp recorded as raw value before lambda scaling for plotting
                # ----------------------------------------------------------
                raw_gp = gp.item() / cfg.lambda_gp
                tracker.record(
                    g_loss=g_loss.item(),
                    d_loss=d_loss.item(),
                    gp=raw_gp,
                )
                g_step += 1

                # ----------------------------------------------------------
                # Periodic sample grid
                # ----------------------------------------------------------
                if g_step % cfg.sample_interval == 0:
                    G.eval()
                    with torch.no_grad():
                        samples = G(fixed_z)
                    save_image_grid(
                        samples,
                        path=samp_dir / f"grid_{g_step:06d}.png",
                        nrow=8,
                    )
                    G.train()

                # ----------------------------------------------------------
                # Console log
                # ----------------------------------------------------------
                if g_step % 50 == 0:
                    print(
                        f"Epoch [{epoch+1:03d}/{cfg.num_epochs}]  "
                        f"Step {g_step:5d}  |  "
                        f"D: {d_loss.item():+.4f}  "
                        f"G: {g_loss.item():+.4f}  "
                        f"GP: {raw_gp:.4f}  "
                        f"W-dist: {(d_real - d_fake).item():+.4f}"
                    )

        # --- End of epoch ---
        if (epoch + 1) % cfg.ckpt_interval == 0:
            save_checkpoint(G, D, opt_G, opt_D, epoch + 1, out)

        # Save loss JSON after every epoch (safe to interrupt)
        tracker.save(out / "losses.json")

    # -----------------------------------------------------------------------
    # Post-training outputs
    # -----------------------------------------------------------------------
    print("\n=== Training complete. Generating final outputs … ===")

    G.eval()

    # 1. 50-sample grid (assignment requirement)
    with torch.no_grad():
        z50    = torch.randn(50, cfg.latent_dim, device=device)
        imgs50 = G(z50)
    save_image_grid(imgs50, path=samp_dir / "samples_final_50.png", nrow=10)

    # 2. Latent interpolation (≥ 8 intermediate steps)
    latent_interpolation(
        G,
        device=device,
        latent_dim=cfg.latent_dim,
        n_steps=10,           # 10 intermediate → 12 total frames in one row
        save_path=samp_dir / "interpolation.png",
        seed=cfg.seed,
    )

    # 3. Training curves
    plot_training_curves(
        tracker,
        save_path=out / "training_curves.png",
        show=False,
    )

    # 4. Final loss JSON
    tracker.save(out / "losses.json")

    print(f"\nAll outputs saved to: {out.resolve()}")


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train WGAN-GP on CelebA")

    parser.add_argument("--data_root",    type=str,   default="./data")
    parser.add_argument("--max_samples",  type=int,   default=None,
                        help="Cap dataset size (useful on free Colab)")
    parser.add_argument("--num_epochs",   type=int,   default=20)
    parser.add_argument("--batch_size",   type=int,   default=64)
    parser.add_argument("--latent_dim",   type=int,   default=128)
    parser.add_argument("--feature_maps", type=int,   default=64)
    parser.add_argument("--n_critic",     type=int,   default=5)
    parser.add_argument("--lambda_gp",    type=float, default=10.0)
    parser.add_argument("--lr",           type=float, default=1e-4)
    parser.add_argument("--output_dir",   type=str,   default="./outputs")
    parser.add_argument("--num_workers",  type=int,   default=2)
    parser.add_argument("--seed",         type=int,   default=42)
    parser.add_argument("--resume",       type=str,   default=None,
                        help="Path to a .pt checkpoint to resume from")

    args = parser.parse_args()

    cfg = Config(
        data_root    = args.data_root,
        max_samples  = args.max_samples,
        num_epochs   = args.num_epochs,
        batch_size   = args.batch_size,
        latent_dim   = args.latent_dim,
        feature_maps = args.feature_maps,
        n_critic     = args.n_critic,
        lambda_gp    = args.lambda_gp,
        lr           = args.lr,
        output_dir   = args.output_dir,
        num_workers  = args.num_workers,
        seed         = args.seed,
    )

    train(cfg, resume_from=args.resume)
