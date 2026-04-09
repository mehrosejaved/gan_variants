"""
cond_train.py — Conditional WGAN-GP training loop for CelebA 64×64.

Attribute : Eyeglasses  (binary, assigned to names starting with R-Z)

Imports dataset.py and utils.py from ../wgan_gp/ so nothing is duplicated.

Outputs (written to Config.output_dir)
---------------------------------------
  checkpoints/   ckpt_epoch{N}.pt
  samples/       grid_{step:06d}.png          (periodic, both classes)
  samples/       samples_final_50.png         (25 with + 25 without glasses)
  samples/       fixed_z_comparison.png       (same z, label 0 vs label 1)
  samples/       interpolation_cond.png       (z1→z2, label fixed)
  training_curves.png
  losses.json

Run
---
    python cond_train.py --data_root ./data --attr_name Eyeglasses
"""

from __future__ import annotations

import argparse
import os
import random
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn

# Reuse dataset and utils from Task 1 — no duplication
_HERE   = Path(__file__).resolve().parent
_TASK1  = _HERE.parent / "wgan_gp"
sys.path.insert(0, str(_TASK1))

from dataset import get_celeba_loader           # noqa: E402
from utils   import (                           # noqa: E402
    gradient_penalty,
    LossTracker,
    plot_training_curves,
    save_image_grid,
)
from cond_models import (                       # noqa: E402
    ConditionalGenerator,
    ConditionalCritic,
    weights_init,
)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class CondConfig:
    # Data
    data_root:   str            = "./data"
    attr_name:   str            = "Eyeglasses"   # assigned attribute
    max_samples: Optional[int]  = None

    # Architecture
    latent_dim:   int = 128
    embed_dim:    int = 32     # label embedding size
    num_classes:  int = 2      # binary attribute → 2 classes
    feature_maps: int = 64

    # Training  (same WGAN-GP hyper-parameters as Task 1)
    num_epochs:   int   = 20
    batch_size:   int   = 64
    n_critic:     int   = 5
    lambda_gp:    float = 10.0
    lr:           float = 1e-4
    adam_b1:      float = 0.0
    adam_b2:      float = 0.9

    # Logging
    output_dir:       str = "./outputs_cond"
    sample_interval:  int = 200
    ckpt_interval:    int = 5
    num_workers:      int = 2
    seed:             int = 42


# ---------------------------------------------------------------------------
# Gradient penalty (conditional version — passes labels to critic)
# ---------------------------------------------------------------------------

def cond_gradient_penalty(
    critic:    nn.Module,
    real:      torch.Tensor,
    fake:      torch.Tensor,
    labels:    torch.Tensor,
    device:    torch.device,
    lambda_gp: float = 10.0,
) -> torch.Tensor:
    """
    Same as utils.gradient_penalty but also passes `labels` to the critic.
    """
    B, C, H, W = real.shape
    alpha  = torch.rand(B, 1, 1, 1, device=device).expand_as(real)
    x_hat  = (alpha * real + (1.0 - alpha) * fake).requires_grad_(True)
    d_hat  = critic(x_hat, labels)

    gradients = torch.autograd.grad(
        outputs=d_hat,
        inputs=x_hat,
        grad_outputs=torch.ones_like(d_hat),
        create_graph=True,
        retain_graph=True,
        only_inputs=True,
    )[0]

    grad_norm = gradients.view(B, -1).norm(2, dim=1)
    return lambda_gp * ((grad_norm - 1.0) ** 2).mean()


# ---------------------------------------------------------------------------
# Post-training visualisations
# ---------------------------------------------------------------------------

@torch.no_grad()
def save_fixed_z_comparison(
    G:          nn.Module,
    device:     torch.device,
    latent_dim: int,
    n_samples:  int = 10,
    save_path:  str | Path = "fixed_z_comparison.png",
    seed:       int = 0,
) -> None:
    """
    Generate n_samples images with the SAME latent vectors but two different
    label values (0 = no glasses, 1 = glasses).

    Layout: top row = label 0, bottom row = label 1, columns share z.
    """
    G.eval()
    torch.manual_seed(seed)
    z = torch.randn(n_samples, latent_dim, device=device)

    labels_0 = torch.zeros(n_samples, dtype=torch.long, device=device)
    labels_1 = torch.ones(n_samples,  dtype=torch.long, device=device)

    row0 = G(z, labels_0)   # (n, 3, 64, 64) — no glasses
    row1 = G(z, labels_1)   # (n, 3, 64, 64) — glasses

    # Interleave so each pair (no-glass, glass) is side by side for easy comparison
    # Stack: [img0_label0, img0_label1, img1_label0, img1_label1, ...]
    interleaved = torch.stack([row0, row1], dim=1).view(n_samples * 2, 3, 64, 64)

    save_image_grid(interleaved, path=save_path, nrow=n_samples)
    print(f"  [fixed-z] saved → {save_path}")
    print(f"  Top row    = label 0 (no Eyeglasses)")
    print(f"  Bottom row = label 1 (Eyeglasses)")


@torch.no_grad()
def cond_latent_interpolation(
    G:          nn.Module,
    device:     torch.device,
    latent_dim: int,
    label:      int = 1,       # keep condition fixed at this value
    n_steps:    int = 10,
    save_path:  str | Path = "interpolation_cond.png",
    seed:       int = 0,
) -> None:
    """
    Linearly interpolate z1 → z2 while keeping the condition label fixed.
    Generates n_steps + 2 frames (endpoints included) in one row.
    """
    G.eval()
    gen = torch.Generator(device=device)
    gen.manual_seed(seed)

    z1 = torch.randn(1, latent_dim, device=device, generator=gen)
    z2 = torch.randn(1, latent_dim, device=device, generator=gen)

    alphas = torch.linspace(0.0, 1.0, n_steps + 2, device=device)
    frames = []
    fixed_label = torch.tensor([label], device=device)

    for a in alphas:
        z   = (1.0 - a) * z1 + a * z2
        img = G(z, fixed_label)
        frames.append(img)

    grid = torch.cat(frames, dim=0)
    save_image_grid(grid, path=save_path, nrow=len(alphas))
    print(f"  [interp] label={label} fixed, {len(alphas)} frames → {save_path}")


# ---------------------------------------------------------------------------
# Helpers (mirrors train.py)
# ---------------------------------------------------------------------------

def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def save_checkpoint(G, D, opt_G, opt_D, epoch, out_dir):
    ckpt_dir = Path(out_dir) / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    torch.save(
        {"epoch": epoch, "G": G.state_dict(), "D": D.state_dict(),
         "opt_G": opt_G.state_dict(), "opt_D": opt_D.state_dict()},
        ckpt_dir / f"ckpt_epoch{epoch:03d}.pt",
    )
    print(f"  [ckpt] saved epoch {epoch}")


def load_checkpoint(path, G, D, opt_G, opt_D, device):
    ckpt = torch.load(path, map_location=device)
    G.load_state_dict(ckpt["G"])
    D.load_state_dict(ckpt["D"])
    opt_G.load_state_dict(ckpt["opt_G"])
    opt_D.load_state_dict(ckpt["opt_D"])
    print(f"  [ckpt] resumed from epoch {ckpt['epoch']}")
    return ckpt["epoch"]


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

def train(cfg: CondConfig, resume_from: Optional[str] = None) -> None:
    set_seed(cfg.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device     : {device}")
    print(f"Attribute  : {cfg.attr_name}")
    print(f"Epochs     : {cfg.num_epochs}  |  Batch : {cfg.batch_size}  |  n_critic : {cfg.n_critic}")

    out      = Path(cfg.output_dir)
    samp_dir = out / "samples"
    samp_dir.mkdir(parents=True, exist_ok=True)

    # Data — returns (images, labels) batches
    loader = get_celeba_loader(
        root=cfg.data_root,
        split="train",
        batch_size=cfg.batch_size,
        num_workers=cfg.num_workers,
        attr_name=cfg.attr_name,
        max_samples=cfg.max_samples,
    )
    print(f"Dataset    : {len(loader.dataset):,} images  →  {len(loader):,} batches/epoch")

    # Models
    G = ConditionalGenerator(
        latent_dim=cfg.latent_dim, embed_dim=cfg.embed_dim,
        num_classes=cfg.num_classes, feature_maps=cfg.feature_maps,
    ).to(device)
    D = ConditionalCritic(
        embed_dim=cfg.embed_dim, num_classes=cfg.num_classes,
        feature_maps=cfg.feature_maps,
    ).to(device)
    G.apply(weights_init)
    D.apply(weights_init)
    print(f"G params   : {sum(p.numel() for p in G.parameters()):,}")
    print(f"D params   : {sum(p.numel() for p in D.parameters()):,}")

    opt_G = torch.optim.Adam(G.parameters(), lr=cfg.lr, betas=(cfg.adam_b1, cfg.adam_b2))
    opt_D = torch.optim.Adam(D.parameters(), lr=cfg.lr, betas=(cfg.adam_b1, cfg.adam_b2))

    start_epoch = 0
    if resume_from:
        start_epoch = load_checkpoint(resume_from, G, D, opt_G, opt_D, device)

    # Fixed noise + fixed labels for consistent periodic grids:
    # first 32 = label 0 (no glasses), last 32 = label 1 (glasses)
    fixed_z      = torch.randn(64, cfg.latent_dim, device=device)
    fixed_labels = torch.cat([
        torch.zeros(32, dtype=torch.long),
        torch.ones(32,  dtype=torch.long),
    ]).to(device)

    tracker = LossTracker()
    g_step = 0
    d_step = 0

    for epoch in range(start_epoch, cfg.num_epochs):
        G.train(); D.train()

        for real, labels in loader:
            real   = real.to(device)
            labels = labels.to(device)
            B      = real.size(0)

            # ── Critic update ──────────────────────────────────────────────
            opt_D.zero_grad()

            z    = torch.randn(B, cfg.latent_dim, device=device)
            fake = G(z, labels).detach()

            d_real = D(real, labels).mean()
            d_fake = D(fake, labels).mean()
            gp     = cond_gradient_penalty(D, real, fake, labels, device, cfg.lambda_gp)
            d_loss = -d_real + d_fake + gp

            d_loss.backward()
            opt_D.step()
            d_step += 1

            # ── Generator update ───────────────────────────────────────────
            if d_step % cfg.n_critic == 0:
                opt_G.zero_grad()

                z    = torch.randn(B, cfg.latent_dim, device=device)
                fake = G(z, labels)
                g_loss = -D(fake, labels).mean()

                g_loss.backward()
                opt_G.step()

                raw_gp = gp.item() / cfg.lambda_gp
                tracker.record(
                    g_loss=g_loss.item(),
                    d_loss=d_loss.item(),
                    gp=raw_gp,
                )
                g_step += 1

                if g_step % cfg.sample_interval == 0:
                    G.eval()
                    with torch.no_grad():
                        samples = G(fixed_z, fixed_labels)
                    save_image_grid(samples, path=samp_dir / f"grid_{g_step:06d}.png", nrow=8)
                    G.train()

                if g_step % 50 == 0:
                    print(
                        f"Epoch [{epoch+1:03d}/{cfg.num_epochs}]  "
                        f"Step {g_step:5d}  |  "
                        f"D: {d_loss.item():+.4f}  "
                        f"G: {g_loss.item():+.4f}  "
                        f"GP: {raw_gp:.4f}"
                    )

        if (epoch + 1) % cfg.ckpt_interval == 0:
            save_checkpoint(G, D, opt_G, opt_D, epoch + 1, out)
        tracker.save(out / "losses.json")

    # ── Post-training outputs ──────────────────────────────────────────────
    print("\n=== Training complete. Generating final outputs … ===")
    G.eval()

    # 1. 50-sample grid — 25 without glasses, 25 with glasses
    with torch.no_grad():
        z50      = torch.randn(50, cfg.latent_dim, device=device)
        labels50 = torch.cat([
            torch.zeros(25, dtype=torch.long),
            torch.ones(25,  dtype=torch.long),
        ]).to(device)
        imgs50 = G(z50, labels50)
    save_image_grid(imgs50, path=samp_dir / "samples_final_50.png", nrow=10)

    # 2. Fixed-z comparison: same z, label 0 vs label 1
    save_fixed_z_comparison(
        G, device=device, latent_dim=cfg.latent_dim,
        n_samples=10, save_path=samp_dir / "fixed_z_comparison.png",
        seed=cfg.seed,
    )

    # 3. Latent interpolation with fixed label=1 (with glasses)
    cond_latent_interpolation(
        G, device=device, latent_dim=cfg.latent_dim,
        label=1, n_steps=10,
        save_path=samp_dir / "interpolation_cond.png",
        seed=cfg.seed,
    )

    # 4. Training curves + loss JSON
    plot_training_curves(tracker, save_path=out / "training_curves.png", show=False)
    tracker.save(out / "losses.json")
    print(f"\nAll outputs saved to: {out.resolve()}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Conditional WGAN-GP on CelebA")
    parser.add_argument("--data_root",    type=str,   default="./data")
    parser.add_argument("--attr_name",    type=str,   default="Eyeglasses")
    parser.add_argument("--max_samples",  type=int,   default=None)
    parser.add_argument("--num_epochs",   type=int,   default=20)
    parser.add_argument("--batch_size",   type=int,   default=64)
    parser.add_argument("--latent_dim",   type=int,   default=128)
    parser.add_argument("--embed_dim",    type=int,   default=32)
    parser.add_argument("--feature_maps", type=int,   default=64)
    parser.add_argument("--n_critic",     type=int,   default=5)
    parser.add_argument("--lambda_gp",    type=float, default=10.0)
    parser.add_argument("--lr",           type=float, default=1e-4)
    parser.add_argument("--output_dir",   type=str,   default="./outputs_cond")
    parser.add_argument("--num_workers",  type=int,   default=2)
    parser.add_argument("--seed",         type=int,   default=42)
    parser.add_argument("--resume",       type=str,   default=None)

    args = parser.parse_args()
    cfg  = CondConfig(
        data_root=args.data_root, attr_name=args.attr_name,
        max_samples=args.max_samples, num_epochs=args.num_epochs,
        batch_size=args.batch_size, latent_dim=args.latent_dim,
        embed_dim=args.embed_dim, feature_maps=args.feature_maps,
        n_critic=args.n_critic, lambda_gp=args.lambda_gp, lr=args.lr,
        output_dir=args.output_dir, num_workers=args.num_workers,
        seed=args.seed,
    )
    train(cfg, resume_from=args.resume)
