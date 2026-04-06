"""
utils.py — Gradient penalty, loss tracking, and training-curve plotting
         for WGAN-GP on CelebA.

Public API
----------
gradient_penalty(critic, real, fake, device, lambda_gp)
    → GP scalar tensor (differentiable)

LossTracker()
    .record(g_loss, d_loss, gp)        — call once per generator update step
    .to_dict()                         — returns raw history lists
    .save(path)  /  LossTracker.load(path)

plot_training_curves(tracker, save_path, show)
    → saves a 3-panel PNG: G loss | D loss | Gradient Penalty

denormalize(tensor)
    → maps [-1,1] back to [0,1] for display / saving

save_image_grid(tensor, path, nrow)
    → saves a grid of generated samples using torchvision
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Optional

import matplotlib
matplotlib.use("Agg")          # non-interactive backend — safe on Colab & headless
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torchvision.utils as vutils


# ---------------------------------------------------------------------------
# Gradient penalty (the core WGAN-GP addition)
# ---------------------------------------------------------------------------

def gradient_penalty(
    critic: nn.Module,
    real: torch.Tensor,
    fake: torch.Tensor,
    device: torch.device,
    lambda_gp: float = 10.0,
) -> torch.Tensor:
    """
    Compute the WGAN-GP gradient penalty.

    For each sample in the batch we:
      1. Draw a random interpolation weight α ~ Uniform(0,1).
      2. Form  x̂ = α·real + (1−α)·fake   (the "interpolated" image).
      3. Compute ‖∇_{x̂} D(x̂)‖₂  per sample.
      4. Penalise  (‖gradient‖₂ − 1)²   (enforces 1-Lipschitz).

    Parameters
    ----------
    critic    : the Critic network (not in eval mode — gradients must flow)
    real      : (B, C, H, W) real images from the dataset
    fake      : (B, C, H, W) generated images (detached from G graph is fine;
                autograd on x̂ provides the needed gradient)
    device    : torch device
    lambda_gp : penalty coefficient (10 is the WGAN-GP paper default)

    Returns
    -------
    gp_loss : scalar tensor, already multiplied by lambda_gp
    """
    B, C, H, W = real.shape

    # Random interpolation weights — one per image in the batch
    alpha = torch.rand(B, 1, 1, 1, device=device)          # (B,1,1,1)
    alpha = alpha.expand_as(real)

    # Interpolated images
    x_hat = (alpha * real + (1.0 - alpha) * fake).requires_grad_(True)

    # Critic score at interpolated points
    d_hat = critic(x_hat)                                   # (B,)

    # Compute ∂D(x̂)/∂x̂  for every element of d_hat simultaneously
    gradients = torch.autograd.grad(
        outputs=d_hat,
        inputs=x_hat,
        grad_outputs=torch.ones_like(d_hat),
        create_graph=True,   # needed so the GP is part of the computational graph
        retain_graph=True,
        only_inputs=True,
    )[0]                                                     # (B, C, H, W)

    # Flatten spatial + channel dims, compute per-sample L2 norm
    gradients = gradients.view(B, -1)                       # (B, C*H*W)
    grad_norm  = gradients.norm(2, dim=1)                   # (B,)

    # Penalty: mean over batch, then scale by lambda
    gp = lambda_gp * ((grad_norm - 1.0) ** 2).mean()
    return gp


# ---------------------------------------------------------------------------
# Loss tracker — records one entry per generator update step
# ---------------------------------------------------------------------------

class LossTracker:
    """
    Accumulates scalar loss values during training and provides
    smoothed plotting via exponential moving average (EMA).

    Usage in training loop
    ----------------------
        tracker = LossTracker()
        ...
        tracker.record(g_loss=g.item(), d_loss=d.item(), gp=gp.item())
        ...
        plot_training_curves(tracker, save_path="curves.png")
        tracker.save("losses.json")
    """

    def __init__(self) -> None:
        self.g_losses:  list[float] = []   # generator loss (one value per G step)
        self.d_losses:  list[float] = []   # critic Wasserstein loss (per G step)
        self.gp_values: list[float] = []   # raw GP value (before lambda scaling)
        self.steps:     list[int]   = []   # global step index

        self._step = 0

    def record(
        self,
        g_loss: float,
        d_loss: float,
        gp: float,
    ) -> None:
        """
        Call once per generator update (i.e. once every n_critic critic steps).

        Parameters
        ----------
        g_loss  : generator loss scalar  (−mean D(fake))
        d_loss  : critic Wasserstein loss (mean D(fake) − mean D(real) + lambda·GP)
        gp      : gradient penalty value *before* lambda scaling
                  (store raw GP so the curve shows its own scale)
        """
        self.g_losses.append(float(g_loss))
        self.d_losses.append(float(d_loss))
        self.gp_values.append(float(gp))
        self.steps.append(self._step)
        self._step += 1

    def __len__(self) -> int:
        return len(self.steps)

    def to_dict(self) -> dict:
        return {
            "steps":     self.steps,
            "g_losses":  self.g_losses,
            "d_losses":  self.d_losses,
            "gp_values": self.gp_values,
        }

    def save(self, path: str | Path) -> None:
        """Persist history to a JSON file so training can be resumed / re-plotted."""
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)
        print(f"[LossTracker] saved → {path}")

    @classmethod
    def load(cls, path: str | Path) -> "LossTracker":
        """Reload a previously saved tracker."""
        with open(path) as f:
            data = json.load(f)
        tracker = cls()
        tracker.steps     = data["steps"]
        tracker.g_losses  = data["g_losses"]
        tracker.d_losses  = data["d_losses"]
        tracker.gp_values = data["gp_values"]
        tracker._step     = (max(data["steps"]) + 1) if data["steps"] else 0
        return tracker


# ---------------------------------------------------------------------------
# Exponential moving average smoother (for cleaner plots)
# ---------------------------------------------------------------------------

def _ema(values: list[float], alpha: float = 0.05) -> np.ndarray:
    """
    Exponential moving average with weight `alpha` for the new value.
    Lower alpha = smoother curve.  0.05 works well for ~thousands of steps.
    """
    arr    = np.array(values, dtype=np.float32)
    smooth = np.empty_like(arr)
    smooth[0] = arr[0]
    for i in range(1, len(arr)):
        smooth[i] = alpha * arr[i] + (1.0 - alpha) * smooth[i - 1]
    return smooth


# ---------------------------------------------------------------------------
# Training-curve plot  (3 panels)
# ---------------------------------------------------------------------------

def plot_training_curves(
    tracker: LossTracker,
    save_path: str | Path = "training_curves.png",
    show: bool = False,
    ema_alpha: float = 0.05,
) -> None:
    """
    Plot and save a 3-panel figure:
        Left   — Generator loss
        Centre — Critic (Wasserstein) loss
        Right  — Gradient penalty (raw, before lambda scaling)

    Raw values are shown faintly; EMA-smoothed line is drawn on top.

    Parameters
    ----------
    tracker   : populated LossTracker instance
    save_path : where to write the PNG (parent dirs are created if needed)
    show      : call plt.show() — set True in interactive notebooks
    ema_alpha : smoothing strength (smaller = smoother)
    """
    if len(tracker) == 0:
        print("[plot_training_curves] tracker is empty — nothing to plot.")
        return

    steps = tracker.steps
    data  = [
        (tracker.g_losses,  "Generator Loss",              "#4C72B0"),
        (tracker.d_losses,  "Critic Loss (Wasserstein)",   "#DD8452"),
        (tracker.gp_values, "Gradient Penalty (raw)",      "#55A868"),
    ]

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    fig.suptitle("WGAN-GP Training Curves", fontsize=14, fontweight="bold")

    for ax, (values, title, color) in zip(axes, data):
        arr    = np.array(values)
        smooth = _ema(values, alpha=ema_alpha)

        # Raw values — thin, semi-transparent
        ax.plot(steps, arr, color=color, alpha=0.20, linewidth=0.8)
        # EMA-smoothed — solid
        ax.plot(steps, smooth, color=color, linewidth=1.8, label="EMA")

        ax.set_title(title, fontsize=11)
        ax.set_xlabel("Generator step", fontsize=9)
        ax.set_ylabel("Loss value",     fontsize=9)
        ax.grid(True, linestyle="--", alpha=0.4)
        ax.legend(fontsize=8)

    plt.tight_layout()

    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"[plot_training_curves] saved → {save_path}")

    if show:
        plt.show()
    plt.close(fig)


# ---------------------------------------------------------------------------
# Image helpers
# ---------------------------------------------------------------------------

def denormalize(tensor: torch.Tensor) -> torch.Tensor:
    """
    Reverse the [-1, 1] normalisation applied in dataset.py.
    Maps tensor back to [0, 1] for display or torchvision saving.

    Works on (C,H,W) or (B,C,H,W) tensors.
    """
    return (tensor * 0.5 + 0.5).clamp(0.0, 1.0)


def save_image_grid(
    tensor: torch.Tensor,
    path: str | Path,
    nrow: int = 10,
    normalize: bool = True,
) -> None:
    """
    Save a batch of images as a grid PNG.

    Parameters
    ----------
    tensor    : (B, C, H, W) in [-1, 1]
    path      : output file path (.png)
    nrow      : images per row in the grid
    normalize : if True, applies denormalize() before saving
    """
    if normalize:
        tensor = denormalize(tensor)
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    vutils.save_image(tensor, path, nrow=nrow, padding=2)
    print(f"[save_image_grid] saved → {path}")


# ---------------------------------------------------------------------------
# Sanity check
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys
    sys.path.insert(0, str(Path(__file__).parent))
    from models import Generator, Critic, weights_init

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # --- Gradient penalty check ---
    G = Generator(latent_dim=128).to(device)
    D = Critic().to(device)
    G.apply(weights_init); D.apply(weights_init)

    z    = torch.randn(8, 128, device=device)
    real = torch.randn(8, 3, 64, 64, device=device)   # stand-in for real images
    fake = G(z).detach()

    gp = gradient_penalty(D, real, fake, device, lambda_gp=10.0)
    print(f"Gradient penalty : {gp.item():.4f}  (should be a small positive number)")

    # --- LossTracker + plot check ---
    tracker = LossTracker()
    rng = np.random.default_rng(0)
    for i in range(200):
        tracker.record(
            g_loss  = float(-rng.normal(1.0, 0.5)),
            d_loss  = float(rng.normal(0.5, 0.3)),
            gp      = float(abs(rng.normal(0.1, 0.05))),
        )

    tracker.save("losses_test.json")
    loaded = LossTracker.load("losses_test.json")
    assert len(loaded) == 200, "Load/save mismatch"

    plot_training_curves(tracker, save_path="training_curves_test.png", show=False)
    print("All checks passed.")
