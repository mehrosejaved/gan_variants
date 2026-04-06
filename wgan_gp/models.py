"""
models.py — Generator and Critic for WGAN-GP on CelebA 64×64.

Architecture: DCGAN-style conv/deconv stacks.

Design notes
------------
* Critic has NO BatchNorm (required by WGAN-GP; BN interacts badly with
  the per-sample gradient penalty).  InstanceNorm or no norm is used instead.
* Generator uses BatchNorm (standard; stable training signal).
* Both classes accept `latent_dim` / `feature_maps` so Task-2 conditioning
  can be bolted on by subclassing or small edits — nothing is hard-coded.
* `weights_init` follows the DCGAN paper (mean=0, std=0.02).
"""

import torch
import torch.nn as nn


# ---------------------------------------------------------------------------
# Weight initialisation (DCGAN convention)
# ---------------------------------------------------------------------------

def weights_init(module: nn.Module) -> None:
    """Apply to a model with `model.apply(weights_init)`."""
    classname = module.__class__.__name__
    if "Conv" in classname:
        nn.init.normal_(module.weight.data, 0.0, 0.02)
    elif "BatchNorm" in classname:
        nn.init.normal_(module.weight.data, 1.0, 0.02)
        nn.init.constant_(module.bias.data, 0.0)


# ---------------------------------------------------------------------------
# Generator  z → 3×64×64
# ---------------------------------------------------------------------------

class Generator(nn.Module):
    """
    Latent vector → RGB image in [-1, 1].

    Spatial progression: 1×1 (latent) → 4×4 → 8×8 → 16×16 → 32×32 → 64×64

    Parameters
    ----------
    latent_dim  : dimension of the input noise vector z
    feature_maps: base channel multiplier (ngf); doubles each block going in,
                  halves each block going out toward the image
    channels    : output image channels (3 for RGB)
    """

    def __init__(
        self,
        latent_dim: int = 128,
        feature_maps: int = 64,
        channels: int = 3,
    ) -> None:
        super().__init__()
        ngf = feature_maps

        self.latent_dim = latent_dim

        # Each block: ConvTranspose2d → BatchNorm → ReLU
        # Final block: ConvTranspose2d → Tanh  (no BN on output layer)
        self.net = nn.Sequential(
            # z is shaped (B, latent_dim, 1, 1) — projection to 4×4 feature map
            nn.ConvTranspose2d(latent_dim, ngf * 8, kernel_size=4, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(ngf * 8),
            nn.ReLU(inplace=True),
            # 4×4 → 8×8
            nn.ConvTranspose2d(ngf * 8, ngf * 4, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(ngf * 4),
            nn.ReLU(inplace=True),
            # 8×8 → 16×16
            nn.ConvTranspose2d(ngf * 4, ngf * 2, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(ngf * 2),
            nn.ReLU(inplace=True),
            # 16×16 → 32×32
            nn.ConvTranspose2d(ngf * 2, ngf, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(ngf),
            nn.ReLU(inplace=True),
            # 32×32 → 64×64
            nn.ConvTranspose2d(ngf, channels, kernel_size=4, stride=2, padding=1, bias=False),
            nn.Tanh(),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        z : (B, latent_dim) or (B, latent_dim, 1, 1)

        Returns
        -------
        images : (B, channels, 64, 64) in [-1, 1]
        """
        if z.dim() == 2:
            z = z.unsqueeze(-1).unsqueeze(-1)   # (B, latent_dim, 1, 1)
        return self.net(z)


# ---------------------------------------------------------------------------
# Critic  3×64×64 → scalar
# ---------------------------------------------------------------------------

class Critic(nn.Module):
    """
    Image → real-valued score (no sigmoid — Wasserstein critic, not classifier).

    Spatial progression: 64×64 → 32×32 → 16×16 → 8×8 → 4×4 → scalar

    NO BatchNorm anywhere in the critic.  InstanceNorm is used instead on
    intermediate layers to give some normalisation benefit without violating
    the WGAN-GP gradient-penalty assumptions.

    Parameters
    ----------
    feature_maps: base channel multiplier (ndf)
    channels    : input image channels (3 for RGB)
    """

    def __init__(
        self,
        feature_maps: int = 64,
        channels: int = 3,
    ) -> None:
        super().__init__()
        ndf = feature_maps

        # Each interior block: Conv2d → InstanceNorm → LeakyReLU
        # First block: no norm (input is raw image pixels)
        # Last  block: Conv2d only → scalar
        self.net = nn.Sequential(
            # 64×64 → 32×32  (no norm on first layer)
            nn.Conv2d(channels, ndf, kernel_size=4, stride=2, padding=1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # 32×32 → 16×16
            nn.Conv2d(ndf, ndf * 2, kernel_size=4, stride=2, padding=1, bias=False),
            nn.InstanceNorm2d(ndf * 2, affine=True),
            nn.LeakyReLU(0.2, inplace=True),
            # 16×16 → 8×8
            nn.Conv2d(ndf * 2, ndf * 4, kernel_size=4, stride=2, padding=1, bias=False),
            nn.InstanceNorm2d(ndf * 4, affine=True),
            nn.LeakyReLU(0.2, inplace=True),
            # 8×8 → 4×4
            nn.Conv2d(ndf * 4, ndf * 8, kernel_size=4, stride=2, padding=1, bias=False),
            nn.InstanceNorm2d(ndf * 8, affine=True),
            nn.LeakyReLU(0.2, inplace=True),
            # 4×4 → 1×1 scalar
            nn.Conv2d(ndf * 8, 1, kernel_size=4, stride=1, padding=0, bias=False),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x : (B, channels, 64, 64)

        Returns
        -------
        score : (B,)  — unbounded real value
        """
        out = self.net(x)           # (B, 1, 1, 1)
        return out.view(out.size(0))  # (B,)


# ---------------------------------------------------------------------------
# Quick sanity check
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    BATCH      = 4
    LATENT_DIM = 128
    FEAT_G     = 64
    FEAT_D     = 64

    G = Generator(latent_dim=LATENT_DIM, feature_maps=FEAT_G).to(device)
    D = Critic(feature_maps=FEAT_D).to(device)

    G.apply(weights_init)
    D.apply(weights_init)

    z     = torch.randn(BATCH, LATENT_DIM, device=device)
    fakes = G(z)
    score = D(fakes)

    print(f"Generator   params : {sum(p.numel() for p in G.parameters()):,}")
    print(f"Critic      params : {sum(p.numel() for p in D.parameters()):,}")
    print(f"G output shape     : {fakes.shape}")   # (4, 3, 64, 64)
    print(f"D output shape     : {score.shape}")   # (4,)
    print(f"Score range        : [{score.min().item():.3f}, {score.max().item():.3f}]")
