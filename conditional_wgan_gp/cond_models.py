"""
cond_models.py — Conditional Generator and Critic for WGAN-GP on CelebA 64×64.

Conditioning strategy
---------------------
Generator
  Label  →  Embedding(num_classes, embed_dim)
  The embedding is concatenated with the noise vector z before the first
  ConvTranspose layer.  Input size becomes (latent_dim + embed_dim, 1, 1).
  This is the simplest effective approach: the label rides alongside z
  through the entire upsampling stack.

Critic
  Label  →  Embedding(num_classes, embed_dim)
            Linear(embed_dim, H*W)
            Reshape to (1, H, W)
  The spatial label map is concatenated with the image as a 4th channel,
  so the critic sees (B, 4, 64, 64) instead of (B, 3, 64, 64).
  This lets the critic directly compare image content against the label
  at every spatial location.

Both classes inherit the WGAN-GP design rules from models.py:
  - No BatchNorm in the critic (replaced by InstanceNorm)
  - No Sigmoid output on the critic

Usage
-----
    from cond_models import ConditionalGenerator, ConditionalCritic, weights_init

    G = ConditionalGenerator(latent_dim=128, embed_dim=32, num_classes=2)
    D = ConditionalCritic(embed_dim=32, num_classes=2)

    z      = torch.randn(B, 128)
    labels = torch.randint(0, 2, (B,))

    imgs   = G(z, labels)      # (B, 3, 64, 64)
    scores = D(imgs, labels)   # (B,)
"""

from __future__ import annotations

import torch
import torch.nn as nn


# ---------------------------------------------------------------------------
# Weight initialisation (same DCGAN convention as models.py)
# ---------------------------------------------------------------------------

def weights_init(module: nn.Module) -> None:
    classname = module.__class__.__name__
    if "Conv" in classname:
        nn.init.normal_(module.weight.data, 0.0, 0.02)
    elif "BatchNorm" in classname:
        nn.init.normal_(module.weight.data, 1.0, 0.02)
        nn.init.constant_(module.bias.data, 0.0)


# ---------------------------------------------------------------------------
# Conditional Generator   (z, label) → 3×64×64
# ---------------------------------------------------------------------------

class ConditionalGenerator(nn.Module):
    """
    Parameters
    ----------
    latent_dim  : dimension of the noise vector z
    embed_dim   : label embedding dimension (concatenated with z)
    num_classes : number of distinct labels (2 for a binary attribute)
    feature_maps: base channel multiplier (ngf)
    channels    : output image channels (3 for RGB)
    """

    def __init__(
        self,
        latent_dim:   int = 128,
        embed_dim:    int = 32,
        num_classes:  int = 2,
        feature_maps: int = 64,
        channels:     int = 3,
    ) -> None:
        super().__init__()
        ngf = feature_maps
        self.latent_dim  = latent_dim
        self.embed_dim   = embed_dim
        input_dim = latent_dim + embed_dim   # z ∥ embed(label)

        self.label_embed = nn.Embedding(num_classes, embed_dim)

        # Identical to unconditional Generator except first layer
        # accepts (latent_dim + embed_dim) channels instead of latent_dim
        self.net = nn.Sequential(
            # (input_dim, 1, 1) → (ngf*8, 4, 4)
            nn.ConvTranspose2d(input_dim, ngf * 8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(ngf * 8),
            nn.ReLU(inplace=True),
            # 4×4 → 8×8
            nn.ConvTranspose2d(ngf * 8, ngf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 4),
            nn.ReLU(inplace=True),
            # 8×8 → 16×16
            nn.ConvTranspose2d(ngf * 4, ngf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 2),
            nn.ReLU(inplace=True),
            # 16×16 → 32×32
            nn.ConvTranspose2d(ngf * 2, ngf, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf),
            nn.ReLU(inplace=True),
            # 32×32 → 64×64
            nn.ConvTranspose2d(ngf, channels, 4, 2, 1, bias=False),
            nn.Tanh(),
        )

    def forward(self, z: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        z      : (B, latent_dim)
        labels : (B,) integer class indices

        Returns
        -------
        images : (B, channels, 64, 64) in [-1, 1]
        """
        emb   = self.label_embed(labels)          # (B, embed_dim)
        x     = torch.cat([z, emb], dim=1)        # (B, latent_dim + embed_dim)
        x     = x.unsqueeze(-1).unsqueeze(-1)      # (B, latent_dim+embed_dim, 1, 1)
        return self.net(x)


# ---------------------------------------------------------------------------
# Conditional Critic   (image, label) → scalar
# ---------------------------------------------------------------------------

class ConditionalCritic(nn.Module):
    """
    Parameters
    ----------
    embed_dim   : label embedding dimension projected to a spatial map
    num_classes : number of distinct labels
    feature_maps: base channel multiplier (ndf)
    channels    : input image channels (3 for RGB)
    image_size  : spatial resolution of input images (64)
    """

    def __init__(
        self,
        embed_dim:    int = 32,
        num_classes:  int = 2,
        feature_maps: int = 64,
        channels:     int = 3,
        image_size:   int = 64,
    ) -> None:
        super().__init__()
        ndf = feature_maps
        self.image_size = image_size

        # Project label embedding to a (1, H, W) spatial map
        self.label_embed = nn.Embedding(num_classes, embed_dim)
        self.label_proj  = nn.Linear(embed_dim, image_size * image_size)

        # Critic input is (channels + 1) because the label map is appended
        in_channels = channels + 1

        self.net = nn.Sequential(
            # (in_channels, 64, 64) → (ndf, 32, 32)  — no norm on first layer
            nn.Conv2d(in_channels, ndf, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # 32×32 → 16×16
            nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False),
            nn.InstanceNorm2d(ndf * 2, affine=True),
            nn.LeakyReLU(0.2, inplace=True),
            # 16×16 → 8×8
            nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False),
            nn.InstanceNorm2d(ndf * 4, affine=True),
            nn.LeakyReLU(0.2, inplace=True),
            # 8×8 → 4×4
            nn.Conv2d(ndf * 4, ndf * 8, 4, 2, 1, bias=False),
            nn.InstanceNorm2d(ndf * 8, affine=True),
            nn.LeakyReLU(0.2, inplace=True),
            # 4×4 → scalar
            nn.Conv2d(ndf * 8, 1, 4, 1, 0, bias=False),
        )

    def forward(self, x: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x      : (B, channels, H, W) image tensor
        labels : (B,) integer class indices

        Returns
        -------
        score : (B,) unbounded real value
        """
        B, C, H, W = x.shape
        emb        = self.label_embed(labels)              # (B, embed_dim)
        label_map  = self.label_proj(emb)                  # (B, H*W)
        label_map  = label_map.view(B, 1, H, W)            # (B, 1, H, W)
        x_cond     = torch.cat([x, label_map], dim=1)      # (B, C+1, H, W)
        out        = self.net(x_cond)                      # (B, 1, 1, 1)
        return out.view(B)


# ---------------------------------------------------------------------------
# Sanity check
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    B = 4

    G = ConditionalGenerator(latent_dim=128, embed_dim=32, num_classes=2).to(device)
    D = ConditionalCritic(embed_dim=32, num_classes=2).to(device)
    G.apply(weights_init)
    D.apply(weights_init)

    z      = torch.randn(B, 128, device=device)
    labels = torch.randint(0, 2, (B,), device=device)

    imgs   = G(z, labels)
    scores = D(imgs, labels)

    print(f"G output : {imgs.shape}   range [{imgs.min():.2f}, {imgs.max():.2f}]")
    print(f"D output : {scores.shape}")
    print(f"G params : {sum(p.numel() for p in G.parameters()):,}")
    print(f"D params : {sum(p.numel() for p in D.parameters()):,}")
