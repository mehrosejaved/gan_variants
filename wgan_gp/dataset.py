"""
dataset.py — CelebA data loading for WGAN-GP (Task 1) and Conditional WGAN-GP (Task 2).

CelebA image dimensions are 178 × 218 (W × H).
Pipeline: CenterCrop(178) → square face crop → Resize(64) → Tensor → Normalize[-1,1].

Usage
-----
# Task 1 — images only
loader = get_celeba_loader(root="./data", batch_size=64)

# Task 2 — images + one binary attribute (e.g. "Smiling")
loader = get_celeba_loader(root="./data", batch_size=64, attr_name="Smiling")

Colab note
----------
torchvision's built-in CelebA download relies on Google Drive and often hits
quota limits.  If that happens, set `download=False` and point `root` at a
directory that already contains:
    root/celeba/img_align_celeba/*.jpg
    root/celeba/list_attr_celeba.txt
    root/celeba/list_eval_partition.txt
You can get these from Kaggle:
    https://www.kaggle.com/datasets/jessicali9530/celeba-dataset
"""

from __future__ import annotations

import os
from typing import Optional, Tuple

import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.datasets import CelebA


# ---------------------------------------------------------------------------
# All 40 CelebA attribute names (index order matches list_attr_celeba.txt)
# Useful for Task 2 attribute look-up.
# ---------------------------------------------------------------------------
CELEBA_ATTRS = [
    "5_o_Clock_Shadow", "Arched_Eyebrows", "Attractive", "Bags_Under_Eyes",
    "Bald", "Bangs", "Big_Lips", "Big_Nose", "Black_Hair", "Blond_Hair",
    "Blurry", "Brown_Hair", "Bushy_Eyebrows", "Chubby", "Double_Chin",
    "Eyeglasses", "Goatee", "Gray_Hair", "Heavy_Makeup", "High_Cheekbones",
    "Male", "Mouth_Slightly_Open", "Mustache", "Narrow_Eyes", "No_Beard",
    "Oval_Face", "Pale_Skin", "Pointy_Nose", "Receding_Hairline",
    "Rosy_Cheeks", "Sideburns", "Smiling", "Straight_Hair", "Wavy_Hair",
    "Wearing_Earrings", "Wearing_Hat", "Wearing_Lipstick",
    "Wearing_Necklace", "Wearing_Necktie", "Young",
]


# ---------------------------------------------------------------------------
# Thin wrapper that optionally exposes a single binary attribute label
# ---------------------------------------------------------------------------

class CelebAWrapper(Dataset):
    """
    Wraps torchvision CelebA.

    Returns
    -------
    attr_name is None  →  image tensor only
    attr_name given    →  (image tensor, label) where label ∈ {0, 1}
    """

    def __init__(
        self,
        root: str,
        split: str = "train",
        transform=None,
        attr_name: Optional[str] = None,
        download: bool = False,
    ) -> None:
        self.dataset = CelebA(
            root=root,
            split=split,
            target_type="attr",
            transform=transform,
            download=download,
        )

        self.attr_idx: Optional[int] = None
        if attr_name is not None:
            if attr_name not in CELEBA_ATTRS:
                raise ValueError(
                    f"Unknown attribute '{attr_name}'. "
                    f"Valid options: {CELEBA_ATTRS}"
                )
            self.attr_idx = CELEBA_ATTRS.index(attr_name)

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, idx: int):
        image, attrs = self.dataset[idx]
        if self.attr_idx is None:
            return image
        label = attrs[self.attr_idx].long()   # 0 or 1
        return image, label


# ---------------------------------------------------------------------------
# Standard transform for both tasks
# ---------------------------------------------------------------------------

def build_transform(image_size: int = 64) -> transforms.Compose:
    """
    CenterCrop makes the 178×218 CelebA images square (178×178),
    then Resize brings them to image_size × image_size.
    Normalize maps [0,1] → [-1,1] with per-channel mean/std = 0.5.
    """
    return transforms.Compose([
        transforms.CenterCrop(178),
        transforms.Resize(image_size),
        transforms.ToTensor(),                              # → [0, 1]
        transforms.Normalize((0.5, 0.5, 0.5),             # → [-1, 1]
                             (0.5, 0.5, 0.5)),
    ])


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def get_celeba_loader(
    root: str = "./data",
    split: str = "train",
    image_size: int = 64,
    batch_size: int = 64,
    num_workers: int = 2,
    attr_name: Optional[str] = None,
    download: bool = False,
    max_samples: Optional[int] = None,
) -> DataLoader:
    """
    Build and return a DataLoader for CelebA.

    Parameters
    ----------
    root        : directory containing the `celeba/` sub-folder
    split       : 'train', 'valid', or 'test'
    image_size  : spatial size of returned images (default 64)
    batch_size  : samples per mini-batch
    num_workers : DataLoader workers (set 0 on Windows if you hit issues)
    attr_name   : if given, each batch yields (images, labels); used in Task 2
    download    : attempt torchvision auto-download (often fails on Colab —
                  see module docstring)
    max_samples : truncate dataset to this many images (useful on Colab free
                  tier to stay within compute budget)

    Returns
    -------
    DataLoader whose batches are:
        Tensor (B, 3, 64, 64) in [-1, 1]          — Task 1
        (Tensor (B,3,64,64), Tensor (B,))          — Task 2
    """
    transform = build_transform(image_size)
    dataset   = CelebAWrapper(
        root=root,
        split=split,
        transform=transform,
        attr_name=attr_name,
        download=download,
    )

    if max_samples is not None:
        indices = list(range(min(max_samples, len(dataset))))
        dataset = torch.utils.data.Subset(dataset, indices)

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=(split == "train"),
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=True,   # keeps batch size fixed; important for GP computation
    )
    return loader


# ---------------------------------------------------------------------------
# Sanity check
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import torchvision.utils as vutils

    DATA_ROOT = "./data"      # <— point this at your CelebA root on Colab

    print("Loading CelebA train split (images only) …")
    loader = get_celeba_loader(
        root=DATA_ROOT,
        split="train",
        batch_size=16,
        num_workers=0,       # safe default for quick test
        download=False,
        max_samples=256,     # fast check — remove for full training
    )

    batch = next(iter(loader))
    print(f"Batch shape  : {batch.shape}")          # (16, 3, 64, 64)
    print(f"Value range  : [{batch.min():.2f}, {batch.max():.2f}]")   # ~[-1, 1]
    print(f"Dataset size : {len(loader.dataset)}")

    # Task-2 path: images + attribute labels
    print("\nLoading with attribute 'Smiling' …")
    loader_attr = get_celeba_loader(
        root=DATA_ROOT,
        split="train",
        batch_size=16,
        num_workers=0,
        attr_name="Smiling",
        max_samples=256,
    )
    imgs, labels = next(iter(loader_attr))
    print(f"Image shape  : {imgs.shape}")            # (16, 3, 64, 64)
    print(f"Label shape  : {labels.shape}")          # (16,)
    print(f"Label values : {labels.tolist()}")       # list of 0s and 1s
