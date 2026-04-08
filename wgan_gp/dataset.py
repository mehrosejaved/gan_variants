"""
dataset.py — CelebA data loading for WGAN-GP (Task 1) and Conditional WGAN-GP (Task 2).

Uses a custom reader instead of torchvision.datasets.CelebA so it works with
the Kaggle download (jessicali9530/celeba-dataset) without any MD5 / folder
restructuring.  Both .csv (Kaggle) and space-delimited .txt (original) formats
are handled automatically.

CelebA image dimensions are 178 × 218 (W × H).
Pipeline: CenterCrop(178) → Resize(64) → Tensor → Normalize[-1, 1].

Usage
-----
# Task 1 — images only
loader = get_celeba_loader(root="./data", batch_size=64)

# Task 2 — images + one binary attribute label
loader = get_celeba_loader(root="./data", batch_size=64, attr_name="Smiling")

Expected folder layout after Kaggle unzip
-----------------------------------------
Torchvision layout (also accepted):
    root/celeba/img_align_celeba/*.jpg
    root/celeba/list_attr_celeba.txt
    root/celeba/list_eval_partition.txt

Kaggle layout (also accepted — no restructuring needed):
    root/img_align_celeba/img_align_celeba/*.jpg   ← nested folder
    root/list_attr_celeba.csv
    root/list_eval_partition.csv
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms


# ---------------------------------------------------------------------------
# All 40 CelebA attribute names (column order in list_attr_celeba.*)
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

# Split index: 0 = train, 1 = valid, 2 = test
_SPLIT_MAP = {"train": 0, "valid": 1, "test": 2}


# ---------------------------------------------------------------------------
# File / folder discovery helpers
# ---------------------------------------------------------------------------

def _find_file(root: str, name_stem: str) -> Path:
    """
    Search root and one level of subdirectories for a file whose stem
    matches name_stem.  Accepts both .txt and .csv extensions.
    Raises FileNotFoundError with a helpful message if not found.
    """
    search_dirs = [Path(root)] + list(Path(root).iterdir().__class__
                                      .__mro__)  # placeholder — replaced below
    search_dirs = [Path(root)] + [p for p in Path(root).iterdir() if p.is_dir()]

    for d in search_dirs:
        for ext in (".txt", ".csv"):
            candidate = d / (name_stem + ext)
            if candidate.exists():
                return candidate

    raise FileNotFoundError(
        f"Could not find '{name_stem}.txt' or '{name_stem}.csv' "
        f"anywhere under '{root}'.\n"
        f"Contents of root: {list(Path(root).iterdir())}"
    )


def _find_image_dir(root: str) -> Path:
    """
    Recursively find the directory that actually contains .jpg files,
    searching for a folder named 'img_align_celeba' at any nesting depth.
    Falls back to the deepest 'img_align_celeba' folder found.
    """
    for dirpath, dirnames, filenames in os.walk(root):
        if os.path.basename(dirpath) == "img_align_celeba":
            jpgs = [f for f in filenames if f.lower().endswith(".jpg")]
            if jpgs:
                return Path(dirpath)

    raise FileNotFoundError(
        f"Could not find any 'img_align_celeba' folder with .jpg files "
        f"under '{root}'.\nPlease check your CelebA download."
    )


# ---------------------------------------------------------------------------
# Annotation parsers  (handle both .csv Kaggle and space-delimited .txt)
# ---------------------------------------------------------------------------

def _parse_attr_file(path: Path) -> dict[str, list[int]]:
    """
    Returns {filename: [attr0, attr1, ..., attr39]} with values in {0, 1}.

    Handles:
      • Original .txt  — first line = count, second = header, then data
        (values are -1 / 1; we convert to 0 / 1)
      • Kaggle   .csv  — first line = header, then data (values are 0 / 1)
    """
    lines = path.read_text().strip().splitlines()

    if path.suffix == ".csv":
        # header: image_id,attr1,attr2,...
        data_lines = lines[1:]
        result = {}
        for line in data_lines:
            parts = line.split(",")
            fname = parts[0].strip()
            vals  = [max(0, int(v)) for v in parts[1:]]   # already 0/1
            result[fname] = vals
        return result
    else:
        # .txt: first line may be count, second is header
        start = 0
        if lines[0].strip().isdigit():
            start = 2          # skip count + header line
        elif not lines[0].strip()[0].isdigit():
            start = 1          # skip header line only
        result = {}
        for line in lines[start:]:
            parts = line.split()
            fname = parts[0].strip()
            # -1 → 0,  1 → 1
            vals  = [1 if int(v) == 1 else 0 for v in parts[1:]]
            result[fname] = vals
        return result


def _parse_partition_file(path: Path) -> dict[str, int]:
    """
    Returns {filename: split_int} where split_int ∈ {0, 1, 2}.

    Handles:
      • Original .txt — 'filename split_int' per line, no header
      • Kaggle   .csv — header 'image_id,partition', then data
    """
    lines = path.read_text().strip().splitlines()

    if path.suffix == ".csv":
        data_lines = lines[1:]   # skip header
        return {
            parts[0].strip(): int(parts[1])
            for line in data_lines
            if len(parts := line.split(",")) == 2
        }
    else:
        result = {}
        for line in lines:
            parts = line.split()
            if len(parts) == 2:
                result[parts[0].strip()] = int(parts[1])
        return result


# ---------------------------------------------------------------------------
# Custom CelebA Dataset (no torchvision dependency)
# ---------------------------------------------------------------------------

class KaggleCelebA(Dataset):
    """
    Reads CelebA from the folder layout produced by the Kaggle download.
    Works equally well with the original torchvision layout.

    Parameters
    ----------
    root      : top-level data directory (e.g. './data')
    split     : 'train', 'valid', or 'test'
    transform : torchvision transforms applied to each PIL image
    attr_name : if given, also returns a binary label for this attribute
    """

    def __init__(
        self,
        root: str,
        split: str = "train",
        transform=None,
        attr_name: Optional[str] = None,
    ) -> None:
        if split not in _SPLIT_MAP:
            raise ValueError(f"split must be one of {list(_SPLIT_MAP)}, got '{split}'")

        self.transform  = transform
        self.attr_idx: Optional[int] = None

        if attr_name is not None:
            if attr_name not in CELEBA_ATTRS:
                raise ValueError(f"Unknown attribute '{attr_name}'. Valid: {CELEBA_ATTRS}")
            self.attr_idx = CELEBA_ATTRS.index(attr_name)

        # Locate files
        self.img_dir   = _find_image_dir(root)
        attr_path      = _find_file(root, "list_attr_celeba")
        partition_path = _find_file(root, "list_eval_partition")

        print(f"[KaggleCelebA] images    : {self.img_dir}")
        print(f"[KaggleCelebA] attrs     : {attr_path}")
        print(f"[KaggleCelebA] partition : {partition_path}")

        # Parse annotations
        attr_map      = _parse_attr_file(attr_path)
        partition_map = _parse_partition_file(partition_path)

        target_split  = _SPLIT_MAP[split]

        # Build index: only filenames belonging to the requested split
        self.filenames: list[str] = []
        self.labels:    list[list[int]] = []

        for fname, split_id in partition_map.items():
            if split_id == target_split and fname in attr_map:
                self.filenames.append(fname)
                self.labels.append(attr_map[fname])

        print(f"[KaggleCelebA] split='{split}' → {len(self.filenames):,} images")

    def __len__(self) -> int:
        return len(self.filenames)

    def __getitem__(self, idx: int):
        img_path = self.img_dir / self.filenames[idx]
        image    = Image.open(img_path).convert("RGB")

        if self.transform is not None:
            image = self.transform(image)

        if self.attr_idx is None:
            return image

        label = torch.tensor(self.labels[idx][self.attr_idx], dtype=torch.long)
        return image, label


# ---------------------------------------------------------------------------
# Standard transform
# ---------------------------------------------------------------------------

def build_transform(image_size: int = 64) -> transforms.Compose:
    """
    CenterCrop(178) makes the portrait image square, Resize(64) downsamples.
    Normalize maps [0, 1] → [-1, 1].
    """
    return transforms.Compose([
        transforms.CenterCrop(178),
        transforms.Resize(image_size),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
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
    download: bool = False,   # kept for API compatibility; ignored (use Kaggle)
    max_samples: Optional[int] = None,
) -> DataLoader:
    """
    Build and return a DataLoader for CelebA.

    Parameters
    ----------
    root        : directory containing CelebA files (any layout)
    split       : 'train', 'valid', or 'test'
    image_size  : spatial size of returned images (default 64)
    batch_size  : samples per mini-batch
    num_workers : DataLoader workers (use 0 if you hit multiprocessing errors)
    attr_name   : if given, batches yield (images, labels) — used in Task 2
    download    : ignored; kept for backwards compatibility
    max_samples : cap dataset size (useful on free Colab tier)

    Returns
    -------
    DataLoader whose batches are:
        Tensor (B, 3, 64, 64) in [-1, 1]        — Task 1
        (Tensor (B, 3, 64, 64), Tensor (B,))    — Task 2
    """
    transform = build_transform(image_size)
    dataset   = KaggleCelebA(
        root=root,
        split=split,
        transform=transform,
        attr_name=attr_name,
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
        drop_last=True,
    )
    return loader


# ---------------------------------------------------------------------------
# Sanity check
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    DATA_ROOT = "./data"

    print("=== Task 1: images only ===")
    loader = get_celeba_loader(
        root=DATA_ROOT,
        split="train",
        batch_size=16,
        num_workers=0,
        max_samples=256,
    )
    batch = next(iter(loader))
    print(f"Batch shape  : {batch.shape}")
    print(f"Value range  : [{batch.min():.2f}, {batch.max():.2f}]")

    print("\n=== Task 2: images + Smiling label ===")
    loader_attr = get_celeba_loader(
        root=DATA_ROOT,
        split="train",
        batch_size=16,
        num_workers=0,
        attr_name="Smiling",
        max_samples=256,
    )
    imgs, labels = next(iter(loader_attr))
    print(f"Image shape  : {imgs.shape}")
    print(f"Label shape  : {labels.shape}")
    print(f"Label values : {labels.tolist()}")
