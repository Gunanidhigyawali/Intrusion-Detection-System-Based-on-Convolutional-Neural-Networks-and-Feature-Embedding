import os
import random
import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
from torchvision.transforms import RandomErasing
from sklearn.model_selection import train_test_split

import config as _cfg
from config import IMG_SIZE, TRAIN_SPLIT, SEED


def _set_seed(seed=SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def _scan_casia(data_path):
    """Return list of (image_path, label_int) for all images in CASIA tree."""
    identity_dirs = sorted(
        d for d in os.listdir(data_path)
        if os.path.isdir(os.path.join(data_path, d))
    )
    samples = []
    for label, identity in enumerate(identity_dirs):
        folder = os.path.join(data_path, identity)
        for fname in os.listdir(folder):
            if fname.lower().endswith((".jpg", ".jpeg", ".png")):
                samples.append((os.path.join(folder, fname), label))
    return samples, len(identity_dirs)


def split_dataset(data_path, train_split=TRAIN_SPLIT, seed=SEED):
    """
    Stratified image-level split so that EVERY identity appears in training.

    Why stratified (not identity-level)?
      ArcFace has one weight row per training identity. If val identities are
      excluded from training, the model has no weights for them and val
      loss/accuracy becomes meaningless. Stratify keeps every class in train;
      val is used only to detect overfitting during training.

    Split layout:
      all images  →  90% train  (stratified by identity)
                  →  10% val    (capped at VAL_SIZE for speed)

    Returns (train_samples, val_samples, num_classes).
    """
    _set_seed(seed)

    all_samples, num_classes = _scan_casia(data_path)
    all_labels = [label for _, label in all_samples]

    train_data, val_data = train_test_split(
        all_samples,
        train_size=train_split,
        stratify=all_labels,   # every identity proportionally in train
        random_state=seed,
    )

    val_size = _cfg.VAL_SIZE   # read at call time so CLI overrides apply
    if val_size and len(val_data) > val_size:
        val_data = val_data[:val_size]

    print(f"\n  Train : {len(train_data):>7,} images  (all {num_classes} identities)")
    cap_note = f"capped at {val_size}" if val_size else "full 25% — no cap"
    print(f"  Val   : {len(val_data):>7,} images  ({cap_note})")

    return train_data, val_data, num_classes


# ── Transforms ────────────────────────────────────────────────────────────────

_img_size = IMG_SIZE[0]   # model expects square images

train_transform = transforms.Compose([
    transforms.Resize((_img_size + 12, _img_size + 12)),
    transforms.RandomCrop(_img_size),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(10),
    transforms.ColorJitter(brightness=0.2, contrast=0.2,
                           saturation=0.15, hue=0.05),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    RandomErasing(p=0.1, scale=(0.02, 0.1)),
])

val_transform = transforms.Compose([
    transforms.Resize((_img_size, _img_size)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
])


# ── Dataset ───────────────────────────────────────────────────────────────────

class FaceDataset(Dataset):
    def __init__(self, samples, transform=None):
        self.samples = samples
        self.transform = transform

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        img = Image.open(path).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img, label
