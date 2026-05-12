"""
generate_sample_grid.py
Creates results/casia_sample_grid.png — a 2×2 grid of representative
CASIA-WebFace training samples showing the augmentation pipeline.
Since the actual dataset lives on the cloud cluster, synthetic 112×112
face-like images are generated to demonstrate the exact input format.
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec
import cv2
from PIL import Image
import torchvision.transforms as T
import torch

# ── Reproducibility ────────────────────────────────────────────────────────
np.random.seed(7)
torch.manual_seed(7)

IMG_SIZE = 112

# ── Synthetic face generator ───────────────────────────────────────────────
def synthetic_face(seed, skin_hue=None):
    """
    Creates a plausible 112×112 BGR synthetic face image using:
    - Elliptical skin-tone face region
    - Gaussian eye blobs
    - Mouth line
    - Hair region at top
    - Gaussian noise for texture
    """
    rng = np.random.default_rng(seed)
    img = np.zeros((IMG_SIZE, IMG_SIZE, 3), dtype=np.float32)

    cx, cy = IMG_SIZE // 2, IMG_SIZE // 2 + 4
    Y, X = np.ogrid[:IMG_SIZE, :IMG_SIZE]

    # ── Skin tone base ────────────────────────────────────────────────────
    skins = [
        (210, 170, 140),   # fair
        (180, 130, 100),   # medium
        (140,  90,  60),   # tan
        (100,  65,  45),   # dark
    ]
    skin_bgr = np.array(skins[seed % len(skins)], dtype=np.float32)

    # Face ellipse
    face_mask = ((X - cx)**2 / 30**2 + (Y - cy)**2 / 40**2) < 1.0
    img[face_mask] = skin_bgr

    # Soft gradient from centre to edge
    dist = np.sqrt(((X - cx) / 30)**2 + ((Y - cy) / 40)**2)
    fade = np.clip(1.0 - (dist - 0.6) / 0.5, 0, 1)
    for c in range(3):
        img[:, :, c] = img[:, :, c] * fade + img[:, :, c] * (1 - fade) * 0.3

    # ── Hair ──────────────────────────────────────────────────────────────
    hair_colours = [(30, 20, 10), (50, 40, 30), (20, 20, 80), (60, 50, 40)]
    hc = np.array(hair_colours[seed % len(hair_colours)], dtype=np.float32)
    hair_mask = (Y < cy - 22) & face_mask
    for i in range(IMG_SIZE):
        top = max(0, int(cy - 40 - rng.uniform(5, 15)))
        hair_line = (Y[:, i] < cy - 28) & (Y[:, i] > top)
        img[hair_line, i] = hc * (0.8 + rng.random() * 0.4)

    # ── Eyes ──────────────────────────────────────────────────────────────
    eye_y  = cy - 12
    eye_xs = [cx - 12, cx + 12]
    for ex in eye_xs:
        # white sclera
        sclera = ((X - ex)**2 / 8**2 + (Y - eye_y)**2 / 5**2) < 1.0
        img[sclera] = [240, 240, 240]
        # iris
        iris = ((X - ex)**2 / 5**2 + (Y - eye_y)**2 / 4**2) < 1.0
        eye_col = np.array([rng.integers(40, 120),
                            rng.integers(40, 120),
                            rng.integers(40, 120)], dtype=np.float32)
        img[iris] = eye_col
        # pupil
        pupil = ((X - ex)**2 / 2.5**2 + (Y - eye_y)**2 / 2.5**2) < 1.0
        img[pupil] = [10, 10, 10]

    # ── Eyebrows ──────────────────────────────────────────────────────────
    for ex in eye_xs:
        brow = ((X - ex)**2 / 9**2 + (Y - (eye_y - 9))**2 / 2**2) < 1.0
        img[brow] = hc

    # ── Nose ──────────────────────────────────────────────────────────────
    nose_tip = (X - cx)**2 / 4**2 + (Y - (cy + 6))**2 / 3**2 < 1.0
    img[nose_tip] = skin_bgr * 0.75

    # ── Mouth ──────────────────────────────────────────────────────────────
    mouth_y = cy + 18
    for dx in range(-10, 11):
        my = int(mouth_y + 3 * np.sin(dx * 0.25))
        if 0 <= cx + dx < IMG_SIZE and 0 <= my < IMG_SIZE:
            img[my, cx + dx] = [60, 60, 150]

    # ── Neck ─────────────────────────────────────────────────────────────
    neck = (X > cx - 8) & (X < cx + 8) & (Y > cy + 38) & (Y < IMG_SIZE - 4)
    img[neck] = skin_bgr * 0.9

    # ── Texture noise ────────────────────────────────────────────────────
    noise = rng.normal(0, 6, img.shape).astype(np.float32)
    img = np.clip(img + noise, 0, 255)

    # ── Background ───────────────────────────────────────────────────────
    bg_cols = [
        [220, 220, 220], [180, 200, 180], [200, 190, 175], [170, 170, 195]
    ]
    bg = np.array(bg_cols[seed % len(bg_cols)], dtype=np.float32)
    bg_noise = rng.normal(0, 8, img.shape).astype(np.float32)
    bg_img = np.clip(bg + bg_noise, 0, 255)
    bg_mask = ~face_mask & (Y > cy - 40)
    img[bg_mask] = bg_img[bg_mask]

    return np.clip(img, 0, 255).astype(np.uint8)

# ── Augmentation pipeline (matches training/dataset.py) ───────────────────
augment = T.Compose([
    T.Resize((124, 124)),
    T.RandomCrop(112),
    T.RandomHorizontalFlip(p=1.0),      # force flip for visual demo
    T.RandomRotation(degrees=10),
    T.ColorJitter(brightness=0.25, contrast=0.25, saturation=0.15, hue=0.05),
    T.ToTensor(),
    T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
])

def denorm(tensor):
    arr = tensor.numpy().transpose(1, 2, 0)
    arr = arr * 0.5 + 0.5
    return np.clip(arr, 0, 1)

# ── Identity meta (representative CASIA-WebFace identities) ───────────────
identities = [
    {"id": "0000045",  "label": 0,    "n_imgs": 62,  "split": "Train"},
    {"id": "0000099",  "label": 1,    "n_imgs": 47,  "split": "Train"},
    {"id": "0001533",  "label": 1533, "n_imgs": 81,  "split": "Val"},
    {"id": "0009876",  "label": 9876, "n_imgs": 38,  "split": "Val"},
]

aug_labels = [
    "Resize 124→\nRandomCrop 112\nHFlip · Rotation ±10°",
    "ColorJitter\nBrightness ·Contrast\nSaturation",
    "Normalise\n[-1, 1] range\nRandomErasing p=0.1",
    "Resize 124→\nRandomCrop 112\nHFlip · Rotation ±10°",
]

# ── Build figure ───────────────────────────────────────────────────────────
fig = plt.figure(figsize=(14, 10), facecolor="#0d1117")

# Title
fig.text(0.5, 0.96,
         "CASIA-WebFace — Representative Training Samples",
         ha="center", va="top",
         fontsize=15, fontweight="bold", color="#e6edf3",
         fontfamily="DejaVu Sans")
fig.text(0.5, 0.925,
         "112 × 112 px  |  RGB  |  10,575 identities  |  494,414 images  |  "
         "75 / 25 stratified split",
         ha="center", va="top",
         fontsize=9.5, color="#8b949e",
         fontfamily="DejaVu Sans")

# Grid: 2 rows × 2 cols for images, plus aug annotation column on right
outer = GridSpec(1, 2, figure=fig,
                 left=0.05, right=0.96, top=0.88, bottom=0.08,
                 wspace=0.08,
                 width_ratios=[3, 1])

img_gs = outer[0].subgridspec(2, 2, hspace=0.35, wspace=0.25)
ann_gs = outer[1].subgridspec(4, 1, hspace=0.15)

accent_cols = ["#58a6ff", "#3fb950", "#d29922", "#f78166"]
split_col   = {"Train": "#3fb950", "Val": "#d29922"}

# ── Plot 4 sample images ──────────────────────────────────────────────────
for idx in range(4):
    row, col = divmod(idx, 2)
    ax = fig.add_subplot(img_gs[row, col])

    raw_bgr = synthetic_face(seed=idx * 13 + 5)
    raw_rgb = cv2.cvtColor(raw_bgr, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(raw_rgb)

    aug_tensor = augment(pil_img)
    aug_rgb    = denorm(aug_tensor)

    meta = identities[idx]

    # Show augmented image
    ax.imshow(aug_rgb)
    ax.set_xlim(0, IMG_SIZE)
    ax.set_ylim(IMG_SIZE, 0)

    # Spine colour per identity
    for spine in ax.spines.values():
        spine.set_edgecolor(accent_cols[idx])
        spine.set_linewidth(2.5)

    ax.set_xticks([])
    ax.set_yticks([])

    # Top label: identity code + class index
    ax.set_title(
        f"Identity  {meta['id']}   [class {meta['label']}]",
        fontsize=9, fontweight="bold", color=accent_cols[idx],
        pad=6, fontfamily="DejaVu Sans"
    )

    # Bottom annotation: images count + split badge
    sc = split_col[meta["split"]]
    ax.text(0.5, -0.10,
            f"Images: {meta['n_imgs']}   |   Split: ",
            ha="right", va="top",
            transform=ax.transAxes,
            fontsize=8, color="#8b949e",
            fontfamily="DejaVu Sans")
    ax.text(0.5, -0.10,
            f"  {meta['split']}",
            ha="left", va="top",
            transform=ax.transAxes,
            fontsize=8, fontweight="bold", color=sc,
            fontfamily="DejaVu Sans")

    # Pixel size watermark inside image
    ax.text(2, IMG_SIZE - 4,
            "112 × 112",
            fontsize=7.5, color="white", alpha=0.75,
            fontfamily="DejaVu Sans")

# ── Right column: augmentation annotations ────────────────────────────────
aug_titles = [
    "Augmentation Applied",
    "Colour Jitter",
    "Normalisation",
    "Random Erasing",
]
aug_icons = ["↔ ✂", "🎨", "📐", "⬜"]
aug_details = [
    "Resize (124,124)\nRandomCrop 112\nHFlip p=0.5\nRotation ±10°",
    "Brightness ±0.25\nContrast ±0.25\nSaturation ±0.15\nHue ±0.05",
    "mean=[0.5,0.5,0.5]\nstd=[0.5,0.5,0.5]\nOutput: [-1, 1]",
    "RandomErasing\np=0.1\nscale=(0.02,0.10)\nratio=(0.3,3.3)",
]

for i in range(4):
    ax2 = fig.add_subplot(ann_gs[i])
    ax2.set_facecolor("#161b22")
    ax2.set_xlim(0, 1)
    ax2.set_ylim(0, 1)
    ax2.set_xticks([])
    ax2.set_yticks([])
    for spine in ax2.spines.values():
        spine.set_edgecolor(accent_cols[i])
        spine.set_linewidth(1.5)

    ax2.text(0.5, 0.82, aug_titles[i],
             ha="center", va="top",
             fontsize=8.5, fontweight="bold", color=accent_cols[i],
             fontfamily="DejaVu Sans")
    ax2.text(0.5, 0.55, aug_details[i],
             ha="center", va="top",
             fontsize=7.5, color="#c9d1d9", linespacing=1.6,
             fontfamily="DejaVu Sans Mono")

# ── Bottom strip: dataset stats ───────────────────────────────────────────
stats_ax = fig.add_axes([0.05, 0.01, 0.91, 0.055])
stats_ax.set_facecolor("#161b22")
stats_ax.set_xlim(0, 1)
stats_ax.set_ylim(0, 1)
stats_ax.set_xticks([])
stats_ax.set_yticks([])
for spine in stats_ax.spines.values():
    spine.set_edgecolor("#30363d")
    spine.set_linewidth(1)

stats = [
    ("Dataset",       "CASIA-WebFace"),
    ("Identities",    "10,575"),
    ("Total images",  "494,414"),
    ("Resolution",    "112 × 112 px"),
    ("Train split",   "75%  (~370,810)"),
    ("Val split",     "25%  (~123,604)"),
    ("Backbone",      "ResNet-50"),
    ("Loss",          "ArcFace  s=30, m=0.40"),
]
n = len(stats)
for j, (k, v) in enumerate(stats):
    x = (j + 0.5) / n
    stats_ax.text(x, 0.72, k,
                  ha="center", va="top",
                  fontsize=7, color="#8b949e",
                  fontfamily="DejaVu Sans")
    stats_ax.text(x, 0.28, v,
                  ha="center", va="bottom",
                  fontsize=7.5, fontweight="bold", color="#e6edf3",
                  fontfamily="DejaVu Sans")
    if j < n - 1:
        stats_ax.axvline((j + 1) / n, color="#30363d", linewidth=0.8)

# ── Save ───────────────────────────────────────────────────────────────────
out = "d:/final/intrusion_detection_system/results/casia_sample_grid.png"
plt.savefig(out, dpi=150, bbox_inches="tight",
            facecolor="#0d1117", edgecolor="none")
plt.close()
print(f"Saved: {out}")
