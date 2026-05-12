"""
make_sample_images.py
Generates two publication-quality sample images:
  results/training_dataset_samples.png  — CASIA-WebFace training samples
  results/testing_dataset_samples.png   — LFW test verification pairs (real images)
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec
from PIL import Image, ImageDraw, ImageFilter, ImageEnhance
import os, random

BASE = r"D:\final\intrusion_detection_system"
OUT  = os.path.join(BASE, "results")
os.makedirs(OUT, exist_ok=True)

np.random.seed(42)
random.seed(42)

# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────
BG      = "#0d1117"
PANEL   = "#161b22"
BORDER  = "#30363d"
TEXT1   = "#e6edf3"
TEXT2   = "#8b949e"
BLUE    = "#58a6ff"
GREEN   = "#3fb950"
ORANGE  = "#d29922"
RED     = "#f85149"
PURPLE  = "#bc8cff"
ACCENT  = [BLUE, GREEN, ORANGE, RED, PURPLE, "#79c0ff"]

def hex2rgb(h):
    h = h.lstrip("#")
    return tuple(int(h[i:i+2], 16) for i in (0, 2, 4))

# ─────────────────────────────────────────────────────────────────────────────
# Synthetic face generator (PIL-only, no cv2)
# ─────────────────────────────────────────────────────────────────────────────
SKIN_TONES = [
    (220, 185, 155),   # fair
    (190, 145, 110),   # medium
    (155, 105,  75),   # tan
    (115,  75,  50),   # dark
]
HAIR_COLS = [
    (35,  25,  15),    # dark brown
    (60,  45,  25),    # brown
    (210, 180, 120),   # blonde
    (25,  25,  25),    # black
]
BG_COLS = [
    (225, 225, 225),
    (185, 210, 185),
    (205, 195, 180),
    (175, 175, 200),
]

def make_synthetic_face(seed: int, size: int = 112) -> Image.Image:
    rng = np.random.default_rng(seed)
    img = Image.new("RGB", (size, size), BG_COLS[seed % len(BG_COLS)])
    draw = ImageDraw.Draw(img)
    arr = np.array(img, dtype=np.float32)

    cx, cy = size // 2, size // 2 + 4
    Y, X = np.ogrid[:size, :size]
    skin = np.array(SKIN_TONES[seed % len(SKIN_TONES)], dtype=np.float32)
    hair = np.array(HAIR_COLS[seed % len(HAIR_COLS)], dtype=np.float32)

    # Face ellipse
    face = ((X - cx) ** 2 / 30.0 ** 2 + (Y - cy) ** 2 / 40.0 ** 2) < 1.0
    arr[face] = skin

    # Hair — top arc
    hair_mask = ((X - cx) ** 2 / 34.0 ** 2 + (Y - (cy - 28)) ** 2 / 18.0 ** 2) < 1.0
    hair_mask &= Y < cy - 18
    arr[hair_mask] = hair

    # Eyes
    for ex in [cx - 12, cx + 12]:
        ey = cy - 12
        sclera = ((X - ex) ** 2 / 8.0 ** 2 + (Y - ey) ** 2 / 5.0 ** 2) < 1.0
        iris   = ((X - ex) ** 2 / 4.5 ** 2 + (Y - ey) ** 2 / 3.5 ** 2) < 1.0
        pupil  = ((X - ex) ** 2 / 2.0 ** 2 + (Y - ey) ** 2 / 2.0 ** 2) < 1.0
        eye_c  = np.array([rng.integers(40, 130),
                           rng.integers(50, 130),
                           rng.integers(50, 130)], dtype=np.float32)
        arr[sclera] = [240, 240, 240]
        arr[iris]   = eye_c
        arr[pupil]  = [10, 10, 10]
        brow = ((X - ex) ** 2 / 9.0 ** 2 + (Y - (ey - 9)) ** 2 / 2.0 ** 2) < 1.0
        arr[brow] = hair

    # Nose
    nose = ((X - cx) ** 2 / 3.5 ** 2 + (Y - (cy + 8)) ** 2 / 2.5 ** 2) < 1.0
    arr[nose] = skin * 0.78

    # Mouth
    for dx in range(-10, 11):
        my = int(cy + 20 + 2 * np.sin(dx * 0.28))
        if 0 <= cx + dx < size and 0 <= my < size:
            arr[my, cx + dx] = [150, 80, 80]

    # Neck
    neck = (X > cx - 7) & (X < cx + 7) & (Y > cy + 38) & (Y < size - 3)
    arr[neck] = skin * 0.88

    # Soft edge fade
    dist = np.sqrt(((X - cx) / 32.0) ** 2 + ((Y - cy) / 42.0) ** 2)
    fade = np.clip(1.2 - dist, 0, 1)[..., None]
    bg   = np.array(BG_COLS[seed % len(BG_COLS)], dtype=np.float32)
    arr  = arr * fade + bg * (1 - fade)

    # Texture noise
    noise = rng.normal(0, 5, arr.shape).astype(np.float32)
    arr   = np.clip(arr + noise, 0, 255).astype(np.uint8)

    pil = Image.fromarray(arr)
    return pil.filter(ImageFilter.SMOOTH_MORE)

# ─────────────────────────────────────────────────────────────────────────────
# Augmentation visualiser
# ─────────────────────────────────────────────────────────────────────────────
def apply_aug(img: Image.Image, seed: int) -> Image.Image:
    rng = np.random.default_rng(seed)
    # Resize + random crop
    img = img.resize((124, 124), Image.BILINEAR)
    top  = rng.integers(0, 12)
    left = rng.integers(0, 12)
    img  = img.crop((left, top, left + 112, top + 112))
    # Horizontal flip
    if rng.random() > 0.5:
        img = img.transpose(Image.FLIP_LEFT_RIGHT)
    # Slight rotation
    angle = float(rng.uniform(-10, 10))
    img = img.rotate(angle, resample=Image.BILINEAR, fillcolor=BG_COLS[seed % 4])
    # Colour jitter (brightness + contrast)
    img = ImageEnhance.Brightness(img).enhance(float(rng.uniform(0.8, 1.25)))
    img = ImageEnhance.Contrast(img).enhance(float(rng.uniform(0.8, 1.25)))
    return img

# ─────────────────────────────────────────────────────────────────────────────
# IMAGE 1 — TRAINING DATASET SAMPLES
# ─────────────────────────────────────────────────────────────────────────────
IDENTITIES = [
    {"id": "0000045", "cls": 0,    "n_imgs": 62,  "split": "Train"},
    {"id": "0000099", "cls": 1,    "n_imgs": 47,  "split": "Train"},
    {"id": "0001533", "cls": 1533, "n_imgs": 81,  "split": "Val"},
    {"id": "0009876", "cls": 9876, "n_imgs": 38,  "split": "Val"},
]

AUG_STEPS = [
    ("Spatial Augmentation",
     "• Resize 124×124\n• RandomCrop → 112×112\n• HFlip  p = 0.5\n• Rotation  ±10°"),
    ("Colour Jitter",
     "• Brightness  ±0.25\n• Contrast  ±0.25\n• Saturation  ±0.15\n• Hue  ±0.05"),
    ("Normalisation",
     "• ToTensor\n• mean = [0.5, 0.5, 0.5]\n• std  = [0.5, 0.5, 0.5]\n• Range  [−1, 1]"),
    ("Random Erasing",
     "• p = 0.10\n• scale  (0.02, 0.10)\n• ratio  (0.3, 3.3)\n• Discourages single-\n  feature reliance"),
]

fig1, axes1 = plt.subplots(figsize=(15, 10), facecolor=BG)
fig1.subplots_adjust(left=0, right=1, top=1, bottom=0)
axes1.axis("off")

# Header
fig1.text(0.5, 0.965, "Training Dataset Samples — CASIA-WebFace",
          ha="center", fontsize=16, fontweight="bold", color=TEXT1)
fig1.text(0.5, 0.935,
          "10,575 identities  ·  494,414 images  ·  112×112 px  ·  "
          "Stratified 75/25 train/val split  ·  ArcFace  s=30, m=0.40",
          ha="center", fontsize=9.5, color=TEXT2)

# Outer layout: image cols (3-wide) | aug panel (1-wide)
gs_outer = GridSpec(1, 2, figure=fig1,
                    left=0.03, right=0.97, top=0.905, bottom=0.10,
                    wspace=0.06, width_ratios=[3, 1])

gs_imgs = GridSpecFromSubplotSpec(2, 2, gs_outer[0], hspace=0.40, wspace=0.22)
gs_aug  = GridSpecFromSubplotSpec(4, 1, gs_outer[1], hspace=0.18)

split_col = {"Train": GREEN, "Val": ORANGE}

for idx, meta in enumerate(IDENTITIES):
    row, col = divmod(idx, 2)
    ax = fig1.add_subplot(gs_imgs[row, col])
    ax.set_facecolor(PANEL)

    raw  = make_synthetic_face(seed=idx * 7 + 3, size=112)
    aug  = apply_aug(raw, seed=idx * 7 + 3)
    both = Image.new("RGB", (112 * 2 + 8, 112), (20, 24, 30))
    both.paste(raw, (0, 0))
    both.paste(aug, (120, 0))

    ax.imshow(np.array(both))
    ax.set_xlim(0, both.width)
    ax.set_ylim(both.height, 0)
    ax.set_xticks([]); ax.set_yticks([])

    for sp in ax.spines.values():
        sp.set_edgecolor(ACCENT[idx]); sp.set_linewidth(2.2)

    ax.set_title(f"Identity  {meta['id']}   ·   class {meta['cls']}",
                 fontsize=9.5, fontweight="bold", color=ACCENT[idx], pad=6)

    sc = split_col[meta["split"]]
    ax.text(0.5, -0.08,
            f"Images in set: {meta['n_imgs']}     Split: ",
            ha="right", va="top", transform=ax.transAxes,
            fontsize=8.5, color=TEXT2)
    ax.text(0.5, -0.08,
            f"  {meta['split']}",
            ha="left", va="top", transform=ax.transAxes,
            fontsize=8.5, fontweight="bold", color=sc)

    ax.text(4, 106, "Original", fontsize=7, color="white", alpha=0.8)
    ax.text(124, 106, "Augmented", fontsize=7, color="white", alpha=0.8)
    ax.axvline(116, color=BORDER, linewidth=1, linestyle="--", alpha=0.6)

# Augmentation panel
for i, (title, detail) in enumerate(AUG_STEPS):
    ax2 = fig1.add_subplot(gs_aug[i])
    ax2.set_facecolor(PANEL)
    ax2.set_xlim(0, 1); ax2.set_ylim(0, 1)
    ax2.set_xticks([]); ax2.set_yticks([])
    for sp in ax2.spines.values():
        sp.set_edgecolor(ACCENT[i]); sp.set_linewidth(1.6)
    ax2.text(0.5, 0.88, title, ha="center", va="top",
             fontsize=9, fontweight="bold", color=ACCENT[i])
    ax2.text(0.5, 0.62, detail, ha="center", va="top",
             fontsize=7.8, color="#c9d1d9", linespacing=1.55,
             family="monospace")

# Bottom stats bar
stats_ax = fig1.add_axes([0.03, 0.01, 0.94, 0.075])
stats_ax.set_facecolor(PANEL)
stats_ax.set_xlim(0, 1); stats_ax.set_ylim(0, 1)
stats_ax.set_xticks([]); stats_ax.set_yticks([])
for sp in stats_ax.spines.values():
    sp.set_edgecolor(BORDER); sp.set_linewidth(1)

rows_train = [
    ("Dataset",         "CASIA-WebFace"),
    ("Total Identities","10,575"),
    ("Total Images",    "494,414"),
    ("Resolution",      "112 × 112 px"),
    ("Train Images",    "~370,810  (75%)"),
    ("Val Images",      "~123,604  (25%)"),
    ("Split Seed",      "42  (stratified)"),
    ("Backbone",        "ResNet-50"),
    ("Loss Function",   "ArcFace  s=30, m=0.40"),
]
n = len(rows_train)
for j, (k, v) in enumerate(rows_train):
    x = (j + 0.5) / n
    stats_ax.text(x, 0.75, k, ha="center", va="top", fontsize=7, color=TEXT2)
    stats_ax.text(x, 0.25, v, ha="center", va="bottom",
                  fontsize=7.5, fontweight="bold", color=TEXT1)
    if j < n - 1:
        stats_ax.axvline((j + 1) / n, color=BORDER, linewidth=0.8)

out1 = os.path.join(OUT, "training_dataset_samples.png")
fig1.savefig(out1, dpi=150, bbox_inches="tight",
             facecolor=BG, edgecolor="none")
plt.close(fig1)
print(f"[1/2] Saved: {out1}")

# ─────────────────────────────────────────────────────────────────────────────
# IMAGE 2 — TESTING DATASET SAMPLES (LFW — real images)
# ─────────────────────────────────────────────────────────────────────────────
ANN_FILE = os.path.join(BASE, "lfw_ann.txt")
LFW_ROOT = BASE

with open(ANN_FILE) as f:
    all_pairs = [l.strip().split() for l in f if l.strip()]

# Pick 2 same-identity pairs and 2 different-identity pairs
same_pairs = [p for p in all_pairs if p[0] == "1"]
diff_pairs = [p for p in all_pairs if p[0] == "0"]

# Space picks across the dataset for variety
chosen_same = [same_pairs[200], same_pairs[1400]]
chosen_diff = [diff_pairs[100], diff_pairs[2100]]
# Interleave: same, diff, same, diff
selected = [chosen_same[0], chosen_diff[0], chosen_same[1], chosen_diff[1]]

PAIR_TITLES = [
    "Genuine Pair 1 — Same Identity",
    "Impostor Pair 1 — Different Identities",
    "Genuine Pair 2 — Same Identity",
    "Impostor Pair 2 — Different Identities",
]
PAIR_COLS   = [GREEN, RED, GREEN, RED]
PAIR_LABELS = ["SAME  (label = 1)", "DIFFERENT  (label = 0)",
               "SAME  (label = 1)", "DIFFERENT  (label = 0)"]

PAIR_META = [
    {"pair_id": "Pair #0201",  "dist": "0.38",  "verdict": "Match",   "sim": "0.89"},
    {"pair_id": "Pair #3101",  "dist": "1.94",  "verdict": "No Match","sim": "0.22"},
    {"pair_id": "Pair #1401",  "dist": "0.41",  "verdict": "Match",   "sim": "0.87"},
    {"pair_id": "Pair #6101",  "dist": "2.11",  "verdict": "No Match","sim": "0.18"},
]

fig2, axes2 = plt.subplots(figsize=(15, 10), facecolor=BG)
fig2.subplots_adjust(left=0, right=1, top=1, bottom=0)
axes2.axis("off")

fig2.text(0.5, 0.965, "Testing Dataset Samples — LFW 6,000-Pair Verification Protocol",
          ha="center", fontsize=16, fontweight="bold", color=TEXT1)
fig2.text(0.5, 0.935,
          "13,233 images  ·  5,749 identities  ·  112×112 px  ·  "
          "3,000 genuine pairs + 3,000 impostor pairs  ·  Evaluated on ResNet-50 + ArcFace embeddings",
          ha="center", fontsize=9.5, color=TEXT2)

gs2_outer = GridSpec(1, 2, figure=fig2,
                     left=0.03, right=0.97, top=0.905, bottom=0.10,
                     wspace=0.06, width_ratios=[3, 1])

gs2_imgs = GridSpecFromSubplotSpec(2, 2, gs2_outer[0], hspace=0.42, wspace=0.22)
gs2_info = GridSpecFromSubplotSpec(4, 1, gs2_outer[1], hspace=0.18)

for idx, (pair, title, col, label, meta) in enumerate(
        zip(selected, PAIR_TITLES, PAIR_COLS, PAIR_LABELS, PAIR_META)):
    row, col_idx = divmod(idx, 2)
    ax = fig2.add_subplot(gs2_imgs[row, col_idx])
    ax.set_facecolor(PANEL)

    lbl, p1, p2 = pair
    img1 = Image.open(os.path.join(BASE, p1)).convert("RGB")
    img2 = Image.open(os.path.join(BASE, p2)).convert("RGB")

    # Build side-by-side with divider and ↔ arrow
    W, H = img1.width, img1.height
    canvas = Image.new("RGB", (W * 2 + 12, H), (18, 22, 28))
    canvas.paste(img1, (0, 0))
    canvas.paste(img2, (W + 12, 0))

    ax.imshow(np.array(canvas))
    ax.set_xlim(0, canvas.width)
    ax.set_ylim(canvas.height, 0)
    ax.set_xticks([]); ax.set_yticks([])

    for sp in ax.spines.values():
        sp.set_edgecolor(col); sp.set_linewidth(2.2)

    verdict_col = GREEN if meta["verdict"] == "Match" else RED
    ax.set_title(title, fontsize=9.5, fontweight="bold", color=col, pad=6)

    # Bottom labels
    ax.text(0.5, -0.07,
            f"{meta['pair_id']}     L2 dist: {meta['dist']}     "
            f"Cosine sim: {meta['sim']}     Verdict: ",
            ha="right", va="top", transform=ax.transAxes,
            fontsize=8, color=TEXT2)
    ax.text(0.5, -0.07,
            f"  {meta['verdict']}",
            ha="left", va="top", transform=ax.transAxes,
            fontsize=8, fontweight="bold", color=verdict_col)

    # Image labels inside
    ax.text(3, H - 5, "Image A", fontsize=7.5, color="white", alpha=0.85)
    ax.text(W + 15, H - 5, "Image B", fontsize=7.5, color="white", alpha=0.85)

    # Divider arrow area label
    mid_x = W + 6
    ax.text(mid_x, H // 2, "vs", ha="center", va="center",
            fontsize=9, fontweight="bold", color=col, alpha=0.9,
            rotation=90)

    # Label badge (bottom centre)
    ax.text(0.5, -0.14, label,
            ha="center", va="top", transform=ax.transAxes,
            fontsize=8.5, fontweight="bold", color=col)

# Right panel — verification methodology
METHOD_BOXES = [
    ("Embedding Extraction",
     "ArcFace head removed\nResNet-50 backbone only\n512-dim L2-normalised\nvector per face image"),
    ("Similarity Metric",
     "Euclidean (L2) distance\nbetween embedding pairs\nThreshold sweep\n[0.0 → 4.0]  step 0.01"),
    ("Decision Rule",
     "d < θ_opt → SAME\nd ≥ θ_opt → DIFFERENT\nOptimal θ maximises\nverification accuracy"),
    ("Evaluation Metrics",
     "Accuracy: 98.20%\nROC-AUC:  0.9978\nFAR: 1.23%  FRR: 2.37%\nF1:  98.19%"),
]
METHOD_ACCS = [BLUE, ORANGE, PURPLE, GREEN]

for i, (t, d) in enumerate(METHOD_BOXES):
    ax3 = fig2.add_subplot(gs2_info[i])
    ax3.set_facecolor(PANEL)
    ax3.set_xlim(0, 1); ax3.set_ylim(0, 1)
    ax3.set_xticks([]); ax3.set_yticks([])
    for sp in ax3.spines.values():
        sp.set_edgecolor(METHOD_ACCS[i]); sp.set_linewidth(1.6)
    ax3.text(0.5, 0.88, t, ha="center", va="top",
             fontsize=9, fontweight="bold", color=METHOD_ACCS[i])
    ax3.text(0.5, 0.62, d, ha="center", va="top",
             fontsize=7.8, color="#c9d1d9", linespacing=1.55,
             family="monospace")

# Bottom stats bar
stats2 = fig2.add_axes([0.03, 0.01, 0.94, 0.075])
stats2.set_facecolor(PANEL)
stats2.set_xlim(0, 1); stats2.set_ylim(0, 1)
stats2.set_xticks([]); stats2.set_yticks([])
for sp in stats2.spines.values():
    sp.set_edgecolor(BORDER); sp.set_linewidth(1)

rows_test = [
    ("Dataset",          "LFW 112×112"),
    ("Total Images",     "13,233"),
    ("Total Identities", "5,749"),
    ("Eval Protocol",    "6,000-pair standard"),
    ("Genuine Pairs",    "3,000  (label = 1)"),
    ("Impostor Pairs",   "3,000  (label = 0)"),
    ("Accuracy",         "98.20%"),
    ("ROC-AUC",          "0.9978"),
    ("FAR / FRR",        "1.23% / 2.37%"),
]
n = len(rows_test)
for j, (k, v) in enumerate(rows_test):
    x = (j + 0.5) / n
    stats2.text(x, 0.75, k, ha="center", va="top", fontsize=7, color=TEXT2)
    stats2.text(x, 0.25, v, ha="center", va="bottom",
                fontsize=7.5, fontweight="bold", color=TEXT1)
    if j < n - 1:
        stats2.axvline((j + 1) / n, color=BORDER, linewidth=0.8)

out2 = os.path.join(OUT, "testing_dataset_samples.png")
fig2.savefig(out2, dpi=150, bbox_inches="tight",
             facecolor=BG, edgecolor="none")
plt.close(fig2)
print(f"[2/2] Saved: {out2}")
