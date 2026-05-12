"""
Central configuration — every tunable value lives here.
All other modules import from this file; nothing is hardcoded elsewhere.
"""

import torch

# ── Paths ─────────────────────────────────────────────────────────────────────

LFW_ROOT        = "/content/lfw/lfw"
ANN_PATH        = "/content/lfw/lfw/lfw_ann.txt"
CHECKPOINT_PATH = "training/last_epoch.pth"

RESULTS_DIR  = "results"
TRAINING_DIR = "training"
INFERENCE_DIR = "inference"

# ── Training — data ───────────────────────────────────────────────────────────

DATA_PATH   = r"D:\face_recognition\processed_dataset\casia_webface"   # CASIA-WebFace root (identity sub-dirs)
NUM_CLASSES = 10575                     # number of CASIA identities
TRAIN_SPLIT = 0.1                      # fraction used for training (stratified by identity)
VAL_SIZE    = None                      # None = use all 25% val images (no cap)
TEST_SIZE   = 4000                      # cap test images for speed
SEED        = 42                        # global RNG seed

# ── Training — loader ─────────────────────────────────────────────────────────

BATCH_SIZE   = 128
NUM_WORKERS  = 4

# ── Training — model init ─────────────────────────────────────────────────────

PRETRAINED = True         # epoch 1 starts from ImageNet weights

# ── Training — optimiser ──────────────────────────────────────────────────────

NUM_EPOCHS       = 50
LEARNING_RATE    = 0.1
MOMENTUM         = 0.9
WEIGHT_DECAY     = 5e-4
LABEL_SMOOTHING  = 0.1
WARMUP_EPOCHS    = 3      # linear LR warm-up before cosine decay
CHECKPOINT_EVERY = 1      # save a periodic checkpoint every N epochs

# ── Training — ArcFace margin loss ────────────────────────────────────────────

ARCFACE_S = 30.0   # scale (hypersphere radius)
ARCFACE_M = 0.40   # additive angular margin

# ── Device ────────────────────────────────────────────────────────────────────

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ── Preprocessing ─────────────────────────────────────────────────────────────

IMG_SIZE   = (112, 112)   # (width, height) fed to the model
PIXEL_MEAN = 127.5        # subtracted from every channel
PIXEL_STD  = 0.0078125    # = 1/128, scales to [-1, 1]

# ── Model ─────────────────────────────────────────────────────────────────────

EMBEDDING_SIZE = 512
DROPOUT        = 0.4

# ── Evaluation / threshold sweep ──────────────────────────────────────────────

THRESHOLD_MIN  = 0.0
THRESHOLD_MAX  = 4.0
THRESHOLD_STEP = 0.01

# ── Plot appearance ───────────────────────────────────────────────────────────

PLOT_DPI          = 150
FIG_SIZE_ROC      = (7, 6)
FIG_SIZE_CM       = (6, 5)
FIG_SIZE_DIST     = (9, 5)
HIST_BINS         = 60
FONT_SIZE_LABEL   = 12
FONT_SIZE_TITLE   = 13
FONT_SIZE_LEGEND  = 11
