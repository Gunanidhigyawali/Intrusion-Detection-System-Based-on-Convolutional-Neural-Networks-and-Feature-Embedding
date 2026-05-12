import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    roc_curve,
    auc,
    confusion_matrix,
    classification_report,
)

from config import (
    THRESHOLD_MIN, THRESHOLD_MAX, THRESHOLD_STEP,
    PLOT_DPI,
    FIG_SIZE_ROC, FIG_SIZE_CM, FIG_SIZE_DIST,
    HIST_BINS,
    FONT_SIZE_LABEL, FONT_SIZE_TITLE, FONT_SIZE_LEGEND,
)


def compute_distances(embeddings1, embeddings2):
    return np.sum((embeddings1 - embeddings2) ** 2, axis=1)


def find_best_threshold(distances, labels):
    thresholds = np.arange(THRESHOLD_MIN, THRESHOLD_MAX, THRESHOLD_STEP)
    best_acc, best_th = 0, 0
    for th in thresholds:
        acc = np.mean((distances < th).astype(int) == labels)
        if acc > best_acc:
            best_acc, best_th = acc, th
    return best_th, best_acc


def distribution_check(distances, labels):
    same = distances[labels == 1]
    diff = distances[labels == 0]
    print("\n🔍 DISTRIBUTION CHECK")
    print(f"  Same person  — mean: {same.mean():.4f}  std: {same.std():.4f}")
    print(f"  Diff person  — mean: {diff.mean():.4f}  std: {diff.std():.4f}")
    return same, diff


# ── ROC Curve ────────────────────────────────────────────────────────────────

def plot_roc_curve(labels, distances, save_path):
    scores = -distances   # higher = more similar
    fpr, tpr, _ = roc_curve(labels, scores)
    roc_auc = auc(fpr, tpr)

    fig, ax = plt.subplots(figsize=FIG_SIZE_ROC)
    ax.plot(fpr, tpr, color="darkorange", lw=2,
            label=f"AUC = {roc_auc:.4f}")
    ax.plot([0, 1], [0, 1], color="navy", lw=1.5, linestyle="--",
            label="Random classifier")
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel("False Positive Rate", fontsize=FONT_SIZE_LABEL)
    ax.set_ylabel("True Positive Rate", fontsize=FONT_SIZE_LABEL)
    ax.set_title("ROC Curve — LFW Face Verification", fontsize=FONT_SIZE_TITLE)
    ax.legend(loc="lower right", fontsize=FONT_SIZE_LEGEND)
    fig.tight_layout()
    fig.savefig(save_path, dpi=PLOT_DPI)
    plt.close(fig)
    print(f"✅ ROC curve saved  → {save_path}  (AUC={roc_auc:.4f})")
    return roc_auc


# ── Confusion Matrix ──────────────────────────────────────────────────────────

def plot_confusion_matrix(labels, distances, threshold, save_path):
    preds = (distances < threshold).astype(int)
    cm = confusion_matrix(labels, preds)

    fig, ax = plt.subplots(figsize=FIG_SIZE_CM)
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues", ax=ax,
        xticklabels=["Different (0)", "Same (1)"],
        yticklabels=["Different (0)", "Same (1)"],
        linewidths=0.5, linecolor="gray",
    )
    ax.set_ylabel("True Label", fontsize=FONT_SIZE_LABEL)
    ax.set_xlabel("Predicted Label", fontsize=FONT_SIZE_LABEL)
    ax.set_title(f"Confusion Matrix  (threshold = {threshold:.4f})",
                 fontsize=FONT_SIZE_TITLE)
    fig.tight_layout()
    fig.savefig(save_path, dpi=PLOT_DPI)
    plt.close(fig)
    print(f"\n✅ Confusion matrix saved → {save_path}")
    print("\n📋 Classification Report:")
    print(classification_report(labels, preds,
                                target_names=["Different", "Same"]))
    return cm


# ── Distance Distribution ─────────────────────────────────────────────────────

def plot_distance_distribution(distances, labels, save_path):
    same = distances[labels == 1]
    diff = distances[labels == 0]

    fig, ax = plt.subplots(figsize=FIG_SIZE_DIST)
    ax.hist(same, bins=HIST_BINS, alpha=0.7, color="steelblue",
            label="Same person")
    ax.hist(diff, bins=HIST_BINS, alpha=0.7, color="tomato",
            label="Different person")
    ax.set_xlabel("L2 Distance", fontsize=FONT_SIZE_LABEL)
    ax.set_ylabel("Count", fontsize=FONT_SIZE_LABEL)
    ax.set_title("L2 Distance Distribution — Same vs Different",
                 fontsize=FONT_SIZE_TITLE)
    ax.legend(fontsize=FONT_SIZE_LEGEND)
    fig.tight_layout()
    fig.savefig(save_path, dpi=PLOT_DPI)
    plt.close(fig)
    print(f"✅ Distance distribution saved → {save_path}")
