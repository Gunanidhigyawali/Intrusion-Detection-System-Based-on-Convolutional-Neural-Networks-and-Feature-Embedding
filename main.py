"""
Usage
-----
  python main.py train          # train on CASIA-WebFace
  python main.py eval           # evaluate best model on LFW
  python main.py train --epochs 30 --lr 0.05 --batch 64
  python main.py eval  --checkpoint training/best_model.pth
"""

import os
import argparse


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(
        description="ArcFace face-verification — train or evaluate"
    )
    sub = parser.add_subparsers(dest="mode", required=True)

    # ── train ──
    t = sub.add_parser("train", help="Train ArcFace on CASIA-WebFace")
    t.add_argument("--data",       default=None,  help="Override DATA_PATH in config")
    t.add_argument("--epochs",     type=int,   default=None, help="Override NUM_EPOCHS")
    t.add_argument("--lr",         type=float, default=None, help="Override LEARNING_RATE")
    t.add_argument("--batch",      type=int,   default=None, help="Override BATCH_SIZE")
    t.add_argument("--workers",    type=int,   default=None, help="Override NUM_WORKERS")
    t.add_argument("--out",        default=None, help="Override TRAINING_DIR")

    # ── eval ──
    e = sub.add_parser("eval", help="Evaluate model on LFW pairs")
    e.add_argument("--checkpoint", default=None, help="Override CHECKPOINT_PATH")
    e.add_argument("--lfw",        default=None, help="Override LFW_ROOT")
    e.add_argument("--ann",        default=None, help="Override ANN_PATH")
    e.add_argument("--out",        default=None, help="Override RESULTS_DIR")

    return parser.parse_args()


# ── Train ─────────────────────────────────────────────────────────────────────

def run_train(args):
    import config as cfg

    # apply CLI overrides before any training import reads them
    if args.data:    cfg.DATA_PATH    = args.data
    if args.epochs:  cfg.NUM_EPOCHS   = args.epochs
    if args.lr:      cfg.LEARNING_RATE = args.lr
    if args.batch:   cfg.BATCH_SIZE   = args.batch
    if args.workers: cfg.NUM_WORKERS  = args.workers
    if args.out:     cfg.TRAINING_DIR = args.out

    print(f"Device      : {cfg.DEVICE}")
    print(f"Data        : {cfg.DATA_PATH}")
    print(f"Epochs      : {cfg.NUM_EPOCHS}")
    print(f"LR          : {cfg.LEARNING_RATE}")
    print(f"Batch size  : {cfg.BATCH_SIZE}")
    print(f"Output dir  : {cfg.TRAINING_DIR}")

    from training.train import main as train_main
    train_main()


# ── Eval ──────────────────────────────────────────────────────────────────────

def run_eval(args):
    import config as cfg

    if args.checkpoint: cfg.CHECKPOINT_PATH = args.checkpoint
    if args.lfw:        cfg.LFW_ROOT        = args.lfw
    if args.ann:        cfg.ANN_PATH        = args.ann
    if args.out:        cfg.RESULTS_DIR     = args.out

    os.makedirs(cfg.RESULTS_DIR, exist_ok=True)

    print(f"Device      : {cfg.DEVICE}")
    print(f"Checkpoint  : {cfg.CHECKPOINT_PATH}")
    print(f"LFW root    : {cfg.LFW_ROOT}")
    print(f"Results dir : {cfg.RESULTS_DIR}")

    from model import ArcFaceModel
    from preprocessing import load_pairs
    from inference import load_model, extract_embeddings
    from evaluations import (
        compute_distances, find_best_threshold, distribution_check,
        plot_roc_curve, plot_confusion_matrix, plot_distance_distribution,
    )

    model = load_model(cfg.CHECKPOINT_PATH, ArcFaceModel)

    pairs = load_pairs(cfg.LFW_ROOT, cfg.ANN_PATH)
    if not pairs:
        raise ValueError("No pairs loaded — check LFW_ROOT / ANN_PATH in config.py.")

    emb1, emb2, labels = extract_embeddings(pairs, model)
    distances = compute_distances(emb1, emb2)
    distribution_check(distances, labels)

    best_th, best_acc = find_best_threshold(distances, labels)

    print("\n" + "=" * 50)
    print("LFW VERIFICATION RESULTS")
    print("=" * 50)
    print(f"  Best Threshold : {best_th:.4f}")
    print(f"  Accuracy       : {best_acc * 100:.2f}%")

    roc_auc = plot_roc_curve(
        labels, distances,
        save_path=os.path.join(cfg.RESULTS_DIR, "roc_curve.png"),
    )
    print(f"  AUC            : {roc_auc:.4f}")

    plot_confusion_matrix(
        labels, distances, best_th,
        save_path=os.path.join(cfg.RESULTS_DIR, "confusion_matrix.png"),
    )
    plot_distance_distribution(
        distances, labels,
        save_path=os.path.join(cfg.RESULTS_DIR, "distance_distribution.png"),
    )
    print(f"\n✅ All plots saved to '{cfg.RESULTS_DIR}/'")


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    args = parse_args()
    if args.mode == "train":
        run_train(args)
    else:
        run_eval(args)
