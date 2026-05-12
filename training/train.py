"""
Training entry point — called by main.py or directly:
  python main.py train [--epochs N] [--lr F] [--batch N] [--data PATH]
  python training/train.py

Resume behaviour
----------------
  - training_result.json does NOT exist → fresh run, epoch 1 uses ImageNet weights
  - training_result.json EXISTS          → resume from next epoch, load best_model.pth

Within a run
------------
  - Epoch 1       : ImageNet pretrained backbone
  - Epoch 2+      : load best_model.pth weights before each epoch
  - Optimizer/LR  : run continuously (not reset each epoch)
  - After crash   : scheduler is fast-forwarded to the correct epoch on resume

The saved best_model.pth is compatible with inference.py (strict=False drops
the arcface head weights).
"""

import os
import sys
import json
from datetime import datetime
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

import config
from model import ArcFaceModel
from training.dataset import split_dataset, FaceDataset, train_transform, val_transform
from training.loss import ArcMarginProduct


# ── Combined model ────────────────────────────────────────────────────────────

class TrainModel(nn.Module):
    def __init__(self, num_classes, pretrained=False):
        super().__init__()
        base = ArcFaceModel(embedding_size=config.EMBEDDING_SIZE,
                            pretrained=pretrained)
        self.backbone  = base.backbone
        self.embedding = base.embedding
        self.arcface   = ArcMarginProduct(config.EMBEDDING_SIZE, num_classes)

    def forward(self, x, labels=None):
        feat = self.backbone(x)
        emb  = self.embedding(feat)
        if labels is not None:
            return self.arcface(emb, labels), emb
        return emb


# ── Scheduler ─────────────────────────────────────────────────────────────────

def build_scheduler(optimizer, warmup_epochs, total_epochs):
    warmup = torch.optim.lr_scheduler.LinearLR(
        optimizer, start_factor=0.01, end_factor=1.0,
        total_iters=warmup_epochs,
    )
    cosine = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max(total_epochs - warmup_epochs, 1), eta_min=1e-6,
    )
    return torch.optim.lr_scheduler.SequentialLR(
        optimizer, schedulers=[warmup, cosine],
        milestones=[warmup_epochs],
    )


# ── Train / validate ──────────────────────────────────────────────────────────

def train_epoch(model, loader, criterion, optimizer, device, epoch):
    model.train()
    core = model.module if isinstance(model, nn.DataParallel) else model
    total_loss = correct = total = 0

    pbar = tqdm(loader, desc=f"Epoch {epoch+1:>3} [train]", leave=False)
    for images, labels in pbar:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        logits, emb = model(images, labels)
        loss = criterion(logits, labels)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        optimizer.step()

        with torch.no_grad():
            clean = F.linear(F.normalize(emb),
                             F.normalize(core.arcface.weight)) * core.arcface.s
            correct += clean.argmax(1).eq(labels).sum().item()
            total   += labels.size(0)

        total_loss += loss.item()
        pbar.set_postfix(loss=f"{loss.item():.4f}",
                         acc=f"{100.*correct/total:.2f}%",
                         lr=f"{optimizer.param_groups[0]['lr']:.2e}")

    return total_loss / len(loader), 100. * correct / total


def validate(model, loader, criterion, device):
    model.eval()
    core = model.module if isinstance(model, nn.DataParallel) else model
    total_loss = correct = total = 0

    with torch.no_grad():
        pbar = tqdm(loader, desc="              [val]  ", leave=False)
        for images, labels in pbar:
            images, labels = images.to(device), labels.to(device)
            emb    = model(images)
            cosine = F.linear(F.normalize(emb), F.normalize(core.arcface.weight))
            logits = cosine * core.arcface.s
            loss   = criterion(logits, labels)
            total_loss += loss.item()
            correct    += logits.argmax(1).eq(labels).sum().item()
            total      += labels.size(0)
            pbar.set_postfix(loss=f"{loss.item():.4f}",
                             acc=f"{100.*correct/total:.2f}%")

    return total_loss / len(loader), 100. * correct / total


# ── JSON result file ──────────────────────────────────────────────────────────

RESULT_FILENAME = "training_result.json"


def _result_path():
    return os.path.join(config.TRAINING_DIR, RESULT_FILENAME)


def load_run_state():
    """
    Read training_result.json and return:
        start_epoch   — next epoch to run (0 = fresh)
        best_val_acc  — best val accuracy so far
        best_epoch    — epoch number that achieved best_val_acc
    """
    path = _result_path()
    if not os.path.exists(path):
        return 0, 0.0, 0
    with open(path) as f:
        data = json.load(f)
    epochs_done = len(data.get("epochs", []))
    best        = data.get("best", {})
    return epochs_done, best.get("val_acc", 0.0), best.get("epoch", 0)


def _write_result(data):
    with open(_result_path(), "w") as f:
        json.dump(data, f, indent=2)


def init_result_file(cfg_snapshot):
    """Called only on a FRESH run (no existing JSON)."""
    _write_result({
        "run_started":  datetime.now().isoformat(timespec="seconds"),
        "last_updated": datetime.now().isoformat(timespec="seconds"),
        "config":       cfg_snapshot,
        "resume_log":   [],
        "epochs":       [],
        "best":         {},
    })


def mark_resume(start_epoch, best_val_acc):
    """Append a resume event to the existing JSON."""
    path = _result_path()
    with open(path) as f:
        data = json.load(f)
    data.setdefault("resume_log", []).append({
        "resumed_at":       datetime.now().isoformat(timespec="seconds"),
        "continue_from_epoch": start_epoch + 1,
        "best_val_acc_at_resume": round(best_val_acc, 4),
    })
    _write_result(data)


def log_epoch(epoch_record, best_record):
    """Append one epoch and update best. Called after every epoch."""
    path = _result_path()
    with open(path) as f:
        data = json.load(f)
    data["epochs"].append(epoch_record)
    data["best"]         = best_record
    data["last_updated"] = datetime.now().isoformat(timespec="seconds")
    _write_result(data)


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    os.makedirs(config.TRAINING_DIR, exist_ok=True)
    print(f"Device : {config.DEVICE}")

    # Single checkpoint file — full training state + best model weights
    last_ckpt = os.path.join(config.TRAINING_DIR, "last_epoch.pth")

    # ── Check for existing run ────────────────────────────────────────────────
    start_epoch, best_val_acc, best_epoch = load_run_state()
    is_resume = os.path.exists(last_ckpt)

    if start_epoch >= config.NUM_EPOCHS:
        print(f"All {config.NUM_EPOCHS} epochs already complete. "
              f"Increase NUM_EPOCHS in config.py to continue.")
        return

    # ── Data ─────────────────────────────────────────────────────────────────
    print("\n🔄 Splitting dataset...")
    train_samples, val_samples, num_classes = split_dataset(
        config.DATA_PATH, config.TRAIN_SPLIT, config.SEED
    )
    print(f"  ArcFace classes : {num_classes}")

    train_loader = DataLoader(
        FaceDataset(train_samples, train_transform),
        batch_size=config.BATCH_SIZE, shuffle=True,
        num_workers=config.NUM_WORKERS, pin_memory=True, drop_last=True,
    )
    val_loader = DataLoader(
        FaceDataset(val_samples, val_transform),
        batch_size=config.BATCH_SIZE, shuffle=False,
        num_workers=config.NUM_WORKERS, pin_memory=True,
    )

    # ── Config snapshot (written once to JSON on fresh run) ───────────────────
    cfg_snapshot = {
        "data_path":       config.DATA_PATH,
        "num_classes":     config.NUM_CLASSES,
        "train_split":     config.TRAIN_SPLIT,
        "val_size":        config.VAL_SIZE,
        "seed":            config.SEED,
        "batch_size":      config.BATCH_SIZE,
        "num_epochs":      config.NUM_EPOCHS,
        "learning_rate":   config.LEARNING_RATE,
        "momentum":        config.MOMENTUM,
        "weight_decay":    config.WEIGHT_DECAY,
        "label_smoothing": config.LABEL_SMOOTHING,
        "warmup_epochs":   config.WARMUP_EPOCHS,
        "arcface_s":       config.ARCFACE_S,
        "arcface_m":       config.ARCFACE_M,
        "embedding_size":  config.EMBEDDING_SIZE,
        "dropout":         config.DROPOUT,
        "img_size":        list(config.IMG_SIZE),
        "pretrained":      config.PRETRAINED,
    }

    # ── Model + optimiser + scheduler ─────────────────────────────────────────
    if is_resume:
        # Load last_epoch.pth — exact state from the last completed epoch
        last = torch.load(last_ckpt, map_location=config.DEVICE)
        num_classes  = last["num_classes"]          # use saved num_classes
        start_epoch  = last["epoch"] + 1            # next epoch to run
        best_val_acc = last["best_val_acc"]
        best_epoch   = last["best_epoch"]

        print(f"\nResuming from epoch {start_epoch + 1}/{config.NUM_EPOCHS}")
        print(f"  Optimizer LR  : {last['optimizer_state_dict']['param_groups'][0]['lr']:.2e}")
        print(f"  Best so far   : val_acc={best_val_acc:.2f}%  (epoch {best_epoch})")

        model = TrainModel(num_classes=num_classes, pretrained=False).to(config.DEVICE)
        core  = model.module if isinstance(model, nn.DataParallel) else model
        core.load_state_dict(last["model_state_dict"])

        optimizer = torch.optim.SGD(
            model.parameters(), lr=config.LEARNING_RATE,
            momentum=config.MOMENTUM, weight_decay=config.WEIGHT_DECAY,
        )
        optimizer.load_state_dict(last["optimizer_state_dict"])

        scheduler = build_scheduler(optimizer, config.WARMUP_EPOCHS, config.NUM_EPOCHS)
        scheduler.load_state_dict(last["scheduler_state_dict"])

        mark_resume(start_epoch, best_val_acc)

    else:
        print(f"\nFresh run — epoch 1 uses ImageNet pretrained backbone")
        model = TrainModel(num_classes=num_classes,
                           pretrained=config.PRETRAINED).to(config.DEVICE)
        optimizer = torch.optim.SGD(
            model.parameters(), lr=config.LEARNING_RATE,
            momentum=config.MOMENTUM, weight_decay=config.WEIGHT_DECAY,
        )
        scheduler = build_scheduler(optimizer, config.WARMUP_EPOCHS, config.NUM_EPOCHS)
        init_result_file(cfg_snapshot)

    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)
        print(f"Using {torch.cuda.device_count()} GPUs")

    criterion           = nn.CrossEntropyLoss(label_smoothing=config.LABEL_SMOOTHING)
    core                = model.module if isinstance(model, nn.DataParallel) else model
    best_model_weights  = core.state_dict()   # initialised to starting weights
    print(f"  Result file → {_result_path()}\n")

    # ── Training loop ─────────────────────────────────────────────────────────
    for epoch in range(start_epoch, config.NUM_EPOCHS):
        print(f"Epoch {epoch+1}/{config.NUM_EPOCHS}  "
              f"lr={optimizer.param_groups[0]['lr']:.2e}")

        # Epoch 2+ : swap in best model weights stored inside last_epoch.pth
        # Optimizer and scheduler keep their continuous state (LR, momentum)
        if epoch > 0 and os.path.exists(last_ckpt):
            last = torch.load(last_ckpt, map_location=config.DEVICE)
            core = model.module if isinstance(model, nn.DataParallel) else model
            core.load_state_dict(last["best_model_state_dict"])
            print(f"  ↩  Best weights loaded  (val_acc={last['best_val_acc']:.2f}%)")

        train_loss, train_acc = train_epoch(
            model, train_loader, criterion, optimizer, config.DEVICE, epoch
        )
        val_loss, val_acc = validate(model, val_loader, criterion, config.DEVICE)
        scheduler.step()

        lr  = optimizer.param_groups[0]["lr"]
        gap = round(train_acc - val_acc, 4)

        print(f"  Train  loss={train_loss:.4f}  acc={train_acc:.2f}%")
        print(f"  Val    loss={val_loss:.4f}  acc={val_acc:.2f}%")
        print(f"  Gap    {gap:+.2f}%\n")

        is_best = val_acc > best_val_acc
        if is_best:
            best_val_acc       = val_acc
            best_epoch         = epoch + 1
            best_model_weights = (model.module if isinstance(model, nn.DataParallel)
                                  else model).state_dict()
            print(f"  ✅ New best  (val_acc={val_acc:.2f}%)\n")
        else:
            print(f"  ↘  No improvement  (best={best_val_acc:.2f}% @ epoch {best_epoch})\n")

        # Single file: best model weights + full training state for resume
        core = model.module if isinstance(model, nn.DataParallel) else model
        torch.save({
            # ── for resume ────────────────────────────────────────────────
            "epoch":                epoch,        # last completed epoch index
            "model_state_dict":     core.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "num_classes":          num_classes,
            # ── best tracking ─────────────────────────────────────────────
            "best_model_state_dict": best_model_weights,
            "best_val_acc":          best_val_acc,
            "best_epoch":            best_epoch,
            "val_acc":               val_acc,     # current epoch val_acc
        }, last_ckpt)

        log_epoch(
            epoch_record={
                "epoch":      epoch + 1,
                "train_loss": round(train_loss, 6),
                "train_acc":  round(train_acc, 4),
                "val_loss":   round(val_loss, 6),
                "val_acc":    round(val_acc, 4),
                "lr":         round(lr, 8),
                "gap":        gap,
                "is_best":    is_best,
                "timestamp":  datetime.now().isoformat(timespec="seconds"),
            },
            best_record={
                "epoch":      best_epoch,
                "val_acc":    round(best_val_acc, 4),
                "checkpoint": last_ckpt,
            },
        )

        if (epoch + 1) % config.CHECKPOINT_EVERY == 0:
            ckpt_path = os.path.join(config.TRAINING_DIR,
                                     f"checkpoint_epoch_{epoch+1}.pth")
            torch.save({"epoch": epoch, "model_state_dict": core.state_dict(),
                        "val_acc": val_acc}, ckpt_path)
            print(f"  💾 Checkpoint → {ckpt_path}\n")

    print(f"{'='*60}")
    print(f"Training complete — best val acc: {best_val_acc:.2f}%  (epoch {best_epoch})")
    print(f"Checkpoint  → {last_ckpt}")
    print(f"Results     → {_result_path()}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
