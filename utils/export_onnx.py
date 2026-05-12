"""
Export ArcFace model to ONNX.

Usage:
    python export_onnx.py
    python export_onnx.py --checkpoint D:\final\intrusion_detection_system\Intrusion_detection_training_checkpoint_epoch_50.pth
    python export_onnx.py --output my_model.onnx
"""

import argparse
import torch
import torch.nn.functional as F
from model import ArcFaceModel
from config import EMBEDDING_SIZE, IMG_SIZE, DEVICE


# ── Inference wrapper (normalized embeddings, no Dropout) ────────────────────

class ArcFaceONNX(ArcFaceModel):
    """Wraps ArcFaceModel: eval mode + L2-normalised output for ONNX export."""
    def forward(self, x):
        emb = super().forward(x)
        return F.normalize(emb, p=2, dim=1)


def export(checkpoint_path, output_path):
    print(f"Checkpoint : {checkpoint_path}")
    print(f"Output     : {output_path}")
    print(f"Device     : {DEVICE}")

    # ── Load model ────────────────────────────────────────────────────────────
    ckpt = torch.load(checkpoint_path, map_location=DEVICE)

    # Support both last_epoch.pth (best_model_state_dict) and plain checkpoints
    if "best_model_state_dict" in ckpt:
        state_dict = ckpt["best_model_state_dict"]
        print(f"Using best_model_state_dict  (best val_acc={ckpt.get('best_val_acc', '?')})")
    elif "model_state_dict" in ckpt:
        state_dict = ckpt["model_state_dict"]
    else:
        state_dict = ckpt   # raw state dict

    # Strip DataParallel / TrainModel prefixes if present
    clean = {}
    for k, v in state_dict.items():
        k = k.replace("module.", "")          # DataParallel
        if k.startswith("arcface."):          # TrainModel head — skip
            continue
        clean[k] = v

    model = ArcFaceONNX(embedding_size=EMBEDDING_SIZE)
    model.load_state_dict(clean, strict=False)
    model.to(DEVICE)
    model.eval()
    print("Model loaded.")

    # ── Dummy input ───────────────────────────────────────────────────────────
    h, w = IMG_SIZE
    dummy = torch.randn(1, 3, h, w, device=DEVICE)

    # ── Export ────────────────────────────────────────────────────────────────
    torch.onnx.export(
        model,
        dummy,
        output_path,
        input_names=["input"],
        output_names=["embedding"],
        dynamic_axes={
            "input":     {0: "batch_size"},
            "embedding": {0: "batch_size"},
        },
        opset_version=17,
        export_params=True,
    )
    print(f"\n✅ ONNX model saved → {output_path}")

    # ── Quick verify ──────────────────────────────────────────────────────────
    try:
        import onnx
        m = onnx.load(output_path)
        onnx.checker.check_model(m)
        print("✅ ONNX graph check passed")
        inp  = m.graph.input[0].type.tensor_type.shape
        out  = m.graph.output[0].type.tensor_type.shape
        print(f"   Input  shape : {[d.dim_param or d.dim_value for d in inp.dim]}")
        print(f"   Output shape : {[d.dim_param or d.dim_value for d in out.dim]}")
    except ImportError:
        print("(install 'onnx' package to enable graph verification)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--checkpoint",
        default=r"D:\final\intrusion_detection_system\Intrusion_detection_training_checkpoint_epoch_50.pth",
    )
    parser.add_argument(
        "--output",
        default="arcface_model.onnx",
    )
    args = parser.parse_args()
    export(args.checkpoint, args.output)
