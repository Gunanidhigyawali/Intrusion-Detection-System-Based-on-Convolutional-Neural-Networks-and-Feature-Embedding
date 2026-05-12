import os
import cv2
import time
import numpy as np
import onnxruntime as ort
from sklearn.metrics import accuracy_score, confusion_matrix
from numpy.linalg import norm

# -----------------------------
# CONFIG
# -----------------------------
PAIRS_FILE = "lfw_ann.txt"
ONNX_MODEL = "arcface_model.onnx"
IMG_SIZE = 112

# -----------------------------
# LOAD MODEL (FORCE CUDA)
# -----------------------------
providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']

session = ort.InferenceSession(
    ONNX_MODEL,
    providers=providers
)

print("Execution Providers:", session.get_providers())

input_name = session.get_inputs()[0].name

# -----------------------------
# OPTIONAL: CUDA SYNC (for accurate timing)
# -----------------------------
def cuda_sync():
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.synchronize()
    except:
        pass

# -----------------------------
# PREPROCESS
# -----------------------------
def preprocess(img_path):
    img = cv2.imread(img_path)

    if img is None:
        raise ValueError(f"Image not found: {img_path}")

    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    img = img / 255.0
    img = (img - 0.5) / 0.5

    img = np.transpose(img, (2, 0, 1))
    img = np.expand_dims(img, axis=0).astype(np.float32)

    return img

# -----------------------------
# GET EMBEDDING (GPU TIMING)
# -----------------------------
def get_embedding(img_path):
    img = preprocess(img_path)

    cuda_sync()
    start = time.perf_counter()

    emb = session.run(None, {input_name: img})[0][0]

    cuda_sync()
    end = time.perf_counter()

    emb = emb / np.linalg.norm(emb)

    return emb, (end - start)

# -----------------------------
# COSINE SIMILARITY
# -----------------------------
def cosine_similarity(a, b):
    return np.dot(a, b) / (norm(a) * norm(b))

# -----------------------------
# LOAD PAIRS
# -----------------------------
def load_pairs(pairs_file):
    pairs = []

    with open(pairs_file, 'r') as f:
        for line in f:
            parts = line.strip().split()

            if len(parts) != 3:
                continue

            label = int(parts[0])
            path1 = parts[1]
            path2 = parts[2]

            pairs.append((path1, path2, label))

    return pairs

# -----------------------------
# COMPUTE EMBEDDINGS (FAST)
# -----------------------------
def compute_embeddings(pairs):
    embedding_cache = {}
    total_time = 0
    total_images = 0

    # 🔥 WARM-UP (VERY IMPORTANT for GPU)
    print("Warming up GPU...")
    dummy = np.random.randn(1, 3, 112, 112).astype(np.float32)
    for _ in range(10):
        session.run(None, {input_name: dummy})

    print("Warm-up done.\n")

    for img1, img2, _ in pairs:
        for img in [img1, img2]:
            if img not in embedding_cache:
                try:
                    emb, t = get_embedding(img)
                    embedding_cache[img] = emb

                    total_time += t
                    total_images += 1

                except Exception as e:
                    print("Skipping:", e)

    avg_time = total_time / total_images
    fps = 1.0 / avg_time

    return embedding_cache, avg_time, total_time, fps

# -----------------------------
# EVALUATE
# -----------------------------
def evaluate(pairs, embeddings, threshold):
    y_true = []
    y_pred = []

    for img1, img2, label in pairs:
        if img1 not in embeddings or img2 not in embeddings:
            continue

        emb1 = embeddings[img1]
        emb2 = embeddings[img2]

        sim = cosine_similarity(emb1, emb2)
        pred = 1 if sim > threshold else 0

        y_true.append(label)
        y_pred.append(pred)

    acc = accuracy_score(y_true, y_pred)
    cm = confusion_matrix(y_true, y_pred)

    return acc, cm

# -----------------------------
# FIND BEST THRESHOLD
# -----------------------------
def find_best_threshold(pairs, embeddings):
    best_acc = 0
    best_th = 0

    for th in np.arange(0.0, 1.0, 0.01):
        acc, _ = evaluate(pairs, embeddings, th)

        if acc > best_acc:
            best_acc = acc
            best_th = th

    return best_th, best_acc

# -----------------------------
# MAIN
# -----------------------------
if __name__ == "__main__":

    print("Loading pairs...")
    pairs = load_pairs(PAIRS_FILE)
    print(f"Total pairs: {len(pairs)}")

    start_total = time.time()

    # Compute embeddings (GPU)
    print("\nComputing embeddings on GPU...")
    embeddings, avg_time, total_time, fps = compute_embeddings(pairs)

    print(f"Total unique images: {len(embeddings)}")

    # Find threshold
    print("\nFinding best threshold...")
    best_th, best_acc = find_best_threshold(pairs, embeddings)

    print(f"\nBest Threshold: {best_th:.2f}")
    print(f"Best Accuracy: {best_acc * 100:.2f}%")

    # Final evaluation
    print("\nFinal Evaluation...")
    acc, cm = evaluate(pairs, embeddings, best_th)

    end_total = time.time()

    print(f"\nFinal Accuracy: {acc * 100:.2f}%")
    print("\nConfusion Matrix:")
    print(cm)

    print("\n⏱️ Timing (GPU):")
    print(f"Average inference time per image: {avg_time * 1000:.2f} ms")
    print(f"FPS: {fps:.2f}")
    print(f"Total inference time (model only): {total_time:.2f} sec")
    print(f"Total evaluation time (full pipeline): {end_total - start_total:.2f} sec")