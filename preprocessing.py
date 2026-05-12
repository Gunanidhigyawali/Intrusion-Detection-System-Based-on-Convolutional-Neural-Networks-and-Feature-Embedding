import os
import cv2
import numpy as np

from config import IMG_SIZE, PIXEL_MEAN, PIXEL_STD


def preprocess(img):
    img = cv2.resize(img, IMG_SIZE)
    img = img.astype(np.float32)
    img -= PIXEL_MEAN
    img *= PIXEL_STD
    img = np.transpose(img, (2, 0, 1))
    return img


def load_pairs(lfw_root, ann_path):
    pairs = []
    with open(ann_path, "r") as f:
        for i, line in enumerate(f):
            parts = line.strip().split()
            if len(parts) != 3:
                continue
            try:
                label = int(parts[0])
                path1 = os.path.join(lfw_root, parts[1])
                path2 = os.path.join(lfw_root, parts[2])
                if i < 3:
                    print(f"DEBUG: {path1} | {path2}")
                pairs.append((path1, path2, label))
            except Exception as e:
                print(f"Skipping line {i}: {e}")
    print(f"\n✅ FINAL PAIRS LOADED: {len(pairs)}")
    return pairs
