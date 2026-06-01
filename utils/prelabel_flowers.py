"""
prelabel_flowers.py

Prelabeling script for bright, small flower detection in multispectral imagery
using exposure normalization, median blur for hot-pixel rejection, and
White Top-Hat morphological filtering for soil/glint removal.

Usage:
    python prelabel_flowers.py /path/to/image/directory
    python prelabel_flowers.py /path/to/dir --validate
    python prelabel_flowers.py /path/to/dir --tune
    python prelabel_flowers.py /path/to/dir --output_dir /path/to/output
"""

import argparse
import itertools
import random
import re
import sys
from pathlib import Path

import cv2
import numpy as np

# ── Paths ─────────────────────────────────────────────────────────────────────

GT_IMAGE_PATH  = "/Users/llund/A4-RPiMS/test_data/valid_flowers/camMS-1778177340686989_visualized.jpg"
GT_LABEL_PATH  = "/Users/llund/A4-RPiMS/test_data/valid_flowers/labels/train/camMS-1778177340686989_visualized.txt"
ZERO_DET_DIR   = "/Users/llund/A4-RPiMS/zero-det"

# ── Detection params ───────────────────────────────────────────────────────────

PARAMS = {
    "input_width":      5120,
    "input_height":     800,
    "band_width":       1280,   # leftmost band only
    "overexposure_threshold": 240,

    # Core Detection
    "thresh_value":     10,     # Global cutoff on the Top-Hat response
    "min_scale_floor":  110.00,  # Prevents noise stretching on dark frames
    "min_area_px":      8,      # Minimum blob area (lowered for tiny flowers)
    "max_area_px":      200,    # Maximum blob area
    "tophat_size":      21,     # Kernel size for White Top-Hat morphology

    # Output
    "class_id":         0,
    "bbox_scale":       2.2,
    "gt_max_bbox_w":    16.32,   # Auto-populated from GT during startup/tuning
    "gt_max_bbox_h":    14.15,   # Auto-populated from GT during startup/tuning
}

GT_SIZE_MARGIN = 0.20

# ── Core pipeline ─────────────────────────────────────────────────────────────

def derive_gt_size_bounds():
    gt_path = Path(GT_LABEL_PATH)
    if not gt_path.exists():
        return

    img_h, img_w = PARAMS["input_height"], PARAMS["band_width"]
    max_w = max_h = 0.0

    with open(gt_path) as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 5:
                _, _, _, bw, bh = map(float, parts)
                max_w = max(max_w, bw * img_w)
                max_h = max(max_h, bh * img_h)

    if max_w > 0 and max_h > 0:
        PARAMS["gt_max_bbox_w"] = max_w * (1 + GT_SIZE_MARGIN)
        PARAMS["gt_max_bbox_h"] = max_h * (1 + GT_SIZE_MARGIN)


def load_band(image_path: Path):
    img = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(f"Could not read: {image_path}")
    raw = img[:, :PARAMS["band_width"]]
    return raw, float(np.percentile(raw, 99))


def detect_flowers(band: np.ndarray, p99: float):
    scale = max(p99, float(PARAMS.get("min_scale_floor", 130.0)))
    normed = np.clip(band.astype(np.float32) / scale * 255.0, 0, 255).astype(np.uint8)

    k_size = int(PARAMS.get("tophat_size", 15))
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k_size, k_size))
    tophat = cv2.morphologyEx(normed, cv2.MORPH_TOPHAT, kernel)

    _, binary = cv2.threshold(tophat, int(PARAMS["thresh_value"]), 255, cv2.THRESH_BINARY)
    n_labels, _, stats, centroids = cv2.connectedComponentsWithStats(binary, connectivity=8)

    blobs = []
    for i in range(1, n_labels):
        area = int(stats[i, cv2.CC_STAT_AREA])

        if not (PARAMS["min_area_px"] <= area <= PARAMS["max_area_px"]):
            continue

        w, h = stats[i, cv2.CC_STAT_WIDTH], stats[i, cv2.CC_STAT_HEIGHT]
        aspect_ratio = max(w, h) / max(min(w, h), 1)
        if aspect_ratio > 2.5:  # reject streaks/glint
            continue

        cx, cy = centroids[i]
        r = np.sqrt(area / np.pi)

        if PARAMS["gt_max_bbox_w"] is not None:
            out_dim = r * 2 * PARAMS["bbox_scale"]
            if out_dim > PARAMS["gt_max_bbox_w"] or out_dim > PARAMS["gt_max_bbox_h"]:
                continue

        blobs.append((cy, cx, r))
    
    return blobs


def blobs_to_yolo(blobs, img_h, img_w):
    lines = []
    scale = PARAMS["bbox_scale"]
    cls = PARAMS["class_id"]

    for y, x, r in blobs:
        bw, bh = r * 2 * scale, r * 2 * scale
        cx, cy = max(0.0, min(1.0, x / img_w)), max(0.0, min(1.0, y / img_h))
        nw, nh = min(bw / img_w, 1.0), min(bh / img_h, 1.0)
        lines.append(f"{cls} {cx:.6f} {cy:.6f} {nw:.6f} {nh:.6f}")
    return lines


# ── Validation and tuning ─────────────────────────────────────────────────────

def load_yolo_labels(label_path: str, img_h: int, img_w: int):
    boxes = []
    with open(label_path) as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 5:
                _, cx, cy, bw, bh = map(float, parts)
                boxes.append((cx * img_w, cy * img_h, bw * img_w, bh * img_h))
    return boxes


def match_detections(gt_boxes, pred_blobs):
    matched_gt = set()
    tp = fp = 0
    for y, x, r in pred_blobs:
        matched = False
        for i, (gcx, gcy, gbw, gbh) in enumerate(gt_boxes):
            if np.hypot(x - gcx, y - gcy) < (gbw / 2) and i not in matched_gt:
                matched_gt.add(i)
                matched = True
                break
        if matched: tp += 1
        else: fp += 1
    fn = len(gt_boxes) - len(matched_gt)
    return tp, fp, fn


def auto_tune(target_dir: Path, fp_weight=0.5):
    print(f"\nRunning Top-Hat Grid Search with Wild Data Sanity Check...")
    
    gt_band, gt_p99 = load_band(Path(GT_IMAGE_PATH))
    h, w = gt_band.shape
    gt_boxes = load_yolo_labels(str(GT_LABEL_PATH), h, w)
    expected_flowers = max(len(gt_boxes), 1)

    zero_dir = Path(ZERO_DET_DIR)
    zero_pairs = [load_band(p) for p in zero_dir.iterdir() if p.suffix.lower() in {".jpg", ".tif"}] if zero_dir.exists() else []

    wild_paths = [p for p in target_dir.iterdir() if p.suffix.lower() in {".png", ".jpg", ".jpeg", ".tif", ".tiff"}]
    sanity_pairs = []
    for p in random.sample(wild_paths, min(25, len(wild_paths))):
        try:
            sanity_pairs.append(load_band(p))
        except Exception:
            pass
            
    print(f"Sampled {len(sanity_pairs)} random images from target directory for noise checking.\n")

    best_score = -np.inf
    best_params = {
        "thresh_value": PARAMS["thresh_value"],
        "min_scale_floor": PARAMS.get("min_scale_floor", 130.0),
        "min_area_px": PARAMS["min_area_px"],
        "tophat_size": PARAMS.get("tophat_size", 15)
    }

    grid_thresh = range(10, 50, 10)
    grid_scale_floor = [110.0, 130.0, 150.0]   
    grid_min_area = [3, 5, 8]                  # Capture 2x2 pixel blobs
    grid_tophat = [9, 15, 21]                  # Sweep kernel sizes

    for thr, scale_fl, min_a, tophat_sz in itertools.product(grid_thresh, grid_scale_floor, grid_min_area, grid_tophat):
        PARAMS["thresh_value"] = thr
        PARAMS["min_scale_floor"] = scale_fl
        PARAMS["min_area_px"] = min_a
        PARAMS["tophat_size"] = tophat_sz
        
        # Score the GT Image
        filtered = detect_flowers(gt_band, gt_p99)
        tp, fp_gt, fn = match_detections(gt_boxes, filtered)
        total_gt_dets = len(filtered)
        
        precision = tp / (tp + fp_gt) if (tp + fp_gt) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        
            f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0

        zero_fp = sum(len(detect_flowers(zb, zp99)) for zb, zp99 in zero_pairs if zp99 < PARAMS["overexposure_threshold"])

        sanity_dets = 0
        valid_sanity_imgs = 0
        for sb, sp99 in sanity_pairs:
            if sp99 < PARAMS["overexposure_threshold"]:
                sanity_dets += len(detect_flowers(sb, sp99))
                valid_sanity_imgs += 1
                
        sanity_avg = sanity_dets / max(valid_sanity_imgs, 1)

        score = f1
        score -= (fp_weight * (zero_fp / 1000.0))

        wild_noise_penalty = 0.0
        if sanity_avg > 150:
            wild_noise_penalty = 10.0
        elif sanity_avg > 80:
            wild_noise_penalty = (sanity_avg - 80) * 0.01
        score -= wild_noise_penalty

        if score > -2.0 and f1 > 0.01:
            print(f"  Thr={thr:<3} Floor={scale_fl:<5} MinPx={min_a:<2} Hat={tophat_sz:<2} | F1={f1:.3f} P={precision:.3f} R={recall:.3f} WildAvg={sanity_avg:<4.0f} | Score={score:+.3f}")

        if score > best_score:
            best_score = score
            best_params = {
                "thresh_value": thr, 
                "min_scale_floor": scale_fl, 
                "min_area_px": min_a,
                "tophat_size": tophat_sz
            }

    best_params["gt_max_bbox_w"] = PARAMS.get("gt_max_bbox_w")
    best_params["gt_max_bbox_h"] = PARAMS.get("gt_max_bbox_h")

    print("\n" + "="*50)
    print(f"BEST PARAMETERS FOUND (Score: {best_score:.3f})")
    for k, v in best_params.items():
        if isinstance(v, float):
            print(f"  {k}: {v:.2f}")
        else:
            print(f"  {k}: {v}")
    print("="*50)
    
    script_path = Path(__file__).resolve()
    updated = script_path.read_text()
    
    for key, val in best_params.items():
        if val is None:
            continue
            
        # Regex looks for the key, a colon, and then EITHER a number OR the word 'None'
        pattern = rf'("{key}"\s*:\s*)([0-9]*\.?[0-9]+|None)'
        
        if isinstance(val, float):
            updated, _ = re.subn(pattern, rf'\g<1>{val:.2f}', updated)
        else:
            updated, _ = re.subn(pattern, rf'\g<1>{val}', updated)
            
    if updated != script_path.read_text():
        script_path.write_text(updated)
        print("\nSuccessfully updated PARAMS in script.")
    else:
        print("\nNo parameter changes needed script rewrite.")


def validate():
    gt_band, gt_p99 = load_band(Path(GT_IMAGE_PATH))
    gt_boxes = load_yolo_labels(str(GT_LABEL_PATH), gt_band.shape[0], gt_band.shape[1])
    blobs = detect_flowers(gt_band, gt_p99)
    tp, fp, fn = match_detections(gt_boxes, blobs)
    
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    print(f"GT Validation | TP: {tp}, FP: {fp}, FN: {fn} | Precision: {precision:.2f}, Recall: {recall:.2f}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("image_dir", help="Directory of images to prelabel")
    parser.add_argument("--validate", action="store_true")
    parser.add_argument("--tune", action="store_true")
    parser.add_argument("--output_dir", default=None, help="Where to write label files")
    args = parser.parse_args()

    derive_gt_size_bounds()
    img_dir = Path(args.image_dir)

    if args.tune:
        auto_tune(img_dir)
        return

    if args.validate:
        validate()

    output_dir = Path(args.output_dir) if args.output_dir else img_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    images = [p for p in img_dir.iterdir() if p.suffix.lower() in {".png", ".jpg", ".jpeg", ".tif", ".tiff"}]

    for img_path in images:
        try:
            band, p99 = load_band(img_path)
        except Exception as e:
            print(f"Skipping {img_path.name}: Corrupted or unreadable image ({e})")
            continue
            
        lbl_path = output_dir / (img_path.stem + ".txt")
        
        if p99 >= PARAMS["overexposure_threshold"]:
            lbl_path.write_text("")
            continue

        blobs = detect_flowers(band, p99)
        lbl_path.write_text("\n".join(blobs_to_yolo(blobs, band.shape[0], band.shape[1])))
        print(f"Processed {img_path.name}: {len(blobs)} detections")

if __name__ == "__main__":
    main()