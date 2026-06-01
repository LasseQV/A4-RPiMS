"""
detect_labels.py

Run yolo26n detection on a directory of images and write YOLO-format label files.

Usage:
    python detect_labels.py /path/to/images
    python detect_labels.py /path/to/images --model utils/yolo26n_new_best.pt
    python detect_labels.py /path/to/images --output_dir /path/to/labels --conf 0.25
"""

import argparse
from pathlib import Path

import cv2
from ultralytics import YOLO

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".tif", ".tiff"}
DEFAULT_MODEL = Path(__file__).parent / "utils" / "yolo26n_new_best.pt"


def detect(image_dir: Path, model_path: Path, output_dir: Path, conf: float):
    output_dir.mkdir(parents=True, exist_ok=True)
    model = YOLO(str(model_path))

    images = sorted(p for p in image_dir.iterdir() if p.suffix.lower() in IMAGE_EXTS)
    if not images:
        print(f"No images found in {image_dir}")
        return

    print(f"Running detection on {len(images)} images → {output_dir}")

    for img_path in images:
        img = cv2.imread(str(img_path))
        if img is None:
            print(f"  Skipping {img_path.name}: unreadable")
            continue
        crop = img[:, : img.shape[1] // 4]  # left quarter (one multispectral band)

        results = model.predict(crop, conf=conf, verbose=False)
        label_path = output_dir / (img_path.stem + ".txt")

        lines = []
        for box in results[0].boxes:
            cls = int(box.cls)
            cx, cy, w, h = box.xywhn[0].tolist()  # normalized xywh
            lines.append(f"{cls} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}")

        label_path.write_text("\n".join(lines))
        print(f"  {img_path.name}: {len(lines)} detection(s)")

    print(f"Done. Labels written to {output_dir}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("image_dir", help="Directory of input images")
    parser.add_argument("--model", default=str(DEFAULT_MODEL), help="Path to .pt model file")
    parser.add_argument("--output_dir", default=None, help="Where to write label .txt files (default: <image_dir>/labels)")
    parser.add_argument("--conf", type=float, default=0.25, help="Confidence threshold (default: 0.25)")
    args = parser.parse_args()

    image_dir = Path(args.image_dir)
    output_dir = Path(args.output_dir) if args.output_dir else image_dir / "labels"

    detect(image_dir, Path(args.model), output_dir, args.conf)


if __name__ == "__main__":
    main()
