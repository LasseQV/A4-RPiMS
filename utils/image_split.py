"""
Utility for splitting multiband images into per-band files.

Supported formats:
  e*   (Spectral Devices MSIS) — 2×2 grid
  cam* (Custom RPi camera)     — 1×4 grid; right two bands rotated 180°
"""

import cv2
from pathlib import Path


def split_and_save_multiband_image(image_path: Path, output_dir: Path) -> None:
    if not image_path.name.startswith(("e", "cam")):
        print(f"Skipping {image_path.name}: filename does not start with 'e' or 'cam', i.e. not one of our multiband formats")
        return
    elif image_path.name.startswith("e"):
        print(f"Splitting {image_path.name} as 2x2 grid")
        rows = 2
        cols = 2
    elif image_path.name.startswith("cam"):
        print(f"Splitting {image_path.name} as 1x4 grid")
        rows = 1
        cols = 4
    img = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        print(f"Could not read: {image_path}")
        return

    h, w = img.shape
    for r in range(rows):
        for c in range(cols):
            band = img[r * h // rows:(r + 1) * h // rows, c * w // cols:(c + 1) * w // cols]
            if image_path.name.startswith("cam") and c >= 2:
                band = cv2.rotate(band, cv2.ROTATE_180)
            if r == 0 and c == 0:
                # keep existing filename for leftmost band so downstream code can locate it by stem alone
                out_path = output_dir / f"{image_path.stem}{image_path.suffix}"
            else:
                out_path = output_dir / f"{image_path.stem}_band{r}_{c}{image_path.suffix}"
            cv2.imwrite(str(out_path), band)
            print(f"  → {out_path.name}: {w//cols}x{h//rows}")


def split_and_save_multiband_images(image_dir: Path, output_dir: Path) -> None:
    output_dir.mkdir(exist_ok=True)
    for image_path in image_dir.glob("*.jpg"):
        split_and_save_multiband_image(image_path, output_dir)
    for image_path in image_dir.glob("*.tif"):
        split_and_save_multiband_image(image_path, output_dir)
