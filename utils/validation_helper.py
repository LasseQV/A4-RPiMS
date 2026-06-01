#!/usr/bin/env python3
"""
validation_helper.py — visualise YOLO predictions on the leftmost MS band for manual review.
"""

import argparse
import cv2
import numpy as np
from pathlib import Path
from typing import List, Tuple, Dict
import json


class LabelValidator:
    """Helper class to visualize and validate YOLO predictions"""
    
    def __init__(self, images_dir: str, labels_dir: str, output_dir: str,
                 band_width: int = 1280, image_height: int = 800):
        self.images_dir = Path(images_dir)
        self.labels_dir = Path(labels_dir)
        self.output_dir = Path(output_dir)
        self.band_width = band_width
        self.image_height = image_height
        
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.colors = [
            (0, 255, 0),    # Green - class 0
            (255, 0, 0),    # Blue - class 1
            (0, 0, 255),    # Red - class 2
            (255, 255, 0),  # Cyan - class 3
            (255, 0, 255),  # Magenta - class 4
            (0, 255, 255),  # Yellow - class 5
        ]
    
    def parse_yolo_label(self, label_path: Path, img_width: int, img_height: int) -> List[Dict]:
        detections = []
        
        if not label_path.exists():
            return detections
        
        with open(label_path, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 5:
                    class_id = int(parts[0])
                    center_x = float(parts[1]) * img_width
                    center_y = float(parts[2]) * img_height
                    width = float(parts[3]) * img_width
                    height = float(parts[4]) * img_height
                    
                    x1 = int(center_x - width / 2)
                    y1 = int(center_y - height / 2)
                    x2 = int(center_x + width / 2)
                    y2 = int(center_y + height / 2)
                    
                    detections.append({
                        'class_id': class_id,
                        'bbox': (x1, y1, x2, y2),
                        'center': (int(center_x), int(center_y))
                    })
        
        return detections
    
    def draw_detections(self, image: np.ndarray, detections: List[Dict]) -> np.ndarray:
        vis_image = image.copy()
        
        for det in detections:
            class_id = det['class_id']
            x1, y1, x2, y2 = det['bbox']
            
            color = self.colors[class_id % len(self.colors)]
            cv2.rectangle(vis_image, (x1, y1), (x2, y2), color, 2)
            label = f"Class {class_id}"
            (label_w, label_h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            cv2.rectangle(vis_image, (x1, y1 - label_h - 4), (x1 + label_w, y1), color, -1)
            cv2.putText(vis_image, label, (x1, y1 - 2),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            cv2.circle(vis_image, det['center'], 3, color, -1)
        
        return vis_image
    
    def process_image(self, image_path: Path):
        full_image = cv2.imread(str(image_path))
        if full_image is None:
            print(f"ERROR: Could not load {image_path}")
            return

        band_image = full_image[:, :self.band_width, :]
        label_path = self.labels_dir / f"{image_path.stem}.txt"
        detections = self.parse_yolo_label(label_path, self.band_width, self.image_height)
        vis_image = self.draw_detections(band_image, detections)
        output_path = self.output_dir / f"{image_path.stem}_visualized.jpg"
        cv2.imwrite(str(output_path), vis_image)
        
        return len(detections)
    
    def process_all(self):
        image_extensions = {'.jpg', '.jpeg', '.png', '.bmp'}
        image_files = [f for f in self.images_dir.iterdir() 
                      if f.suffix.lower() in image_extensions]
        
        if not image_files:
            print(f"No images found in {self.images_dir}")
            return
        
        print(f"Processing {len(image_files)} images...")
        
        total_detections = 0
        stats = []
        
        for idx, image_path in enumerate(image_files, 1):
            num_dets = self.process_image(image_path)
            if num_dets is None:
                continue
            total_detections += num_dets
            
            stats.append({
                'image': image_path.name,
                'detections': num_dets
            })
            
            print(f"[{idx}/{len(image_files)}] {image_path.name}: {num_dets} detections")
        
        summary = {
            'total_images': len(image_files),
            'total_detections': total_detections,
            'avg_detections_per_image': total_detections / len(image_files),
            'image_stats': stats
        }
        
        summary_path = self.output_dir / 'validation_summary.json'
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2)
        
        print(f"\n✓ Visualizations saved to: {self.output_dir}")
        print(f"  Total detections: {total_detections}")
        print(f"  Average per image: {total_detections / len(image_files):.2f}")
        print(f"  Summary saved to: {summary_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Visualize YOLO predictions for manual validation"
    )
    
    parser.add_argument('--images', '-i', type=str, required=True,
                       help='Directory containing original images')
    parser.add_argument('--labels', '-l', type=str, required=True,
                       help='Directory containing YOLO label files')
    parser.add_argument('--output', '-o', type=str, required=True,
                       help='Output directory for visualizations')
    parser.add_argument('--band-width', type=int, default=1280,
                       help='Width of the leftmost band (default: 1280)')
    parser.add_argument('--height', type=int, default=800,
                       help='Image height (default: 800)')
    
    args = parser.parse_args()
    
    validator = LabelValidator(
        images_dir=args.images,
        labels_dir=args.labels,
        output_dir=args.output,
        band_width=args.band_width,
        image_height=args.height
    )
    
    validator.process_all()


if __name__ == "__main__":
    main()
