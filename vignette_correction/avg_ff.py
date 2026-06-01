import cv2
import numpy as np
import glob
import os

def create_average_flat_field(image_folder, output_path):
    """Averages all images in a folder to create a synthetic flat-field."""
    image_files = glob.glob(os.path.join(image_folder, '*.[tT][iI][fF]')) + \
                  glob.glob(os.path.join(image_folder, '*.[jJ][pP][gG]'))

    if not image_files:
        print("No images found in the folder.")
        return

    sample_img = cv2.imread(image_files[0], cv2.IMREAD_GRAYSCALE)
    accumulator = np.zeros_like(sample_img, dtype=np.float64)

    for i, file_path in enumerate(image_files):
        img = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)
        accumulator += img
        print(f"Averaging image {i+1}/{len(image_files)}", end='\r')

    average_image = (accumulator / len(image_files)).astype(np.uint8)
    cv2.imwrite(output_path, average_image)
    print(f"\nAverage flat-field saved to {output_path}")
