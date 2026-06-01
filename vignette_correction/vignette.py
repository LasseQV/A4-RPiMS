import cv2
import numpy as np
import os
import glob
import tempfile
import shutil


def correct_vignette_grayscale(image, k):
    """Corrects vignetting in a single-channel (grayscale) image."""
    if image.ndim != 2:
        raise ValueError(f"Input must be a 2D grayscale image, but got shape {image.shape}")
    height, width = image.shape
    x = np.linspace(-width / 2, width / 2, width)
    y = np.linspace(-height / 2, height / 2, height)
    xx, yy = np.meshgrid(x, y)
    radius = np.sqrt(xx**2 + yy**2)
    max_radius = np.sqrt((width / 2)**2 + (height / 2)**2)
    norm_radius = radius / max_radius
    correction = 1 + k * norm_radius**2
    corrected_image = image.astype(np.float64) * correction
    corrected_image = np.clip(corrected_image, 0, 255)
    return corrected_image.astype(np.uint8)

def create_average_flat_field(image_paths, output_path):
    """Averages all images to create a synthetic flat-field."""
    if not image_paths:
        print("No images found to average.")
        return False
    sample_img = cv2.imread(image_paths[0], cv2.IMREAD_GRAYSCALE)
    accumulator = np.zeros_like(sample_img, dtype=np.float64)
    for i, file_path in enumerate(image_paths):
        img = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)
        accumulator += img
    average_image = (accumulator / len(image_paths)).astype(np.uint8)
    cv2.imwrite(output_path, average_image)
    return True

def find_optimal_k(average_image_path):
    """Finds the best k value by minimizing the standard deviation of a corrected flat-field."""
    flat_field = cv2.imread(average_image_path, cv2.IMREAD_GRAYSCALE)
    if flat_field is None: return 0.0
    
    best_k = 0
    lowest_std_dev = float('inf')

    print("  - Searching for optimal k...", end="", flush=True)
    # Search a reasonable range for k
    for k_test in np.arange(0.0, 3.0, 0.1):
        corrected_image = correct_vignette_grayscale(flat_field, k_test)
        std_dev = np.std(corrected_image)
        if std_dev < lowest_std_dev:
            lowest_std_dev = std_dev
            best_k = k_test
    print(f" Done. Found k = {best_k:.3f}")
    return best_k

def process_images_auto(input_dir, output_dir):
    BANDS = ['band685', 'band725', 'band750', 'band1000']
    LENS_WIDTH = 1280
    image_files = glob.glob(os.path.join(input_dir, '*.[tT][iI][fF]*')) + \
                  glob.glob(os.path.join(input_dir, '*.[jJ][pP][gG]*'))

    if not image_files:
        print(f"❌ No images found in '{input_dir}'. Please check the path.")
        return

    print(f"Found {len(image_files)} images. Starting automated analysis.")
    
    optimal_k_values = {}

    # Use a temporary directory for intermediate files that will be auto-deleted
    with tempfile.TemporaryDirectory() as temp_dir:
        print(f"Using temporary directory: {temp_dir}")
        
        for i, band_name in enumerate(BANDS):
            print(f"\nAnalyzing '{band_name}'...")
            raw_band_dir = os.path.join(temp_dir, band_name)
            os.makedirs(raw_band_dir)

            print(f"  - Splitting images for {band_name}...")
            for file_path in image_files:
                img = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)
                start_col = i * LENS_WIDTH
                end_col = (i + 1) * LENS_WIDTH
                lens_img = img[:, start_col:end_col]
                cv2.imwrite(os.path.join(raw_band_dir, os.path.basename(file_path)), lens_img)

            print(f"  - Creating average flat-field for {band_name}...")
            avg_ff_path = os.path.join(temp_dir, f"{band_name}_average.tif")
            create_average_flat_field(glob.glob(os.path.join(raw_band_dir, '*.*')), avg_ff_path)
            optimal_k_values[band_name] = find_optimal_k(avg_ff_path)

    print("\n--- Analysis Complete ---")
    print("Optimal k-values found:")
    for band, k in optimal_k_values.items():
        print(f"  - {band}: {k:.3f}")
    print("-------------------------\n")
    print("Starting final correction pass...")

    final_output_path = os.path.join(output_dir, 'stitched_corrected')
    os.makedirs(final_output_path, exist_ok=True)
    band_output_paths = {}
    for band in BANDS:
        path = os.path.join(output_dir, 'bands_corrected', band)
        os.makedirs(path, exist_ok=True)
        band_output_paths[band] = path
    
    for i, file_path in enumerate(image_files):
        filename = os.path.basename(file_path)
        print(f"Correcting image {i+1}/{len(image_files)}: {filename}")
        
        img = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)
        if img is None: continue
        
        corrected_lenses = []
        for j, band_name in enumerate(BANDS):
            start_col = j * LENS_WIDTH
            end_col = (j + 1) * LENS_WIDTH
            lens_img = img[:, start_col:end_col]
            
            k_value = optimal_k_values[band_name]
            corrected_img = correct_vignette_grayscale(lens_img, k_value)
            corrected_lenses.append(corrected_img)

            # Save the corrected sub-image (optional)
            cv2.imwrite(os.path.join(band_output_paths[band_name], filename), corrected_img)

        final_image = np.hstack(corrected_lenses)
        cv2.imwrite(os.path.join(final_output_path, filename), final_image)

    print("\nProcessing complete! ✨")


if __name__ == '__main__':
    INPUT_DIRECTORY = 'input_images'
    OUTPUT_DIRECTORY = 'output_images'
    os.makedirs(INPUT_DIRECTORY, exist_ok=True)
    process_images_auto(INPUT_DIRECTORY, OUTPUT_DIRECTORY)