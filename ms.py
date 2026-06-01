from picamera2 import Picamera2
import time
import os
import sys
import numpy as np
from PIL import Image

output_directory = "/home/lars/raw-ms"
image_prefix = "camMS-"
capture_interval_seconds = 0.2

def capture_images_continuously(raw_mode=False):
    today = time.localtime()
    date_directory = os.path.join(output_directory, f"{today.tm_year}/{today.tm_mon}/{today.tm_mday}")
    os.makedirs(date_directory, exist_ok=True)
    picam2 = Picamera2()

    print("Configuring camera for YUV420 capture...")
    camera_config = picam2.create_still_configuration(
        main={"size": (5120, 800), "format": "YUV420"},
        lores={"size": (2560, 400), "format": "YUV420"},
        display="lores",
    )
    picam2.configure(camera_config)
    picam2.start()

    if raw_mode:
        print(f"Starting continuous Monochrome (BMP) image capture to {output_directory}...")
    else:
        print(f"Starting continuous monochrome JPEG image capture to {output_directory}...")
    print("Press Ctrl+C to stop.")

    try:
        image_counter = 0
        while True:
            ns = time.time_ns()
            ms = ns // 1000
            s = ms // 1000000
            ms = ms % 1000000
            timestamp = f"{s}{ms:06d}"

            if raw_mode:
                filename = os.path.join(date_directory, f"{image_prefix}{timestamp}.tif")
                yuv_array = picam2.capture_array("main")
                image_array = yuv_array[0:800, :]
                img = Image.fromarray(image_array)
                img.save(filename, compression="tiff_lzw")
                print(f"Captured {filename}")
            else:
                filename = os.path.join(date_directory, f"{image_prefix}{timestamp}.jpg")
                picam2.capture_file(filename, format="jpeg")
                print(f"Captured {filename}")

            image_counter += 1

            if capture_interval_seconds > 0:
                time.sleep(capture_interval_seconds)

    except KeyboardInterrupt:
        print("\nStopping capture...")
    finally:
        picam2.stop()
        print("Camera stopped.")

if __name__ == "__main__":
    is_raw_mode = "-r" in sys.argv
    capture_images_continuously(raw_mode=is_raw_mode)
