"""
Laptop Webcam Dataset Capture
-----------------------------
Press any key (except the controls below) to save the current webcam frame.
Images are saved in numbered order: image_0001.jpg, image_0002.jpg, ...

Install:
    pip install opencv-python

Run:
    python webcam_dataset_capture.py

Controls:
    Any key except the ones below : Capture and save an image
    q or Esc                       : Quit
    s                              : Skip / do nothing
    r                              : Reset numbering (asks for confirmation)
"""

import cv2
from pathlib import Path

# Settings
CAMERA_INDEX = 0
OUTPUT_DIR = Path("dataset/inattentive")
FILENAME_PREFIX = "image_inattentive"
IMAGE_EXTENSION = ".jpg"
DIGITS = 4
JPEG_QUALITY = 95

def next_image_number(folder: Path) -> int:
    """Continue numbering after the highest existing image filename."""
    folder.mkdir(parents=True, exist_ok=True)
    highest = 0
    pattern = f"{FILENAME_PREFIX}*{IMAGE_EXTENSION}"

    for path in folder.glob(pattern):
        number_part = path.stem.removeprefix(FILENAME_PREFIX)
        if number_part.isdigit():
            highest = max(highest, int(number_part))

    return highest + 1

def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    camera = cv2.VideoCapture(CAMERA_INDEX)
    if not camera.isOpened():
        print("ERROR: Could not open the laptop camera.")
        print("Close other apps using the camera, or try changing CAMERA_INDEX to 1.")
        return

    image_number = next_image_number(OUTPUT_DIR)
    print(f"Saving images to: {OUTPUT_DIR.resolve()}")
    print("Press any key to capture an image.")
    print("Press 's' to skip, or 'q' / Esc to quit.")

    try:
        while True:
            success, frame = camera.read()
            if not success:
                print("ERROR: Could not read a frame from the camera.")
                break

            preview = frame.copy()
            status = f"Next: {FILENAME_PREFIX}{image_number:0{DIGITS}d}{IMAGE_EXTENSION}"
            cv2.putText(
                preview, status, (15, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2, cv2.LINE_AA
            )
            cv2.putText(
                preview, "Any key: Capture | s: Skip | q/Esc: Quit",
                (15, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                (255, 255, 255), 2, cv2.LINE_AA
            )
            cv2.imshow("Webcam Dataset Capture", preview)

            key = cv2.waitKey(1) & 0xFF
            if key == 255:  # No key pressed
                continue

            if key in (ord("q"), 27):  # q or Esc
                break
            if key == ord("s"):
                continue
            if key == ord("r"):
                print("Reset is disabled to protect existing files. Numbering continues automatically.")
                continue

            filename = f"{FILENAME_PREFIX}{image_number:0{DIGITS}d}{IMAGE_EXTENSION}"
            output_path = OUTPUT_DIR / filename

            saved = cv2.imwrite(
                str(output_path),
                frame,
                [int(cv2.IMWRITE_JPEG_QUALITY), JPEG_QUALITY]
            )
            if saved:
                print(f"Saved: {output_path}")
                image_number += 1
            else:
                print(f"ERROR: Could not save {output_path}")

    finally:
        camera.release()
        cv2.destroyAllWindows()
        print("Camera closed.")

if __name__ == "__main__":
    main()
