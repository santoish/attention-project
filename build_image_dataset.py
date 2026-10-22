"""
build_image_dataset.py
Build dataset.csv from labelled photos/screenshots.

Expected folders:
images/
    attentive/
        photo1.jpg
        photo2.png
    inattentive/
        photo3.jpg
        photo4.png

IMPORTANT: Each image's folder label is applied to every detected face in it.
Use images where the visible behavior is reasonably clear. If a photo contains
people showing different behaviors, crop/save each person separately first.
"""
import os
import sys
import cv2
import pandas as pd

from attention_core import Config, Engine

IMAGE_DIR = "images"
OUTPUT_CSV = "image_dataset.csv"
LABELS = ("attentive", "inattentive")
EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".webp")

def main():
    if not os.path.isdir(IMAGE_DIR):
        sys.exit(
            "Folder 'images' not found. Create images/attentive and "
            "images/inattentive, then put your photos inside."
        )

    records = []
    engine = Engine(max_faces=30)
    cfg = Config()
    try:
        for label in LABELS:
            folder = os.path.join(IMAGE_DIR, label)
            if not os.path.isdir(folder):
                print(f"Folder missing (skipping): {folder}")
                continue

            paths = [
                os.path.join(folder, name)
                for name in sorted(os.listdir(folder))
                if name.lower().endswith(EXTS)
            ]
            if not paths:
                print(f"No images found in {folder}")
                continue

            for i, path in enumerate(paths, start=1):
                frame = cv2.imread(path)
                if frame is None:
                    print(f"Could not read, skipping: {path}")
                    continue

                # Reset tracking for every independent photo.
                engine.reset()
                _, rows, _ = engine.process(frame, 1.0, cfg, None)
                if not rows:
                    print(f"No usable face detected: {path}")
                    continue

                rel_path = os.path.relpath(path, IMAGE_DIR).replace("\\", "/")
                for face_no, row in enumerate(rows, start=1):
                    records.append({
                        "ear": float(row["ear"]),
                        "pitch": float(row["pitch"]),
                        "yaw": float(row["yaw"]),
                        "roll": float(row["roll"]),
                        "label": label,
                        "source": rel_path,
                        "face_no": face_no,
                    })

                print(f"[{i}/{len(paths)}] {path}: {len(rows)} face(s)")

    finally:
        engine.close()

    if not records:
        sys.exit(
            "No face features were extracted. Try clearer, front-facing photos "
            "with larger visible faces."
        )

    df = pd.DataFrame(records)
    df.to_csv(OUTPUT_CSV, index=False)

    print(f"\nSaved {OUTPUT_CSV} with {len(df)} face rows.")
    print("\nRows per label:")
    print(df["label"].value_counts().to_string())
    print("\nNext run: python train_image_model.py")

if __name__ == "__main__":
    main()
