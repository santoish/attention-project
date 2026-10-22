"""
process_videos.py - turn your recorded videos into training CSV files
----------------------------------------------------------------------
Put the videos in two folders (one behavior per video):

    videos/attentive/     e.g. a1.mp4, a2.mp4 ...
    videos/inattentive/   e.g. b1.mp4, b2.mp4 ...

Run (inside the project folder, with the environment activated):
    python process_videos.py

For every video it writes data/<label>_<name>.csv and adds a line to labels.csv.
Videos that were already processed are skipped. Then run:
    python build_dataset.py
    python train_rf.py
"""
import csv
import os
import sys

import cv2

from attention_core import Config, Engine

VIDEO_DIR = "videos"
DATA_DIR = "data"
LABELS = ["attentive", "inattentive"]
EXTS = (".mp4", ".avi", ".mov", ".mkv", ".m4v", ".webm")
MAX_WIDTH = 960      # shrink big frames (1080p etc.) to make processing faster
HEADER = ["time", "student_id", "ear", "pitch", "yaw", "roll", "status", "score"]


def process_video(path, out_csv, engine, cfg):
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        print("  cannot open the video, skipping")
        return 0
    fps = cap.get(cv2.CAP_PROP_FPS)
    fps = fps if 1 <= fps <= 240 else 30.0
    engine.reset()
    frame_idx, rows_written = 0, 0
    with open(out_csv, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(HEADER)
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            frame_idx += 1
            h, wd = frame.shape[:2]
            if wd > MAX_WIDTH:
                frame = cv2.resize(frame, (MAX_WIDTH, int(h * MAX_WIDTH / wd)))
            now = frame_idx / fps                      # same clock as the app uses for files
            _, rows, _ = engine.process(frame, now, cfg, None)
            for r in rows:
                w.writerow([f"{now:.2f}", r["student_id"], f"{r['ear']:.3f}",
                            f"{r['pitch']:.1f}", f"{r['yaw']:.1f}", f"{r['roll']:.1f}",
                            r["status"], f"{r['score']:.1f}"])
                rows_written += 1
            if frame_idx % 150 == 0:
                print(f"  ... {frame_idx} frames")
    cap.release()
    return rows_written


def main():
    if not os.path.isdir(VIDEO_DIR):
        sys.exit("Folder 'videos' not found. Create videos/attentive and "
                 "videos/inattentive and put your videos inside.")
    os.makedirs(DATA_DIR, exist_ok=True)

    todo = []
    for label in LABELS:
        folder = os.path.join(VIDEO_DIR, label)
        if not os.path.isdir(folder):
            print(f"note: folder {folder} not found")
            continue
        for name in sorted(os.listdir(folder)):
            if name.lower().endswith(EXTS):
                todo.append((label, os.path.join(folder, name), name))
    if not todo:
        sys.exit("No videos found in videos/attentive or videos/inattentive.")

    labels_new = not os.path.exists("labels.csv")
    known = set()
    if not labels_new:
        with open("labels.csv", newline="") as fh:
            known = {row[0] for row in csv.reader(fh) if row}

    engine, cfg = Engine(), Config()
    done = 0
    with open("labels.csv", "a", newline="") as lf:
        lw = csv.writer(lf)
        if labels_new:
            lw.writerow(["file", "label"])
        for i, (label, path, name) in enumerate(todo, 1):
            csv_name = f"{label}_{os.path.splitext(name)[0]}.csv"
            out_csv = os.path.join(DATA_DIR, csv_name)
            print(f"[{i}/{len(todo)}] {path}  ->  {csv_name}")
            if os.path.exists(out_csv) and csv_name in known:
                print("  already processed, skipping")
                continue
            n = process_video(path, out_csv, engine, cfg)
            if n == 0:
                print("  no faces were found in this video (nothing saved to labels)")
                continue
            if csv_name not in known:
                lw.writerow([csv_name, label])
                known.add(csv_name)
            print(f"  saved {n} rows")
            done += 1
    engine.close()
    print(f"\nFinished: {done} video(s) processed.")
    print("Next:  python build_dataset.py   then   python train_rf.py")


if __name__ == "__main__":
    main()
