"""
video_screenshot_capture.py

Play a video and save screenshots with keyboard shortcuts.

S or any regular key : Save the current frame as a JPG
Q or ESC             : Quit

Usage:
    python video_screenshot_capture.py
    python video_screenshot_capture.py "path/to/video.mp4"
"""

import os
import sys
import time
import cv2

OUTPUT_DIR = "captured_images"
WINDOW_NAME = "Video Screenshot Capture"


def main():
    video_path = sys.argv[1] if len(sys.argv) > 1 else ""

    if not video_path:
        video_path = input(
            "Enter the full path to your video (or drag the video file here): "
        ).strip().strip('"')

    if not os.path.isfile(video_path):
        print(f"Error: video file not found:\n{video_path}")
        input("Press Enter to exit...")
        return

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("Error: OpenCV could not open this video.")
        input("Press Enter to exit...")
        return

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    saved_count = 0
    paused = False
    current_frame = None

    fps = cap.get(cv2.CAP_PROP_FPS)
    if not fps or fps < 1 or fps > 240:
        fps = 25.0
    delay_ms = max(1, int(1000 / fps))

    print("\nVideo opened.")
    print("S or another key = save screenshot")
    print("SPACE = pause/resume")
    print("Q or ESC = quit\n")

    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)

    while True:
        if not paused:
            ok, frame = cap.read()
            if not ok:
                print("Video ended.")
                break
            current_frame = frame

        if current_frame is None:
            break

        display = current_frame.copy()
        instruction = "S/any key: SAVE | SPACE: PAUSE | Q/ESC: QUIT"
        cv2.putText(
            display, instruction, (15, 30),
            cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 0), 2,
            cv2.LINE_AA
        )
        cv2.imshow(WINDOW_NAME, display)

        key = cv2.waitKey(30 if paused else delay_ms) & 0xFF

        if key in (ord("q"), ord("Q"), 27):
            break
        elif key == 32:  # Space
            paused = not paused
        elif key != 255:  # Any other key saves a screenshot
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            filename = f"frame_{saved_count + 1:04d}_{timestamp}.jpg"
            output_path = os.path.join(OUTPUT_DIR, filename)

            if cv2.imwrite(output_path, current_frame):
                saved_count += 1
                print(f"Saved: {output_path}  (total: {saved_count})")
            else:
                print("Could not save screenshot.")

    cap.release()
    cv2.destroyAllWindows()
    print(f"\nFinished. Saved {saved_count} screenshot(s) in '{OUTPUT_DIR}'.")
    input("Press Enter to close...")


if __name__ == "__main__":
    main()
