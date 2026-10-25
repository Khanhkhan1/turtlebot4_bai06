"""Phần 1: read the laptop webcam with OpenCV and show it (q / Esc to quit).

    ros2 run turtlebot4_bai06 webcam_view [camera_index]
"""
import sys
import time

import cv2

WINDOW = "Webcam (OpenCV)"


def main():
    index = int(sys.argv[1]) if len(sys.argv) > 1 and sys.argv[1].isdigit() else 0
    cap = cv2.VideoCapture(index)
    if not cap.isOpened():
        sys.exit(f"Cannot open camera {index} (macOS: allow camera access for the terminal app)")
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    t0, frames, fps = time.monotonic(), 0, 0.0
    while True:
        ok, frame = cap.read()
        if not ok:
            print("Camera read failed")
            break
        frame = cv2.flip(frame, 1)
        frames += 1
        if time.monotonic() - t0 >= 1.0:
            fps, t0, frames = frames / (time.monotonic() - t0), time.monotonic(), 0
        h, w = frame.shape[:2]
        cv2.putText(frame, f"camera {index}  {w}x{h}  {fps:.0f} fps", (10, 25), cv2.FONT_HERSHEY_SIMPLEX,
                    0.7, (0, 255, 0), 2, cv2.LINE_AA)
        cv2.imshow(WINDOW, frame)
        if cv2.waitKey(1) & 0xFF in (27, ord("q")) or cv2.getWindowProperty(WINDOW, cv2.WND_PROP_VISIBLE) < 1:
            break
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
