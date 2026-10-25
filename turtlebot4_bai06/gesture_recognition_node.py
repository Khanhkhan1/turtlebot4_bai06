"""Phần 2: hand gesture recognition from the laptop webcam with MediaPipe.

Webcam (OpenCV) -> MediaPipe Gesture Recognizer (21 hand landmarks + gesture class and score)
-> Wave detected from the fingertip motion (not a MediaPipe class) -> debounce (a gesture must be
held for hold_time) -> /detected_gesture (std_msgs/String, published at publish_rate even when
nothing changes, so the robot can tell a lost signal from "no gesture"):
    None | Closed_Fist | Open_Palm | Pointing_Up | Thumb_Down | Thumb_Up | Victory | ILoveYou | Wave
The OpenCV window shows the hand landmarks, the model's gesture and confidence, the published
gesture and the robot state (/robot_behavior/state).

    ros2 run turtlebot4_bai06 gesture_recognition_node [--ros-args -p camera:=0 -p min_score:=0.6]
"""
import math
import os
import time
import urllib.request
from collections import deque

import cv2
import mediapipe as mp
import rclpy
from mediapipe.tasks.python import BaseOptions, vision
from rclpy.node import Node
from std_msgs.msg import String

WINDOW = "Gesture recognition - webcam"
MODEL_URL = ("https://storage.googleapis.com/mediapipe-models/gesture_recognizer/"
             "gesture_recognizer/float16/latest/gesture_recognizer.task")
DEFAULT_MODEL = os.path.expanduser("~/.cache/turtlebot4_bai06/gesture_recognizer.task")

# Landmark indices: 0 wrist, 4/8/12/16/20 fingertips, 6/10/14/18 middle joints (PIP).
HAND_CONNECTIONS = [(0, 1), (1, 2), (2, 3), (3, 4), (0, 5), (5, 6), (6, 7), (7, 8), (5, 9), (9, 10),
                    (10, 11), (11, 12), (9, 13), (13, 14), (14, 15), (15, 16), (13, 17), (17, 18),
                    (18, 19), (19, 20), (0, 17)]
TIPS, PIPS = (8, 12, 16, 20), (6, 10, 14, 18)
# Stop and cancel react faster than the gestures that select or start a behavior.
FAST = ("Open_Palm", "Closed_Fist")


def open_hand(landmarks):
    """At least 3 of the 4 fingers stretched: fingertip farther from the wrist than its middle joint."""
    w = landmarks[0]
    dist = lambda i: math.hypot(landmarks[i].x - w.x, landmarks[i].y - w.y)  # noqa: E731
    return sum(dist(t) > 1.1 * dist(p) for t, p in zip(TIPS, PIPS)) >= 3


class GestureFilter:
    """Debounce: a gesture becomes the output once the model reports it for its hold time in a row."""

    def __init__(self, hold, fast_hold, none_hold):
        self.hold, self.fast_hold, self.none_hold = hold, fast_hold, none_hold
        self.output = self.candidate = "None"
        self.since = 0.0

    def hold_for(self, label):
        return self.none_hold if label == "None" else self.fast_hold if label in FAST else self.hold

    def update(self, label, now):
        if label != self.candidate:
            self.candidate, self.since = label, now
        if label != self.output and now - self.since >= self.hold_for(label):
            self.output = label
        return self.output

    def progress(self, now):
        """0..1 while a new gesture is being held, None otherwise (drawn as a bar)."""
        if self.candidate == self.output or self.candidate == "None":
            return None
        return min(1.0, (now - self.since) / self.hold_for(self.candidate))


class WaveDetector:
    """Wave = open hand whose fingertips swing left-right: at least `turns` direction changes, each
    after a move larger than `amplitude` (fraction of the image width), within `window` seconds.
    The result is held for `hold` seconds after the last swing."""

    def __init__(self, window=1.5, turns=3, amplitude=0.04, hold=1.0):
        self.window, self.turns_needed, self.amplitude, self.hold = window, turns, amplitude, hold
        self.until = self.last_open = 0.0
        self.reset()

    def reset(self):
        self.direction, self.lo, self.hi, self.extreme = 0, None, None, None
        self.turns = deque()

    def update(self, landmarks, now):
        if landmarks is not None and open_hand(landmarks):
            self.last_open = now
            self._track(sum(landmarks[i].x for i in TIPS) / len(TIPS), now)
        elif landmarks is not None:      # another hand shape (fist, ...): no wave any more
            self.reset()
            self.until = 0.0
        elif now - self.last_open > 0.3:  # hand lost for a while
            self.reset()
        return now < self.until

    def _track(self, x, now):
        if self.direction == 0:
            self.lo = x if self.lo is None else min(self.lo, x)
            self.hi = x if self.hi is None else max(self.hi, x)
            if self.hi - self.lo > self.amplitude:
                self.direction, self.extreme = (1 if x == self.hi else -1), x
            return
        if (x - self.extreme) * self.direction > 0:
            self.extreme = x                      # still moving the same way
        elif abs(x - self.extreme) > self.amplitude:
            self.direction, self.extreme = -self.direction, x
            self.turns.append(now)
        while self.turns and now - self.turns[0] > self.window:
            self.turns.popleft()
        if len(self.turns) >= self.turns_needed:
            self.until = now + self.hold


def resolve_model(path, logger):
    path = os.path.expanduser(path) if path else DEFAULT_MODEL
    if not os.path.exists(path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        logger.info(f"Downloading the MediaPipe gesture model to {path}")
        urllib.request.urlretrieve(MODEL_URL, path + ".part")
        os.replace(path + ".part", path)
    return path


class GestureRecognitionNode(Node):
    def __init__(self):
        super().__init__("gesture_recognition_node")
        self.declare_parameter("camera", 0)            # webcam index
        self.declare_parameter("video", "")            # or a video file instead of the webcam (tests)
        self.declare_parameter("width", 640)
        self.declare_parameter("height", 480)
        self.declare_parameter("mirror", True)         # selfie view: moving the hand right moves it right
        self.declare_parameter("model", "")            # gesture_recognizer.task (downloaded if missing)
        self.declare_parameter("min_score", 0.6)       # lower model confidence counts as None
        self.declare_parameter("hold_time", 0.5)       # s a gesture must be held before it is published
        self.declare_parameter("stop_hold_time", 0.2)  # same for Open_Palm / Closed_Fist
        self.declare_parameter("publish_rate", 10.0)
        self.declare_parameter("show_window", True)

        p = lambda name: self.get_parameter(name).value  # noqa: E731
        self.mirror = bool(p("mirror"))
        self.min_score = float(p("min_score"))
        self.period = 1.0 / float(p("publish_rate"))
        self.show = bool(p("show_window"))
        self.video = p("video")

        options = vision.GestureRecognizerOptions(
            base_options=BaseOptions(model_asset_path=resolve_model(p("model"), self.get_logger())),
            running_mode=vision.RunningMode.VIDEO, num_hands=1,
            min_hand_detection_confidence=0.5, min_hand_presence_confidence=0.5,
            min_tracking_confidence=0.5)
        self.recognizer = vision.GestureRecognizer.create_from_options(options)
        self.filter = GestureFilter(float(p("hold_time")), float(p("stop_hold_time")), none_hold=0.3)
        self.wave = WaveDetector()

        self.source = os.path.expanduser(self.video) if self.video else int(p("camera"))
        self.size = (int(p("width")), int(p("height")))
        self.cap = self._open()
        if not self.cap.isOpened():
            raise RuntimeError(f"cannot open camera/video {self.source!r} "
                               "(macOS: allow camera access for the terminal app)")
        self.frame_time = 1.0 / (self.cap.get(cv2.CAP_PROP_FPS) or 30.0) if self.video else 0.0

        self.pub = self.create_publisher(String, "detected_gesture", 10)
        self.robot_state = "(robot_behavior_node not running)"
        self.create_subscription(String, "robot_behavior/state", self._on_state, 10)

        self.stable = "None"
        self.last_pub = self.last_ts = 0
        self.fps, self.t_fps, self.n_fps = 0.0, time.monotonic(), 0
        if self.show:
            cv2.namedWindow(WINDOW, cv2.WINDOW_AUTOSIZE)
        self.get_logger().info(f"Webcam {self.source!r} -> MediaPipe -> /detected_gesture "
                               f"(min score {self.min_score}, hold {p('hold_time')} s)")

    def _open(self):
        cap = cv2.VideoCapture(self.source)
        if not self.video:
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.size[0])
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.size[1])
        return cap

    def _on_state(self, msg):
        self.robot_state = msg.data

    def run(self):
        while rclpy.ok():
            t_frame = time.monotonic()
            ok, frame = self.cap.read()
            if not ok:
                if self.video:
                    self.get_logger().info("End of video")
                    return
                # No more /detected_gesture: robot_behavior_node stops on the missing signal.
                self.get_logger().error("Camera read failed, reopening", throttle_duration_sec=2.0)
                self.cap.release()
                time.sleep(1.0)
                self.cap = self._open()
                continue
            self._process(frame)
            rclpy.spin_once(self, timeout_sec=0.0)
            if self.show:
                key = cv2.waitKey(1) & 0xFF
                if key in (27, ord("q")) or cv2.getWindowProperty(WINDOW, cv2.WND_PROP_VISIBLE) < 1:
                    return
            if self.frame_time:   # play a video file at its own speed
                time.sleep(max(0.0, self.frame_time - (time.monotonic() - t_frame)))

    def _process(self, frame):
        now = time.monotonic()
        if self.mirror:
            frame = cv2.flip(frame, 1)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        ts = self.last_ts = max(int(now * 1000), self.last_ts + 1)   # VIDEO mode: increasing ms
        result = self.recognizer.recognize_for_video(mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb), ts)

        landmarks = result.hand_landmarks[0] if result.hand_landmarks else None
        raw, score = "None", 0.0
        if result.gestures:
            raw, score = result.gestures[0][0].category_name or "None", result.gestures[0][0].score
        label = raw if score >= self.min_score else "None"

        waving = self.wave.update(landmarks, now)
        stable = self.filter.update(label, now)
        if waving:
            stable = "Wave"
        if stable != self.stable:
            self.get_logger().info(f"Gesture: {stable}")
            self.stable = stable
        if now - self.last_pub >= self.period:
            self.pub.publish(String(data=stable))
            self.last_pub = now

        self.n_fps += 1
        if now - self.t_fps >= 1.0:
            self.fps, self.t_fps, self.n_fps = self.n_fps / (now - self.t_fps), now, 0
        if self.show:
            self._draw(frame, landmarks, raw, score, waving, now)
            cv2.imshow(WINDOW, frame)

    def _draw(self, frame, landmarks, raw, score, waving, now):
        h, w = frame.shape[:2]
        if landmarks is not None:
            pts = [(int(lm.x * w), int(lm.y * h)) for lm in landmarks]
            for a, b in HAND_CONNECTIONS:
                cv2.line(frame, pts[a], pts[b], (255, 255, 255), 2, cv2.LINE_AA)
            for pt in pts:
                cv2.circle(frame, pt, 4, (0, 0, 255), -1, cv2.LINE_AA)
            x0, y0 = min(p[0] for p in pts), min(p[1] for p in pts)
            x1, y1 = max(p[0] for p in pts), max(p[1] for p in pts)
            cv2.rectangle(frame, (x0 - 10, y0 - 10), (x1 + 10, y1 + 10), (0, 255, 255), 1)
            text = "Wave" if waving else f"{raw} {score:.2f}"
            # above the hand, or below it when that would hide it under the text panel
            y_text = y0 - 18 if y0 - 18 > 115 else min(h - 30, y1 + 38)
            cv2.putText(frame, text, (max(5, min(x0 - 10, w - 230)), y_text), cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                        (0, 255, 255), 2, cv2.LINE_AA)

        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (w, 92), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.55, frame, 0.45, 0, frame)
        model_text = f"MediaPipe: {raw} ({score * 100:.0f}%)" if raw != "None" or score else "MediaPipe: no hand"
        if waving:
            model_text += "  + wave motion"
        lines = [(model_text, (255, 255, 255)),
                 (f"/detected_gesture: {self.stable}", (0, 255, 0) if self.stable != "None" else (200, 200, 200)),
                 (f"Robot: {self.robot_state}", (0, 200, 255))]
        for i, (text, color) in enumerate(lines):
            cv2.putText(frame, text, (10, 25 + 28 * i), cv2.FONT_HERSHEY_SIMPLEX, 0.65, color, 2, cv2.LINE_AA)
        cv2.putText(frame, f"{self.fps:.0f} fps", (w - 80, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                    (200, 200, 200), 1, cv2.LINE_AA)

        held = self.filter.progress(now)
        if held is not None and not waving:   # a new gesture being held
            cv2.rectangle(frame, (0, h - 10), (int(w * held), h), (0, 255, 0), -1)
            cv2.putText(frame, f"hold {self.filter.candidate}...", (10, h - 18), cv2.FONT_HERSHEY_SIMPLEX,
                        0.6, (0, 255, 0), 2, cv2.LINE_AA)

    def close(self):
        self.cap.release()
        self.recognizer.close()
        if self.show:
            cv2.destroyAllWindows()


def main():
    rclpy.init()
    node = GestureRecognitionNode()
    try:
        node.run()
    except KeyboardInterrupt:
        pass
    finally:
        node.close()
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == "__main__":
    main()
