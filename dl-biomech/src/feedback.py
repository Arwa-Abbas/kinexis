import cv2
import numpy as np
import pyttsx3
import threading
from config import VISIBILITY_THRESHOLD

# BlazePose connections (pairs of landmark indices)
POSE_CONNECTIONS = [
    (11, 12),
    (11, 13),
    (13, 15),
    (12, 14),
    (14, 16),
    (11, 23),
    (12, 24),
    (23, 24),
    (23, 25),
    (25, 27),
    (24, 26),
    (26, 28),
    (27, 29),
    (29, 31),
    (28, 30),
    (30, 32),
]

# Angle ranges per exercise for GREEN/YELLOW/RED coloring
ANGLE_RANGES = {
    "squat": {"left_knee": (80, 130), "right_knee": (80, 130)},
    "lunge": {"left_knee": (85, 100), "right_knee": (85, 100)},
}


def angle_color(angle, exercise, joint_name) -> tuple:
    """Return BGR color based on whether angle is in safe range."""
    ranges = ANGLE_RANGES.get(exercise, {})
    if joint_name in ranges:
        lo, hi = ranges[joint_name]
        if lo <= angle <= hi:
            return (0, 220, 100)  # green — correct
        elif abs(angle - (lo + hi) / 2) < 20:
            return (0, 200, 255)  # yellow — borderline
        else:
            return (50, 50, 255)  # red — error
    return (200, 200, 200)  # gray — no range defined


def draw_skeleton(frame, landmarks, features, errors, exercise, rep_count):
    """
    Overlay full feedback on a video frame.

    frame: np.array (H, W, 3) BGR
    landmarks: MediaPipe results.pose_landmarks.landmark (33 items)
    features: dict from features.extract_frame_features()
    errors: list of active error dicts from rules engine
    exercise: str, current exercise name
    rep_count: int
    """
    H, W = frame.shape[:2]

    # Convert normalized landmarks to pixel coords
    pts = {}
    for i, lm in enumerate(landmarks):
        if lm.visibility > VISIBILITY_THRESHOLD:
            pts[i] = (int(lm.x * W), int(lm.y * H))

    # ── Draw skeleton connections ──
    for a, b in POSE_CONNECTIONS:
        if a in pts and b in pts:
            err_names = [e["name"] for e in errors]
            color = (50, 50, 255) if errors else (0, 220, 100)
            cv2.line(frame, pts[a], pts[b], color, 2)

    # ── Draw joint dots ──
    for idx, (x, y) in pts.items():
        cv2.circle(frame, (x, y), 4, (255, 255, 255), -1)

    # ── Angle overlays ──
    angle_positions = {
        "left_knee": pts.get(25),
        "right_knee": pts.get(26),
        "left_hip": pts.get(23),
        "right_hip": pts.get(24),
        "left_elbow": pts.get(13),
        "right_elbow": pts.get(14),
    }
    for name, pos in angle_positions.items():
        val = features.get(name)
        if val and pos:
            col = angle_color(val, exercise, name)
            cv2.putText(
                frame,
                f"{val:.0f}°",
                pos,
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                col,
                1,
                cv2.LINE_AA,
            )

    # ── Status banner ──
    banner = f"Exercise: {exercise.upper()}   Reps: {rep_count}"
    cv2.rectangle(frame, (0, H - 50), (W, H), (20, 20, 20), -1)
    cv2.putText(
        frame, banner, (10, H - 28), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (220, 220, 220), 1
    )

    # ── Error overlays ──
    for i, err in enumerate(errors):
        cv2.rectangle(frame, (0, i * 36), (W, (i + 1) * 36), (0, 0, 180), -1)
        cv2.putText(
            frame,
            err["display"],
            (10, i * 36 + 24),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (255, 255, 255),
            2,
        )

    return frame


class VoiceFeedback:
    """Non-blocking TTS using a background thread."""

    def __init__(self):
        self.engine = pyttsx3.init()
        self.engine.setProperty("rate", 160)
        self._lock = threading.Lock()
        self._speaking = False

    def speak(self, text: str):
        """Fire-and-forget: speak in background, don't block video."""
        if self._speaking:
            return

        def _run():
            self._speaking = True
            self.engine.say(text)
            self.engine.runAndWait()
            self._speaking = False

        threading.Thread(target=_run, daemon=True).start()
