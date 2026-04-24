import numpy as np
from scipy.signal import find_peaks
from config import JOINT_MAP, VISIBILITY_THRESHOLD


def joint_angle(a, b, c) -> float:
    """
    Angle (degrees) at joint b, with vectors from b→a and b→c.
    Works with 2D [x,y] or 3D [x,y,z] arrays.
    """
    a, b, c = np.array(a), np.array(b), np.array(c)
    ba = a - b
    bc = c - b
    cos_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-7)
    return float(np.degrees(np.arccos(np.clip(cos_angle, -1.0, 1.0))))


def extract_frame_features(landmarks) -> dict:
    """
    Extract biomechanical features from a single BlazePose frame.

    landmarks: list of 33 objects with .x .y .z .visibility
               OR np.array of shape (33, 4) with [x, y, z, vis]
    Returns: dict of named angle measurements
    """
    # Handle both MediaPipe landmark objects and numpy arrays
    if hasattr(landmarks[0], "x"):
        pts = {
            i: [lm.x, lm.y, lm.z]
            for i, lm in enumerate(landmarks)
            if lm.visibility > VISIBILITY_THRESHOLD
        }
    else:
        pts = {
            i: landmarks[i, :3]
            for i in range(33)
            if landmarks[i, 3] > VISIBILITY_THRESHOLD
        }

    features = {}

    def angle_if_visible(key, a_idx, b_idx, c_idx):
        if all(i in pts for i in [a_idx, b_idx, c_idx]):
            features[key] = joint_angle(pts[a_idx], pts[b_idx], pts[c_idx])
        else:
            features[key] = None

    # ── Lower body ──────────────────────────────────────────
    angle_if_visible("left_knee", 23, 25, 27)  # hip-knee-ankle
    angle_if_visible("right_knee", 24, 26, 28)
    angle_if_visible("left_hip", 11, 23, 25)  # shoulder-hip-knee
    angle_if_visible("right_hip", 12, 24, 26)
    angle_if_visible("left_ankle", 25, 27, 31)  # knee-ankle-foot
    angle_if_visible("right_ankle", 26, 28, 32)

    # ── Upper body ──────────────────────────────────────────
    angle_if_visible("left_elbow", 11, 13, 15)  # shoulder-elbow-wrist
    angle_if_visible("right_elbow", 12, 14, 16)
    angle_if_visible("left_shoulder", 13, 11, 23)  # elbow-shoulder-hip
    angle_if_visible("right_shoulder", 14, 12, 24)

    # ── Trunk inclination (forward lean) ────────────────────
    if 11 in pts and 23 in pts:
        l_shoulder = np.array(pts[11])
        l_hip = np.array(pts[23])
        trunk_vec = l_shoulder - l_hip
        vertical = np.array([0, -1, 0])
        cos_a = np.dot(trunk_vec, vertical) / (np.linalg.norm(trunk_vec) + 1e-7)
        features["trunk_angle"] = float(np.degrees(np.arccos(np.clip(cos_a, -1, 1))))
    else:
        features["trunk_angle"] = None

    # ── Symmetry: L/R knee difference ───────────────────────
    if features["left_knee"] and features["right_knee"]:
        features["knee_symmetry"] = abs(features["left_knee"] - features["right_knee"])
    else:
        features["knee_symmetry"] = None

    return features


def count_reps(angle_sequence: list, prominence: float = 20.0) -> int:
    """
    Count exercise reps from a knee angle time series.
    Uses valley detection (angle goes low = squat position).
    """
    angles = np.array([a if a is not None else 180 for a in angle_sequence])
    # Invert: valleys in knee angle = peaks in inverted signal
    valleys, _ = find_peaks(-angles, prominence=prominence, distance=15)
    return len(valleys)
