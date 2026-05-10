"""Keypoint normalization functions"""

import numpy as np
from .config import LEFT_HIP, RIGHT_HIP, LEFT_SHOULDER, RIGHT_SHOULDER


def normalize_keypoints(keypoint):
    """
    Normalize keypoints:
    - Center at hip center
    - Scale by torso length

    Args:
        keypoint: numpy array shape (1, T, 13, 2)

    Returns:
        normalized keypoints same shape
    """
    # Calculate hip center (between left and right hip)
    hip_center = (keypoint[:, :, LEFT_HIP, :] + keypoint[:, :, RIGHT_HIP, :]) / 2

    # Calculate shoulder center
    shoulder_center = (
        keypoint[:, :, LEFT_SHOULDER, :] + keypoint[:, :, RIGHT_SHOULDER, :]
    ) / 2

    # Torso length (distance between hip and shoulder)
    torso_length = np.linalg.norm(shoulder_center - hip_center, axis=-1, keepdims=True)

    # Center all joints at hip
    keypoint = keypoint - hip_center[:, :, np.newaxis, :]

    # Scale by torso length (add epsilon to avoid division by zero)
    keypoint = keypoint / (torso_length[:, :, np.newaxis, :] + 1e-6)

    return keypoint


def extract_keypoints_from_mat(mat_data):
    """
    Extract x, y, visibility from MATLAB file

    Args:
        mat_data: scipy.io loaded .mat file

    Returns:
        keypoints: (T, 13, 2) array
        visibility: (T, 13) array
        total_frames: int
    """
    x = mat_data["x"]  # (T, 13)
    y = mat_data["y"]  # (T, 13)
    visibility = mat_data["visibility"]  # (T, 13)
    total_frames = mat_data["nframes"][0][0]

    # Stack x and y into keypoints
    keypoints = np.stack([x, y], axis=-1)  # (T, 13, 2)

    return keypoints, visibility, total_frames
