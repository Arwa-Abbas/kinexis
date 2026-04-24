import cv2
import mediapipe as mp
import numpy as np
from pathlib import Path
from tqdm import tqdm
from config import *

mp_pose = mp.solutions.pose


def extract_keypoints_from_video(video_path: str) -> np.ndarray:
    """
    Run BlazePose on every frame.
    Returns: (T, 33, 4) array — T frames, 33 landmarks, [x, y, z, visibility]
    Missing detections filled with zeros.
    """
    pose = mp_pose.Pose(
        min_detection_confidence=0.7,
        min_tracking_confidence=0.5,
        model_complexity=1,  # 0=lite, 1=full, 2=heavy
    )
    cap = cv2.VideoCapture(video_path)
    all_keypoints = []

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(frame_rgb)

        if results.pose_landmarks:
            kps = np.array(
                [
                    [lm.x, lm.y, lm.z, lm.visibility]
                    for lm in results.pose_landmarks.landmark
                ]
            )  # (33, 4)
        else:
            kps = np.zeros((33, 4))  # no detection — fill zeros

        all_keypoints.append(kps)

    cap.release()
    pose.close()
    return np.array(all_keypoints)  # (T, 33, 4)


def keypoints_to_stgcn_tensor(kps: np.ndarray) -> np.ndarray:
    """
    Convert raw BlazePose keypoints → ST-GCN input tensor.
    Input:  (T, 33, 4) — all BlazePose landmarks
    Output: (3, SEQUENCE_LEN, NUM_JOINTS) — xyz, padded/trimmed, 18 joints
    """
    T = kps.shape[0]

    # Select 18 joints from 33
    kps_18 = kps[:, JOINT_MAP, :3]  # (T, 18, 3) — xyz only

    # Normalize: subtract hip center (joints 7+8 = left+right hip)
    hip_center = (kps_18[:, 7, :] + kps_18[:, 8, :]) / 2.0  # (T, 3)
    kps_18 = kps_18 - hip_center[:, None, :]

    # Pad or trim to SEQUENCE_LEN frames
    if T >= SEQUENCE_LEN:
        kps_18 = kps_18[:SEQUENCE_LEN]
    else:
        pad = np.zeros((SEQUENCE_LEN - T, NUM_JOINTS, 3))
        kps_18 = np.concatenate([kps_18, pad], axis=0)

    # Reshape: (T, V, C) → (C, T, V) for ST-GCN convention
    tensor = kps_18.transpose(2, 0, 1)  # (3, 150, 18)
    return tensor


def batch_extract_all(video_dir: str, out_kp_dir: str, out_tensor_dir: str):
    """Process every video in video_dir, save keypoints + ST-GCN tensors."""
    Path(out_kp_dir).mkdir(parents=True, exist_ok=True)
    Path(out_tensor_dir).mkdir(parents=True, exist_ok=True)
    videos = list(Path(video_dir).glob("*.mp4"))

    for vpath in tqdm.tqdm(videos, desc="Extracting keypoints"):
        kp_path = Path(out_kp_dir) / (vpath.stem + "_kp.npy")
        ten_path = Path(out_tensor_dir) / (vpath.stem + "_tensor.npy")
        if ten_path.exists():
            continue

        kps = extract_keypoints_from_video(str(vpath))
        np.save(kp_path, kps)  # raw: (T, 33, 4)

        tensor = keypoints_to_stgcn_tensor(kps)
        np.save(ten_path, tensor)  # ready: (3, 150, 18)


if __name__ == "__main__":
    batch_extract_all(RAW_VIDEO_DIR, KEYPOINTS_DIR, STGCN_DIR)
