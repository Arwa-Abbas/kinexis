"""Build the final PySKL pickle file"""

import pickle
import numpy as np
import scipy.io as sio
from pathlib import Path
from tqdm import tqdm

from .config import RAW_DATA_DIR, PROCESSED_DATA_DIR, ACTION_MAPPING, ALL_ACTIONS
from .normalize import normalize_keypoints, extract_keypoints_from_mat


def scan_labels_directory(labels_dir):
    """Scan all .mat files and categorize by action"""

    mat_files = list(labels_dir.glob("*.mat"))

    print(f"Found {len(mat_files)} .mat files")

    # Count actions
    from collections import Counter

    action_counts = Counter()

    for mat_path in mat_files:
        mat = sio.loadmat(mat_path)
        action_name = mat["action"][0].strip()
        if action_name in ACTION_MAPPING:
            action_counts[action_name] += 1

    return mat_files, action_counts


def build_pyskl_pickle(output_path=None):
    """
    Build the PySKL format pickle file

    Output structure:
    {
        'split': {'train': [...], 'test': [...]},
        'annotations': [
            {
                'frame_dir': str,
                'total_frames': int,
                'img_shape': (h, w),
                'original_shape': (h, w),
                'label': int,
                'keypoint': np.ndarray (1, T, 13, 2),
                'keypoint_score': np.ndarray (1, T, 13)
            },
            ...
        ]
    }
    """

    if output_path is None:
        output_path = PROCESSED_DATA_DIR / "penn_action_pyskl.pkl"

    labels_dir = RAW_DATA_DIR / "labels"

    if not labels_dir.exists():
        raise FileNotFoundError(f"Labels directory not found: {labels_dir}")

    print(f"Processing labels from {labels_dir}")

    split = {"train": [], "test": []}
    annotations = []

    mat_files = list(labels_dir.glob("*.mat"))

    for mat_path in tqdm(mat_files, desc="Processing videos"):
        fname_dir = mat_path.stem
        mat = sio.loadmat(mat_path)

        # Get action label (only keep our 7 classes)
        action_name = mat["action"][0].strip()
        if action_name not in ACTION_MAPPING:
            continue

        label = ACTION_MAPPING[action_name]
        total_frames = mat["nframes"][0][0]

        # Get image dimensions
        w, h, _ = mat["dimensions"][0]
        img_shape = (int(h), int(w))

        # Extract keypoints
        keypoints_3d, visibility, _ = extract_keypoints_from_mat(mat)

        # Reshape to (1, T, 13, 2)
        keypoints = keypoints_3d[np.newaxis, ...]  # (1, T, 13, 2)

        # Normalize keypoints
        keypoints = normalize_keypoints(keypoints)

        # Keypoint scores (visibility)
        keypoint_score = visibility[np.newaxis, ...]  # (1, T, 13)

        # Train/test split
        is_train = mat["train"][0][0] == 1
        if is_train:
            split["train"].append(fname_dir)
        else:
            split["test"].append(fname_dir)

        annotations.append(
            {
                "frame_dir": fname_dir,
                "total_frames": total_frames,
                "img_shape": img_shape,
                "original_shape": img_shape,
                "label": label,
                "keypoint": keypoints.astype(np.float32),
                "keypoint_score": keypoint_score.astype(np.float32),
            }
        )

    # Create final data structure
    data = {"split": split, "annotations": annotations}

    # Save to pickle
    with open(output_path, "wb") as f:
        pickle.dump(data, f)

    print(f"\nSaved to {output_path}")
    print(f"   Total videos: {len(annotations)}")
    print(f"   Train: {len(split['train'])}, Test: {len(split['test'])}")

    return data


def verify_pickle(pickle_path):
    """Verify the generated pickle file"""

    with open(pickle_path, "rb") as f:
        data = pickle.load(f)

    print("\n=== Verification ===")
    print(f"Total videos: {len(data['annotations'])}")
    print(f"Train videos: {len(data['split']['train'])}")
    print(f"Test videos: {len(data['split']['test'])}")

    # Check first annotation
    ann = data["annotations"][0]
    print(f"\nSample annotation:")
    print(f"  frame_dir: {ann['frame_dir']}")
    print(f"  keypoint shape: {ann['keypoint'].shape}")
    print(f"  keypoint_score shape: {ann['keypoint_score'].shape}")
    print(f"  label: {ann['label']}")

    return data


if __name__ == "__main__":
    # Build the pickle file
    data = build_pyskl_pickle()

    # Verify it
    verify_pickle(PROCESSED_DATA_DIR / "penn_action_pyskl.pkl")
