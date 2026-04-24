import cv2
import numpy as np
import os
from pathlib import Path
from config import TARGET_FPS, FRAME_WIDTH, FRAME_HEIGHT


def load_video_frames(video_path: str, max_frames: int = 300) -> np.ndarray:
    """
    Load a video file, resample to TARGET_FPS, resize frames.
    Returns: np.array of shape (T, H, W, 3)
    """
    cap = cv2.VideoCapture(video_path)
    src_fps = cap.get(cv2.CAP_PROP_FPS)
    frame_skip = max(1, int(src_fps / TARGET_FPS))  # sample every N-th frame

    frames = []
    frame_idx = 0

    while cap.isOpened() and len(frames) < max_frames:
        ret, frame = cap.read()
        if not ret:
            break
        if frame_idx % frame_skip == 0:
            frame = cv2.resize(frame, (FRAME_WIDTH, FRAME_HEIGHT))
            frames.append(frame)
        frame_idx += 1

    cap.release()
    return np.array(frames)  # (T, H, W, 3)


def normalize_frame(frame: np.ndarray) -> np.ndarray:
    """Per-channel mean subtraction for lighting invariance."""
    frame = frame.astype(np.float32)
    mean = np.array([104.0, 117.0, 123.0])  # ImageNet BGR mean
    return (frame - mean) / 255.0


def batch_preprocess_videos(video_dir: str, output_dir: str):
    """Run preprocessing on all MP4s in a directory."""
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    videos = list(Path(video_dir).glob("*.mp4"))
    print(f"Found {len(videos)} videos to preprocess")

    for vpath in videos:
        out_path = os.path.join(output_dir, vpath.stem + ".npy")
        if os.path.exists(out_path):
            continue  # skip already processed
        frames = load_video_frames(str(vpath))
        np.save(out_path, frames)
        print(f"  Saved {vpath.name} → {frames.shape}")


if __name__ == "__main__":
    from config import RAW_VIDEO_DIR, DATA_DIR

    batch_preprocess_videos(RAW_VIDEO_DIR, DATA_DIR + "/frames")
