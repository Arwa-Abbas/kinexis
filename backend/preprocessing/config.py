"""Configuration for Penn Action preprocessing"""

import os
from pathlib import Path

# Paths
PROJECT_ROOT = Path(__file__).parent.parent.parent
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw" / "Penn_Action"
PROCESSED_DATA_DIR = DATA_DIR / "processed"

# Create directories
os.makedirs(RAW_DATA_DIR, exist_ok=True)
os.makedirs(PROCESSED_DATA_DIR, exist_ok=True)

# URLs
DATASET_URL = "https://www.cis.upenn.edu/~kostas/Penn_Action.tar.gz"

# Action mapping (15 original actions → 7 classes)
ACTION_MAPPING = {
    "pullup": 0,
    "pushup": 1,
    "bench_press": 2,
    "jumping_jacks": 3,
    "situp": 4,
    "jump_rope": 5,
    "squat": 6,
}

# All 15 actions in Penn Action
ALL_ACTIONS = [
    "baseball_pitch",
    "baseball_swing",
    "bench_press",
    "bowl",
    "clean_and_jerk",
    "golf_swing",
    "jump_rope",
    "jumping_jacks",
    "pullup",
    "pushup",
    "situp",
    "squat",
    "strum_guitar",
    "tennis_forehand",
    "tennis_serve",
]

# Joint indices for normalization
LEFT_HIP = 7
RIGHT_HIP = 8
LEFT_SHOULDER = 1
RIGHT_SHOULDER = 2

# Joint connections for visualization
EDGES = [
    (0, 1),
    (0, 2),  # head to shoulders
    (1, 3),
    (3, 5),  # left arm
    (2, 4),
    (4, 6),  # right arm
    (1, 7),
    (2, 8),  # shoulders to hips
    (7, 9),
    (9, 11),  # left leg
    (8, 10),
    (10, 12),  # right leg
]
