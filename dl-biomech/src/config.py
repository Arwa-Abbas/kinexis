import os

# ── Paths ──────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.dirname(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
RAW_VIDEO_DIR = os.path.join(DATA_DIR, "raw_videos")
KEYPOINTS_DIR = os.path.join(DATA_DIR, "keypoints")
FEATURES_DIR = os.path.join(DATA_DIR, "features")
STGCN_DIR = os.path.join(DATA_DIR, "stgcn_input")
LABELS_CSV = os.path.join(DATA_DIR, "labels.csv")
PRETRAINED_PT = os.path.join(BASE_DIR, "models/pretrained/xview_joint.pt")
CHECKPOINTS_DIR = os.path.join(BASE_DIR, "models/checkpoints")
SESSION_LOG_DIR = os.path.join(BASE_DIR, "outputs/session_logs")

# ── Exercise classes ────────────────────────────────────
EXERCISE_CLASSES = {
    "squat": 0,
    "lunge": 1,
    "shoulder_press": 2,
    "lateral_raise": 3,
    "bicep_curl": 4,
    "deadlift": 5,
}

QUALITY_CLASSES = {
    "correct": 0,
    "knee_valgus": 1,
    "forward_lean": 2,
    "shallow_depth": 3,
    "asymmetry": 4,
    "incomplete_rep": 5,
}

# ── Video processing ────────────────────────────────────
TARGET_FPS = 30
FRAME_WIDTH = 640
FRAME_HEIGHT = 480
SEQUENCE_LEN = 150  # T: frames per ST-GCN input
NUM_JOINTS = 18  # V: joints used from BlazePose
NUM_CHANNELS = 3  # C: x, y, z

# ── BlazePose joint indices we USE (18 of 33) ───────────
# Maps our 18 model joints to BlazePose's 33 landmark indices
JOINT_MAP = [
    0,  # 0: nose
    11,  # 1: left shoulder
    12,  # 2: right shoulder
    13,  # 3: left elbow
    14,  # 4: right elbow
    15,  # 5: left wrist
    16,  # 6: right wrist
    23,  # 7: left hip
    24,  # 8: right hip
    25,  # 9: left knee
    26,  # 10: right knee
    27,  # 11: left ankle
    28,  # 12: right ankle
    29,  # 13: left heel
    30,  # 14: right heel
    31,  # 15: left foot index
    32,  # 16: right foot index
    7,  # 17: left ear (head position)
]

# ── Training hyperparameters ────────────────────────────
BATCH_SIZE = 8
NUM_EPOCHS = 50
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-4
TRAIN_SPLIT = 0.70
VAL_SPLIT = 0.15
PATIENCE = 10  # early stopping patience
DEVICE = "cuda" if __import__("torch").cuda.is_available() else "cpu"

# ── Feedback thresholds ─────────────────────────────────
KNEE_VALGUS_THRESHOLD = 165.0  # degrees — flag if below
FORWARD_LEAN_THRESHOLD = 50.0  # degrees trunk angle — flag if above
ASYMMETRY_THRESHOLD = 12.0  # degrees L/R diff — flag if above
SHALLOW_DEPTH_THRESHOLD = 120.0  # degrees knee flex at bottom
DEBOUNCE_FRAMES = 10  # error must persist N frames before alert
VISIBILITY_THRESHOLD = 0.5  # ignore keypoints below this visibility
