"""
Kinexis — Local Inference
Usage:
    python inference.py --source 0                          # webcam
    python inference.py --source videos/squat.mp4          # video file
    python inference.py --source videos/squat.mp4 --ground_truth squats
    python inference.py --source videos/squat.mp4 --ground_truth squats --ground_truth_reps 8
    python inference.py --source videos/squat.mp4 --output outputs/out.mp4

Changes from previous version:
    - Reverted normalise_skeleton to hip-origin (matches PennAction training preprocessing)
    - Removed exercise_sanity_check entirely — model (85% acc) is more reliable than
      fragile single-frame geometry heuristics that caused whack-a-mole between exercises
    - Removed squat_geometry_hint override — let the model decide
    - Buffer no longer flushed on sustained ankle loss — preserves temporal context
      mid-squat when ankles briefly leave frame
    - pred_history window shrunk: 25→10 for faster exercise label switching
    - classify() now waits for CLIP_LEN // 2 frames instead of hardcoded 20

    v3 fixes (per observed failures):
    - MIN_KNEE_VIS lowered 0.50→0.20: was rejecting push-up and squat frames where
      knees are bent/partially occluded. Ankles remain the primary gate.
    - MediaPipe tracking confidence raised 0.5→0.6 and added skeleton stability check
      to suppress the "jumping skeleton" between detections.
    - CONFIDENCE_THRESHOLD raised 0.60→0.65 to reduce bench/pull-up confusion when
      the model is uncertain.
    - Pull-up rep thresholds tightened: down 90→70, up 150→140. Elbow must fully
      compress to 70° at top and only counts standing when nearly straight at 140°+.
      Prevents noise in the 90-150° mid-range from triggering phantom reps.
    - Bench press rep thresholds tightened: down 80→75, up 150→160. Lying elbow
      extension is larger than upright — needs a wider up threshold to avoid
      counting every small movement.
    - Push-up thresholds adjusted: down 90→80, up 150→145. Chest-to-floor requires
      tighter elbow compression.
    - Sit-up thresholds tightened: down 140→150 (must be properly flat), up 60→70
      (more reachable sit-up height). Wider dead-zone between states prevents noise.
    - Jump rope thresholds loosened: down 155→145, up 168→165. Wider window catches
      the small knee bounce that characterises jump rope.
    - Squat thresholds adjusted: down 115→110, up 155→160. Slightly stricter bottom
      position, slightly more generous standing position.
    - Jumping jack convention clarified: 1 rep = arms-down → arms-up (half cycle).
      Standard gym convention; thresholds unchanged but documented.
"""

import os, json, collections, math, argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import cv2
import mediapipe as mp
from typing import Tuple

# ── Config ────────────────────────────────────────────────────────────────────
DEVICE               = 'cuda' if torch.cuda.is_available() else 'cpu'
CONFIDENCE_THRESHOLD = 0.40  # 0.65
INFER_EVERY          = 5
STABLE_FRAMES_NEEDED = 8     # consecutive high-conf frames before reps start counting
MIN_CONF_FOR_REPS    = 0.40  # 0.65
MIN_ANKLE_VIS        = 0.15  # primary frame gate — ankles must be visible
MIN_KNEE_VIS         = 0.20  # lowered from 0.50 — push-ups and squats have low knee vis

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

MODEL_DIR = os.path.join(BASE_DIR, "backend", "models")

CHECKPOINT_PATH = os.path.join(MODEL_DIR, "best_model.pth")
META_PATH       = os.path.join(MODEL_DIR, "model_meta.json")

with open(META_PATH) as f:
    META = json.load(f)

CLIP_LEN    = META['clip_len']
NUM_JOINTS  = META['num_joints']
NUM_CLASSES = META['num_classes']
LABEL_MAP   = {int(k): v for k, v in META['label_map'].items()}
MP_TO_PENN  = {int(k): int(v) for k, v in META['mediapipe_to_pennaction'].items()}

# ── Joint indices ─────────────────────────────────────────────────────────────
HEAD  = 0
L_SHO, R_SHO = 1, 2
L_ELB, R_ELB = 3, 4
L_WRI, R_WRI = 5, 6
L_HIP, R_HIP = 7, 8
L_KNE, R_KNE = 9, 10
L_ANK, R_ANK = 11, 12

CRITICAL_JOINTS = [L_KNE, R_KNE, L_ANK, R_ANK, L_HIP, R_HIP]

# ── Graph ─────────────────────────────────────────────────────────────────────
EDGES = [(0,1),(0,2),(1,3),(3,5),(2,4),(4,6),(1,7),(2,8),(7,9),(9,11),(8,10),(10,12)]

def build_adj_3subset(n, edges):
    A = np.zeros((3, n, n), dtype=np.float32)
    A[0] = np.eye(n, dtype=np.float32)
    centre = {1, 2, 7, 8}
    for i, j in edges:
        if j in centre: A[1,j,i]=1.; A[2,i,j]=1.
        else:           A[1,i,j]=1.; A[2,j,i]=1.
    for k in range(3):
        rs = A[k].sum(1, keepdims=True).clip(min=1e-6)
        A[k] = A[k] / rs
    return A

A3_np = build_adj_3subset(NUM_JOINTS, EDGES)

# ── Model architecture ────────────────────────────────────────────────────────
class UnitGCN(nn.Module):
    def __init__(self, in_ch, out_ch, A3):
        super().__init__()
        self.register_buffer('A', torch.from_numpy(A3))
        self.bn   = nn.BatchNorm2d(out_ch)
        self.relu = nn.ReLU(inplace=True)
        self.conv = nn.Conv2d(in_ch, out_ch * 3, 1)
        self.down = nn.Sequential(nn.Conv2d(in_ch, out_ch, 1), nn.BatchNorm2d(out_ch))

    def forward(self, x):
        N, C, T, V = x.shape
        y = self.conv(x)
        out_ch = y.shape[1] // 3
        y = y.view(N, 3, out_ch, T, V)
        y = torch.einsum('nkctv,kvw->nctw', y, self.A)
        return self.relu(self.bn(y) + self.down(x))

def _branch_channels(ch) -> Tuple[int, int]:
    return {64: (14,10), 128: (23,21), 256: (46,42)}[ch]

class _ConvModule(nn.Module):
    def __init__(self, in_c, out_c, kernel, padding, dilation):
        super().__init__()
        self.conv = nn.Conv2d(in_c, out_c, kernel, padding=padding, dilation=dilation)
    def forward(self, x): return self.conv(x)

class MSTCN(nn.Module):
    def __init__(self, ch):
        super().__init__()
        bc0, bc = _branch_channels(ch)
        def branch(ic, oc, k=(3,1), p=(1,0), d=(1,1)):
            return nn.Sequential(nn.Conv2d(ic,oc,1), nn.BatchNorm2d(oc),
                                 nn.ReLU(inplace=True), _ConvModule(oc,oc,k,p,d))
        self.branches = nn.ModuleList([
            branch(ch, bc0, (3,1), (1,0), (1,1)),
            branch(ch, bc,  (3,1), (2,0), (2,1)),
            branch(ch, bc,  (3,1), (3,0), (3,1)),
            branch(ch, bc,  (3,1), (4,0), (4,1)),
            branch(ch, bc,  (3,1), (1,0), (1,1)),
            nn.Conv2d(ch, bc, 1),
        ])
        self.transform = nn.Sequential(nn.BatchNorm2d(ch), nn.ReLU(inplace=True), nn.Conv2d(ch,ch,1))
        self.bn = nn.BatchNorm2d(ch)

    def forward(self, x):
        return self.bn(self.transform(torch.cat([b(x) for b in self.branches], dim=1)))

class STGCNBlock(nn.Module):
    def __init__(self, in_ch, out_ch, A3):
        super().__init__()
        self.gcn = UnitGCN(in_ch, out_ch, A3)
        self.tcn = MSTCN(out_ch)
        self.act = nn.ReLU(inplace=True)
        self.residual = (nn.Sequential(nn.Conv2d(in_ch,out_ch,1,bias=False), nn.BatchNorm2d(out_ch))
                         if in_ch != out_ch else nn.Identity())
    def forward(self, x):
        return self.act(self.tcn(self.gcn(x)) + self.residual(x))

class STGCNBackbone(nn.Module):
    def __init__(self, in_channels=2, A3=A3_np):
        super().__init__()
        self.data_bn = nn.BatchNorm1d(in_channels * NUM_JOINTS)
        cfg = [(in_channels,64),(64,64),(64,64),(64,64),
               (64,128),(128,128),(128,128),(128,256),(256,256),(256,256)]
        self.gcn  = nn.ModuleList([STGCNBlock(ic, oc, A3) for ic, oc in cfg])
        self.pool = nn.AdaptiveAvgPool2d(1)
    def forward(self, x):
        N, C, T, V = x.shape
        x = x.permute(0,1,3,2).contiguous().view(N, C*V, T)
        x = self.data_bn(x)
        x = x.view(N, C, V, T).permute(0,1,3,2).contiguous()
        for layer in self.gcn: x = layer(x)
        return self.pool(x).view(N, -1)

class ConfidenceEncoder(nn.Module):
    def __init__(self, num_joints=13, out_dim=32):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.mlp  = nn.Sequential(nn.Linear(num_joints,32), nn.BatchNorm1d(32), nn.ReLU(True),
                                  nn.Linear(32, out_dim), nn.Sigmoid())
    def forward(self, conf):
        c = conf.squeeze(1).permute(0,2,1)
        return self.mlp(self.pool(c).squeeze(-1))

class STGCNFineTuned(nn.Module):
    def __init__(self, num_classes=7, A3=A3_np):
        super().__init__()
        self.backbone = STGCNBackbone(in_channels=2, A3=A3)
        self.conf_enc = ConfidenceEncoder(num_joints=NUM_JOINTS, out_dim=32)
        self.head     = nn.Sequential(nn.Linear(288,128), nn.BatchNorm1d(128), nn.ReLU(True),
                                      nn.Dropout(0.4), nn.Linear(128, num_classes))
    def forward(self, xy, conf):
        return self.head(torch.cat([self.backbone(xy), self.conf_enc(conf)], dim=1))

# ── Buffer ────────────────────────────────────────────────────────────────────
class SlidingWindowBuffer:
    def __init__(self, clip_len=CLIP_LEN):
        self.clip_len  = clip_len
        self.buffer    = collections.deque(maxlen=clip_len)
        self._pad_xy   = np.zeros((NUM_JOINTS, 2), dtype=np.float32)
        self._pad_conf = np.zeros(NUM_JOINTS,      dtype=np.float32)

    def push(self, xy_norm, conf):
        self.buffer.append({
            'xy':   np.asarray(xy_norm, dtype=np.float32),
            'conf': np.asarray(conf,    dtype=np.float32),
        })

    def __len__(self): return len(self.buffer)

    def get_tensors(self):
        frames = list(self.buffer)
        pad    = frames[-1] if frames else {'xy': self._pad_xy, 'conf': self._pad_conf}
        while len(frames) < self.clip_len:
            frames.append(pad)
        xy_seq   = np.stack([f['xy']   for f in frames], axis=0).astype(np.float32)  # (T,V,2)
        conf_seq = np.stack([f['conf'] for f in frames], axis=0).astype(np.float32)  # (T,V)
        xy_t   = torch.from_numpy(xy_seq.transpose(2,0,1)).unsqueeze(0)              # (1,2,T,V)
        conf_t = torch.from_numpy(conf_seq[np.newaxis]).unsqueeze(0)                 # (1,1,T,V)
        return xy_t, conf_t

# ── Angles ────────────────────────────────────────────────────────────────────
def angle_between(a, b, c):
    a, b, c = np.array(a, float), np.array(b, float), np.array(c, float)
    ba, bc  = a - b, c - b
    cos_a   = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-8)
    return math.degrees(math.acos(np.clip(cos_a, -1., 1.)))

def torso_incline_angle(j):
    """
    Returns torso incline relative to horizontal.

    0°   = perfectly horizontal
    90°  = perfectly vertical

    Elevated push-ups usually sit around 25-50°.
    """

    sho_mid = (np.array(j[L_SHO]) + np.array(j[R_SHO])) / 2.0
    hip_mid = (np.array(j[L_HIP]) + np.array(j[R_HIP])) / 2.0

    vec = sho_mid - hip_mid

    angle = abs(math.degrees(math.atan2(vec[1], vec[0])))

    # convert to incline-from-horizontal
    return min(angle, abs(180 - angle))

def _wrist_above_shoulder(wrist, sho_l, sho_r):
    """
    Signed vertical offset of wrist above the shoulder midpoint, normalised by
    shoulder width so it is scale-invariant.

    When lying on a bench the camera typically views from the side or slight angle.
    Pressing UP raises the wrist away from the chest toward the ceiling → positive.
    Lowering the bar brings the wrist back to shoulder/chest level → near zero or negative.

    Typical ranges (camera ~side-on):
      Bar pressed up (arms extended): +0.8 to +1.8
      Bar at chest  (arms bent):      -0.2 to +0.3

    In image coords Y increases downward, so wrist ABOVE shoulder → wrist.y < sho_mid.y
    → raw delta is negative → we negate so "up" = positive.
    """
    sho_l   = np.array(sho_l, float)
    sho_r   = np.array(sho_r, float)
    wrist   = np.array(wrist, float)
    sho_mid = (sho_l + sho_r) / 2.0
    sho_w   = np.linalg.norm(sho_r - sho_l)
    if sho_w < 1e-3:
        return 0.0
    # positive = wrist higher in image (smaller y) than shoulder midpoint
    return float((sho_mid[1] - wrist[1]) / sho_w)

def _wrist_chest_dist(wrist, sho_l, sho_r):
    """Legacy Euclidean distance — kept for any callers; not used for bench reps."""
    sho_l, sho_r, wrist = np.array(sho_l, float), np.array(sho_r, float), np.array(wrist, float)
    chest = (sho_l + sho_r) / 2.0
    sho_width = np.linalg.norm(sho_r - sho_l)
    if sho_width < 1e-3:
        return 0.0
    return float(np.linalg.norm(wrist - chest) / sho_width)

def wrist_hip_offset(wrist, hip_l, hip_r, shoulder_l, shoulder_r):
    """
    Returns the signed vertical offset of the wrist relative to the hip midpoint,
    normalised by torso height (hip-mid → shoulder-mid) so it is scale-invariant.

    In image coordinates Y increases downward, so:
      wrist ABOVE hip → negative raw offset → we negate so result is POSITIVE when up.

    Typical jump-rope range (forward swing style, wrists near hips):
      Top of swing  → ~+0.3 to +0.6  (wrist level with or above hip)
      Bottom of swing → ~-0.1 to -0.3 (wrist below hip)
    """
    hip_mid = (np.array(hip_l, float) + np.array(hip_r, float)) / 2.0
    sho_mid = (np.array(shoulder_l, float) + np.array(shoulder_r, float)) / 2.0
    torso_h = np.linalg.norm(sho_mid - hip_mid)
    if torso_h < 1e-3:
        return 0.0
    # positive = wrist higher than hip (image y smaller)
    return float((hip_mid[1] - np.array(wrist, float)[1]) / torso_h)

def get_angles(j):
    return {
        'left_elbow':     angle_between(j[L_SHO], j[L_ELB], j[L_WRI]),
        'right_elbow':    angle_between(j[R_SHO], j[R_ELB], j[R_WRI]),
        'left_knee':      angle_between(j[L_HIP], j[L_KNE], j[L_ANK]),
        'right_knee':     angle_between(j[R_HIP], j[R_KNE], j[R_ANK]),
        'left_hip':       angle_between(j[L_SHO], j[L_HIP], j[L_KNE]),
        'right_hip':      angle_between(j[R_SHO], j[R_HIP], j[R_KNE]),
        'left_shoulder':  angle_between(j[L_ELB], j[L_SHO], j[L_HIP]),
        'right_shoulder': angle_between(j[R_ELB], j[R_SHO], j[R_HIP]),
        # Wrist-hip vertical offset (normalised by torso height) — primary signal
        # for jump rope.  Positive = wrist above hip midpoint, negative = below.
        # Typical swing: top ~ +0.35, bottom ~ -0.15  →  ~0.50 total excursion.
        'left_wrist_hip_off':  wrist_hip_offset(j[L_WRI], j[L_HIP], j[R_HIP], j[L_SHO], j[R_SHO]),
        'right_wrist_hip_off': wrist_hip_offset(j[R_WRI], j[L_HIP], j[R_HIP], j[L_SHO], j[R_SHO]),
        # Wrist-chest distance (normalised by shoulder width) — kept for display.
        'left_wrist_chest':  _wrist_chest_dist(j[L_WRI], j[L_SHO], j[R_SHO]),
        'right_wrist_chest': _wrist_chest_dist(j[R_WRI], j[L_SHO], j[R_SHO]),
        # Wrist height above shoulder midpoint (normalised by shoulder width) —
        # PRIMARY signal for bench press rep counting.
        # Pressing up → wrist rises above shoulder → positive and increasing.
        # Lowering bar → wrist drops back to shoulder level → near zero / negative.
        'left_wrist_above_sho':  _wrist_above_shoulder(j[L_WRI], j[L_SHO], j[R_SHO]),
        'right_wrist_above_sho': _wrist_above_shoulder(j[R_WRI], j[L_SHO], j[R_SHO]),
        # Hip displacement from frame-start baseline — used as bench press stillness gate.
        # Computed as raw pixel Y of hip midpoint; gate logic normalises per-session.
        '_hip_mid_y': float((j[L_HIP][1] + j[R_HIP][1]) / 2.0),
        '_torso_incline': torso_incline_angle(j),
    }

# ── Ideal angle ranges for form quality display ───────────────────────────────
# These represent the "good form" window at the PEAK EFFORT position.
#
# Squat:    peak effort = bottom of squat → knees/hips compressed (small angles)
# Sit-up:   peak effort = fully upright   → hip compressed (small angle)
# Push-up:  peak effort = chest to floor  → elbows compressed (small angles)
#
# The form score penalises deviation from these ranges.
IDEAL_ANGLES = {
    'squats':        {'left_knee':  (70, 115),  'right_knee':  (70, 115),
                      'left_hip':   (50, 100),  'right_hip':   (50, 100)},
    'push_ups':      {'left_elbow': (70, 100),  'right_elbow': (70, 100),
                      'left_shoulder': (40, 80), 'right_shoulder': (40, 80)},
    'bench_press':   {'left_wrist_above_sho':  (0.6, 2.0),
                      'right_wrist_above_sho': (0.6, 2.0)},
    'pull_ups':      {'left_elbow': (70, 100),  'right_elbow': (70, 100),
                      'left_shoulder': (150, 180), 'right_shoulder': (150, 180)},
    'sit_ups':       {'left_hip':   (40, 80),   'right_hip':   (40, 80)},
    'jumping_jacks': {'left_shoulder': (150, 180), 'right_shoulder': (150, 180),
                      'left_knee':  (160, 180),  'right_knee':  (160, 180)},
    'jump_rope':     {'left_knee':            (140, 175), 'right_knee':            (140, 175),
                      'left_wrist_hip_off':  (0.1, 0.6), 'right_wrist_hip_off':  (0.1, 0.6)},
}

# Joints whose values are dimensionless ratios (not degrees); need a different
# deviation scale for the form-score calculation.
_RATIO_JOINTS = {'left_wrist_hip_off',    'right_wrist_hip_off',
                 'left_wrist_chest',      'right_wrist_chest',
                 'left_wrist_above_sho',  'right_wrist_above_sho'}
_RATIO_SCALE  = 0.5   # ratio deviation of 0.5 → form score of 0  (≈ 45° equivalent)

def get_angle_deviations(exercise, angles):
    """Returns per-joint deviation from ideal range and an overall form score 0-100."""
    if exercise not in IDEAL_ANGLES:
        return {}, 100.0
    deviations = {}
    total_dev_norm = 0.0   # accumulated deviation normalised to a 0-100 penalty scale
    count          = 0
    for joint, (lo, hi) in IDEAL_ANGLES[exercise].items():
        if joint not in angles:
            continue
        a   = angles[joint]
        dev = max(0.0, lo - a) if a < lo else (max(0.0, a - hi) if a > hi else 0.0)
        deviations[joint] = dev
        # Normalise: for angles use 45° scale; for ratio joints use 0.5 scale
        scale = _RATIO_SCALE if joint in _RATIO_JOINTS else 45.0
        total_dev_norm += dev / scale
        count          += 1
    form_score = max(0.0, 100.0 - (total_dev_norm / max(count, 1)) * 100.0)
    return deviations, form_score

# ── Rep counter ───────────────────────────────────────────────────────────────
# Angle conventions:
#
#   reversed=False  (squats, push-ups, bench, jacks, rope):
#     small angle = STATE_DOWN (effort position), large angle = STATE_UP (rest position)
#     Rep counted on DOWN → UP transition.
#
#   reversed=True   (pull-ups, sit-ups):
#     large angle = STATE_DOWN (rest/hang/lying), small angle = STATE_UP (effort)
#     Rep counted on DOWN → UP transition.
#
# Thresholds are set so there is a clear dead-zone between down and up —
# the angle must cross the threshold meaningfully, not just flicker around it.
REP_RULES = {
    # SQUAT — knee angle.  Standing ~165-180°, bottom ~70-110°.
    'squats':        dict(keys=['left_knee','right_knee'],
                          down=110,   # must bend past 110° to register bottom of squat
                          up=160,     # must straighten past 160° to register standing
                          reversed=False),

    # PUSH-UP — elbow angle.  Arms extended ~160-175°, chest-to-floor ~70-85°.
    'push_ups':      dict(keys=['left_elbow','right_elbow'],
                          down=85,    # tighter — chest must actually approach floor
                          up=145,     # arms reasonably extended at top
                          reversed=False),

    # BENCH PRESS — wrist height above shoulder midpoint (normalised by shoulder width).
    #
    #   When lying on a bench the hips barely move; all work is done by the arms.
    #   The wrist rises above the shoulder when pressing up, and drops back to
    #   shoulder/chest level when lowering — a clear, camera-angle-robust signal.
    #
    #   Euclidean wrist-chest distance (previous signal) collapses when the camera
    #   is above/front-on because pressing "up" toward the ceiling looks like no
    #   horizontal movement. Vertical offset relative to the shoulder is unambiguous.
    #
    #   Typical normalised ranges (shoulder-width units):
    #     Bar at chest  (down): -0.1 to +0.25   → STATE_DOWN
    #     Arms extended (up):    +0.7 to +1.6   → STATE_UP
    #
    #   hip_still_thresh: secondary gate — if hips are moving more than this many
    #   pixels per frame (raw), the person is not lying still → not a bench press rep.
    #   Checked in RepCounter.update().
    'bench_press':   dict(keys=['left_wrist_above_sho', 'right_wrist_above_sho'],
                          down=0.25,   # wrist near/below shoulder → bar at chest
                          up=0.70,     # wrist well above shoulder → bar pressed up
                          reversed=False,
                          hip_still_thresh=30.0),  # max hip pixel movement allowed

    # PULL-UP — elbow angle, reversed.
    #   Dead hang (down):  ~160-175°  → STATE_DOWN
    #   Chin over bar (up): ~50-75°   → STATE_UP
    #   Tightened heavily — previous (90,150) caused phantom reps on any arm movement.
    'pull_ups':      dict(keys=['left_elbow','right_elbow'],
                          down=145,   # must be nearly straight (hanging) to reset state
                          up=75,      # must be deeply bent (chin up) to count rep
                          reversed=True),

    # SIT-UP — hip angle (shoulder→hip→knee), reversed.
    #   Lying flat (down): ~150-175°  → STATE_DOWN
    #   Sitting up (up):   ~55-70°    → STATE_UP
    'sit_ups':       dict(keys=['left_hip','right_hip'],
                          down=145,   # must be properly flat to reset (was 130 — too easy)
                          up=70,      # more reachable sit-up height than previous 60°
                          reversed=True),

    # JUMPING JACK — shoulder angle.
    #   Arms at sides (down): ~20-40°   → STATE_DOWN
    #   Arms overhead (up):   ~155-175° → STATE_UP
    #   1 rep = arms-down → arms-up (standard gym convention).
    'jumping_jacks': dict(keys=['left_shoulder','right_shoulder'],
                          down=40,
                          up=130,
                          reversed=False),

    # JUMP ROPE — wrist-hip vertical offset (normalised, primary) + knee gate.
    #
    #   During jump rope the wrist circles near the hips, so the forearm stays
    #   roughly horizontal the whole time — angular measures are tiny (~20-40°)
    #   and unreliable.  Instead we track the signed vertical position of each
    #   wrist relative to the hip midpoint, normalised by torso height:
    #
    #     positive value = wrist ABOVE hip midpoint  → STATE_UP   (rope overhead)
    #     negative value = wrist BELOW hip midpoint  → STATE_DOWN (rope at bottom)
    #
    #   Typical excursion during a normal forward-swing jump rope:
    #     top of swing  → offset ~ +0.30 to +0.55
    #     bottom of swing → offset ~ -0.05 to -0.20
    #
    #   We use a symmetric dead-zone (+0.20 / -0.05) so there is clear hysteresis
    #   and single-frame noise cannot flip the state.
    #
    #   knee_down_thresh: secondary gate — knees must show at least a slight bend
    #   (avg knee angle < 172°) when the wrist reaches STATE_UP, preventing
    #   accidental arm raises while standing still from counting as reps.
    #
    #   NOTE: because the signal is an offset (not an angle), we set reversed=False
    #   and rely on the fact that larger positive = up, smaller/negative = down —
    #   exactly matching the normal (reversed=False) convention in RepCounter.
    'jump_rope':     dict(keys=['left_wrist_hip_off', 'right_wrist_hip_off'],
                          # Observed range from debug: +0.12 (bottom) to +0.55 (top).
                          # Wrists stay above hips throughout — never go negative.
                          # Threshold within the observed swing range, not around zero.
                          down=0.20,   # wrist near bottom of swing
                          up=0.38,     # wrist near top of swing
                          reversed=False),
}
STATE_UP, STATE_DOWN, STATE_INIT = 'up', 'down', 'init'

class RepCounter:
    def __init__(self):
        self.reps              = {ex: 0 for ex in REP_RULES}
        self.states            = {ex: STATE_INIT for ex in REP_RULES}
        self.stable_conf_count = {ex: 0 for ex in REP_RULES}
        self._counting_enabled = {ex: False for ex in REP_RULES}
        self._prev_hip_y       = None   # for bench press hip-stillness gate

    def reset(self, exercise=None):
        for ex in ([exercise] if exercise else list(REP_RULES)):
            self.reps[ex]              = 0
            self.states[ex]            = STATE_INIT
            self.stable_conf_count[ex] = 0
            self._counting_enabled[ex] = False
        self._prev_hip_y = None

    def update(self, exercise, angles, confidence=1.0):
        if exercise not in REP_RULES:
            return 0

        # Gate: require N consecutive high-confidence frames before reps count
        if confidence >= MIN_CONF_FOR_REPS:
            self.stable_conf_count[exercise] = min(
                self.stable_conf_count[exercise] + 1, STABLE_FRAMES_NEEDED)
        else:
            self.stable_conf_count[exercise] = max(
                self.stable_conf_count[exercise] - 1, 0)
        self._counting_enabled[exercise] = (
            self.stable_conf_count[exercise] >= STABLE_FRAMES_NEEDED)

        rule      = REP_RULES[exercise].copy()
        avg_angle = np.mean([angles[k] for k in rule['keys'] if k in angles])
        state     = self.states[exercise]
        rev       = rule.get('reversed', False)

        # Elevated push-up adjustment
        if exercise == 'push_ups' and '_torso_incline' in angles:
            incline = angles['_torso_incline']

            # Elevated push-ups:
            # body more vertical -> elbows compress less
            if incline > 25:
                rule['down'] = 100
                rule['up']   = 135

        # Normal  (reversed=False): small angle = down, large angle = up
        # Reversed (reversed=True): large angle = down, small angle = up
        # A rep is always counted on the DOWN → UP transition.
        at_up   = avg_angle < rule['up']   if rev else avg_angle > rule['up']
        at_down = avg_angle > rule['down'] if rev else avg_angle < rule['down']

        # Jump rope secondary gate: knees must show at least a slight bend when
        # the forearm reaches UP position, filtering accidental arm raises while
        # standing still (knees fully extended ≈ 175-180° → no jump → no rep).
        knee_gate_ok = True
        if 'knee_down_thresh' in rule and at_up:
            thresh = rule['knee_down_thresh']
            avg_knee = np.mean([angles[k] for k in ('left_knee', 'right_knee')
                                if k in angles] or [180.0])
            knee_gate_ok = avg_knee < thresh   # bent knee means angle < threshold

        if exercise == 'jump_rope':
            print(f'[jump_rope] avg_wrist_hip={avg_angle:+.3f}  at_up={at_up}  at_down={at_down}  '
                  f'state={state}  enabled={self._counting_enabled[exercise]}  reps={self.reps[exercise]}')

        # Bench press hip-stillness gate: reject rep if hips moved too much this frame.
        # Raw hip_mid_y is stored in angles under '_hip_mid_y' by get_angles().
        hip_still_ok = True
        if exercise == 'bench_press' and 'hip_still_thresh' in rule:
            cur_hip_y = angles.get('_hip_mid_y', None)
            if cur_hip_y is not None:
                if self._prev_hip_y is not None:
                    hip_delta = abs(cur_hip_y - self._prev_hip_y)
                    hip_still_ok = hip_delta < rule['hip_still_thresh']
                self._prev_hip_y = cur_hip_y
            print(f'[bench] avg_wrist_above_sho={avg_angle:+.3f}  at_up={at_up}  at_down={at_down}  '
                  f'state={state}  hip_still={hip_still_ok}  enabled={self._counting_enabled[exercise]}  '
                  f'reps={self.reps[exercise]}')

        if   state == STATE_INIT and at_up:   self.states[exercise] = STATE_UP
        elif state == STATE_INIT:             self.states[exercise] = STATE_DOWN
        elif state == STATE_UP  and at_down:  self.states[exercise] = STATE_DOWN
        elif state == STATE_DOWN and at_up and knee_gate_ok and hip_still_ok:
            self.states[exercise] = STATE_UP
            if self._counting_enabled[exercise]:
                self.reps[exercise] += 1

        return self.reps[exercise]

# ── Rule-based form engine ──────────────────────────────────────────────

FORM_RULES = {

    'squats': [

        {
            'name': 'depth',
            'severity': 'warn',
            'message': 'Go deeper — knees past 90°',
            'condition': lambda a, s:
                s == STATE_DOWN and
                (a['left_knee'] + a['right_knee']) / 2 > 115
        },

        {
            'name': 'forward_lean',
            'severity': 'warn',
            'message': 'Keep chest more upright',
            'condition': lambda a, s:
                (a['left_hip'] + a['right_hip']) / 2 < 45
        },

        {
            'name': 'good',
            'severity': 'good',
            'message': 'Good squat form',
            'condition': lambda a, s: True
        }
    ],

    'push_ups': [

        {
            'name': 'depth',
            'severity': 'warn',
            'message': 'Lower chest closer to floor',
            'condition': lambda a, s:
                s == STATE_DOWN and
                (a['left_elbow'] + a['right_elbow']) / 2 > 100
        },

        {
            'name': 'lockout',
            'severity': 'warn',
            'message': 'Extend arms fully at the top',
            'condition': lambda a, s:
                s == STATE_UP and
                (a['left_elbow'] + a['right_elbow']) / 2 < 145
        },

        {
            'name': 'good',
            'severity': 'good',
            'message': 'Strong push-up reps',
            'condition': lambda a, s: True
        }
    ],

    'bench_press': [

        {
            'name': 'depth',
            'severity': 'warn',
            'message': 'Lower bar fully to chest',
            'condition': lambda a, s:
                s == STATE_DOWN and
                (
                    a['left_wrist_above_sho'] +
                    a['right_wrist_above_sho']
                ) / 2 > 0.35
        },

        {
            'name': 'lockout',
            'severity': 'warn',
            'message': 'Press higher at the top',
            'condition': lambda a, s:
                s == STATE_UP and
                (
                    a['left_wrist_above_sho'] +
                    a['right_wrist_above_sho']
                ) / 2 < 0.7
        },

        {
            'name': 'good',
            'severity': 'good',
            'message': 'Controlled bench press',
            'condition': lambda a, s: True
        }
    ],

    'pull_ups': [

        {
            'name': 'range',
            'severity': 'warn',
            'message': 'Pull higher — chin above bar',
            'condition': lambda a, s:
                s == STATE_UP and
                (a['left_elbow'] + a['right_elbow']) / 2 > 100
        },

        {
            'name': 'full_extension',
            'severity': 'warn',
            'message': 'Fully extend arms at bottom',
            'condition': lambda a, s:
                s == STATE_DOWN and
                (a['left_elbow'] + a['right_elbow']) / 2 < 145
        },

        {
            'name': 'good',
            'severity': 'good',
            'message': 'Good pull-up control',
            'condition': lambda a, s: True
        }
    ],

    'sit_ups': [

        {
            'name': 'height',
            'severity': 'warn',
            'message': 'Sit up higher',
            'condition': lambda a, s:
                s == STATE_UP and
                (a['left_hip'] + a['right_hip']) / 2 > 80
        },

        {
            'name': 'good',
            'severity': 'good',
            'message': 'Good sit-up range',
            'condition': lambda a, s: True
        }
    ],

    'jumping_jacks': [

        {
            'name': 'arms',
            'severity': 'warn',
            'message': 'Raise arms fully overhead',
            'condition': lambda a, s:
                s == STATE_UP and
                (a['left_shoulder'] + a['right_shoulder']) / 2 < 130
        },

        {
            'name': 'good',
            'severity': 'good',
            'message': 'Full jumping jack motion',
            'condition': lambda a, s: True
        }
    ],

    'jump_rope': [

        {
            'name': 'good',
            'severity': 'good',
            'message': 'Good rhythm',
            'condition': lambda a, s: True
        }
    ]
}

# ── Form feedback ─────────────────────────────────────────────────────────────
def get_form_feedback(exercise, angles, rep_state):
    if exercise not in FORM_RULES:
        return [('good', 'Keep going!')]

    feedback = []

    for rule in FORM_RULES[exercise]:

        try:
            if rule['condition'](angles, rep_state):
                feedback.append(
                    (rule['severity'], rule['message'])
                )

        except Exception as e:
            print(f'[rule_engine] {exercise}:{rule["name"]} failed -> {e}')

    # prevent "good" message spam when warnings exist
    warnings_exist = any(sev != 'good' for sev, _ in feedback)

    if warnings_exist:
        feedback = [
            (sev, msg)
            for sev, msg in feedback
            if sev != 'good'
        ]

    return feedback[:3]

# ── Session accuracy tracker ──────────────────────────────────────────────────
class SessionAccuracyTracker:
    def __init__(self, ground_truth=None, ground_truth_reps=None):
        self.ground_truth      = ground_truth
        self.ground_truth_reps = ground_truth_reps
        self.n_predictions     = 0
        self.n_correct         = 0
        self.total_conf        = 0.0
        self.class_counts      = collections.Counter()

    def update(self, predicted, confidence):
        self.n_predictions += 1
        self.total_conf    += confidence
        self.class_counts[predicted] += 1
        if self.ground_truth and predicted == self.ground_truth:
            self.n_correct += 1

    def accuracy(self):
        if self.n_predictions == 0:
            return 0.0, 'no predictions yet'
        if self.ground_truth:
            return self.n_correct / self.n_predictions, f"accuracy vs '{self.ground_truth}'"
        return self.total_conf / self.n_predictions, 'mean confidence (proxy)'

    def rep_accuracy(self, counted_reps):
        if self.ground_truth_reps is None or self.ground_truth_reps == 0:
            return None
        error = abs(counted_reps - self.ground_truth_reps)
        pct   = max(0.0, 1.0 - error / self.ground_truth_reps)
        return pct, error

    def summary(self, final_reps=None):
        acc, label = self.accuracy()
        lines = [f'\n{"="*50}', f'  {label}',
                 f'  Score       : {acc*100:.1f}%',
                 f'  Predictions : {self.n_predictions}']
        if self.ground_truth:
            lines.append(f'  Correct     : {self.n_correct}')
        if self.ground_truth_reps is not None and final_reps is not None:
            rep_acc = self.rep_accuracy(final_reps)
            if rep_acc:
                pct, err = rep_acc
                lines.append(f'  Rep accuracy: {pct*100:.1f}%  '
                             f'(counted {final_reps}, expected {self.ground_truth_reps}, '
                             f'off by {err})')
        lines.append('  Per-class:')
        for ex, cnt in self.class_counts.most_common():
            lines.append(f'    {ex:<16}: {cnt} predictions')
        lines.append('='*50)
        return '\n'.join(lines)

# ── Overlay helpers ───────────────────────────────────────────────────────────
SEVERITY_COLORS = {'good': (0,220,0), 'warn': (0,165,255), 'error': (0,0,220)}
PENN_DRAW_EDGES = [(0,1),(0,2),(1,3),(3,5),(2,4),(4,6),
                   (1,7),(2,8),(7,8),(7,9),(9,11),(8,10),(10,12)]

def draw_skeleton(frame, joints_px, conf):
    for i, j in PENN_DRAW_EDGES:
        if conf[i] < 0.15 or conf[j] < 0.15:
            continue
        cv2.line(frame, tuple(joints_px[i].astype(int)), tuple(joints_px[j].astype(int)), (0,255,255), 2)
    for idx, pt in enumerate(joints_px):
        if conf[idx] < 0.15:
            continue
        cv2.circle(frame, tuple(pt.astype(int)), 4, (255,128,0), -1)

def draw_overlay(frame, exercise, confidence, reps, feedback, rep_state, session_acc):
    panel_h = 165 + len(feedback) * 28
    overlay = frame.copy()
    cv2.rectangle(overlay, (0,0), (360, panel_h), (20,20,20), -1)
    cv2.addWeighted(overlay, 0.72, frame, 0.28, 0, frame)
    label = exercise.replace('_',' ').upper() if exercise else 'DETECTING...'
    cv2.putText(frame, label,                         (10, 30),  cv2.FONT_HERSHEY_SIMPLEX, 0.9,  (255,255,255), 2)
    conf_pct  = int(confidence * 100)
    bar_color = (0,200,0) if confidence > CONFIDENCE_THRESHOLD else (0,100,220)
    cv2.rectangle(frame, (10,40), (10 + conf_pct*2, 58), bar_color, -1)
    cv2.putText(frame, f'Conf {conf_pct}%',           (10, 72),  cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200,200,200), 1)
    cv2.putText(frame, f'Reps: {reps}  [{rep_state}]',(10, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255,230,0),  2)
    acc_val, _ = session_acc
    acc_color  = (0,220,0) if acc_val >= 0.7 else (0,165,255)
    cv2.putText(frame, f'Session: {acc_val*100:.1f}%',(10, 130), cv2.FONT_HERSHEY_SIMPLEX, 0.6,  acc_color,    2)
    for i, (sev, msg) in enumerate(feedback):
        cv2.putText(frame, f'  {msg}', (10, 165 + i*28), cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                    SEVERITY_COLORS.get(sev, (200,200,200)), 1)

def draw_right_panel(frame, exercise, angles, deviations,
                     form_score, reps,
                     confidence=0.0,
                     rep_accuracy=None,
                     stability=0):
    if not exercise or exercise not in IDEAL_ANGLES:
        return
    panel_w = 310
    h, w    = frame.shape[:2]
    x0      = w - panel_w - 10
    y0      = 10
    relevant_joints = list(IDEAL_ANGLES[exercise].keys())
    # new: panel_h = 160 + len(relevant_joints) * 26 # old: panel_h = 50 + len(relevant_joints) * 26 + 40
    metrics_block_h = 90
    panel_h = 190 + len(relevant_joints) * 26 + metrics_block_h
    overlay = frame.copy()
    cv2.rectangle(overlay, (x0 - 8, y0), (x0 + panel_w, y0 + panel_h), (20,20,20), -1)
    cv2.addWeighted(overlay, 0.55, frame, 0.45, 0, frame)
    cv2.putText(frame, 'JOINT ANGLES', (x0, y0 + 22), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (200,200,255), 2)
    for i, joint in enumerate(relevant_joints):
        y         = y0 + 48 + i * 26
        angle_val = angles.get(joint, 0.0)
        dev       = deviations.get(joint, 0.0)
        color     = (0,220,0) if dev == 0.0 else (0,165,255) if dev < 15 else (0,60,220)
        label     = joint.replace('_', ' ')
        is_ratio = joint in _RATIO_JOINTS
        val_str  = f'{angle_val:+.3f}    ' if is_ratio else f'{angle_val:5.1f}deg'
        cv2.putText(frame, f'{label:<16} {val_str}', (x0, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.48, color, 1)
        if dev > 0.5:
            err_txt = f'Err:{dev:.1f}'
            if joint in _RATIO_JOINTS:
                err_txt = f'Err:{dev:.2f}'
            cv2.putText(frame, err_txt, (x0 + 215, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.42, color, 1)
    y_bar     = y0 + panel_h - 28
    bar_max   = panel_w - 10
    bar_fill  = int(bar_max * form_score / 100.0)
    bar_color = (0,220,0) if form_score >= 75 else (0,165,255) if form_score >= 50 else (0,60,220)
    cv2.rectangle(frame, (x0, y_bar), (x0 + bar_max,  y_bar + 14), (60,60,60), -1)
    cv2.rectangle(frame, (x0, y_bar), (x0 + bar_fill, y_bar + 14), bar_color,  -1)
    cv2.putText(frame, f'Form: {form_score:.0f}%  Reps: {reps}', (x0, y_bar - 6),
                cv2.FONT_HERSHEY_SIMPLEX, 0.52, (220,220,220), 1)
    
    # ── Bottom-right metrics HUD ─────────────────────────────────
    hud_w = 300
    hud_h = 105 if rep_accuracy is not None else 78

    hud_x = w - hud_w - 15
    hud_y = h - hud_h - 15

    overlay = frame.copy()

    cv2.rectangle(
        overlay,
        (hud_x, hud_y),
        (hud_x + hud_w, hud_y + hud_h + 10),
        (15, 15, 15),
        -1
    )

    cv2.addWeighted(overlay, 0.78, frame, 0.22, 0, frame)

    cv2.putText(
        frame,
        'LIVE METRICS',
        (hud_x + 12, hud_y + 24),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.58,
        (230,230,230),
        1
    )

    # Classification confidence
    conf_color = (
        (0,220,0) if confidence >= 0.75 else
        (0,165,255) if confidence >= 0.50 else
        (0,0,220)
    )

    cv2.putText(
        frame,
        'Cls Confidence',
        (hud_x + 12, hud_y + 50),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (210,210,210),
        1
    )

    cv2.putText(
        frame,
        f'{confidence*100:.1f}%',
        (hud_x + 185, hud_y + 50),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.62,
        conf_color,
        1
    )

    # Tracking stability
    stab_color = (
        (0,220,0) if stability >= STABLE_FRAMES_NEEDED else
        (0,165,255)
    )

    cv2.putText(
        frame,
        'Tracking Stability',
        (hud_x + 12, hud_y + 76),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (210,210,210),
        1
    )

    cv2.putText(
        frame,
        f'{stability}/{STABLE_FRAMES_NEEDED}',
        (hud_x + 185, hud_y + 76),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.62,
        stab_color,
        1
    )

    # Rep accuracy
    if rep_accuracy is not None:

        rep_color = (
            (0,220,0) if rep_accuracy >= 0.90 else
            (0,165,255) if rep_accuracy >= 0.70 else
            (0,0,220)
        )

        cv2.putText(
            frame,
            'Rep Accuracy',
            (hud_x + 12, hud_y + 102),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (210,210,210),
            1
        )

        cv2.putText(
            frame,
            f'{rep_accuracy*100:.1f}%',
            (hud_x + 185, hud_y + 102),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.62,
            rep_color,
            1
        )

def draw_visibility_warning(frame, ankle_vis, current_exercise=None):
    h = frame.shape[0]

    # Exercises that need lower-body visibility
    lower_body_required = {
        'squats',
        'jumping_jacks',
        'jump_rope'
    }

    # Push-ups / bench press often crop ankles naturally
    if current_exercise not in lower_body_required:
        return

    if ankle_vis < MIN_ANKLE_VIS:
        cv2.putText(frame,
                    'Step back — ankles not visible!',
                    (10, h - 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.75,
                    (0,50,255),
                    2)

    elif ankle_vis < 0.4:
        cv2.putText(frame,
                    'Move back slightly for best results',
                    (10, h - 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0,165,255),
                    1)

# ── Classify ──────────────────────────────────────────────────────────────────
@torch.no_grad()
def classify(model, buffer):
    # Wait for at least half a clip before classifying — avoids garbage predictions
    # on an almost-empty buffer padded with zeros.
    if len(buffer) < CLIP_LEN // 2:
        return None, 0.0
    xy, conf = buffer.get_tensors()
    logits   = model(xy.to(DEVICE), conf.to(DEVICE))
    probs    = F.softmax(logits, dim=-1)[0]

    # Bias correction for elevated / horizontal push-ups.
    # Push-ups often get confused with jumping_jacks when
    # lower body is occluded.

    if len(buffer) > 0:

        latest = list(buffer.buffer)[-1]['xy']

        sho_mid = (latest[L_SHO] + latest[R_SHO]) / 2.0
        hip_mid = (latest[L_HIP] + latest[R_HIP]) / 2.0

        dx = abs(sho_mid[0] - hip_mid[0])
        dy = abs(sho_mid[1] - hip_mid[1])

        horizontal_torso = dx > dy

        if horizontal_torso:

            push_idx = [k for k,v in LABEL_MAP.items() if v == 'push_ups'][0]
            jack_idx = [k for k,v in LABEL_MAP.items() if v == 'jumping_jacks'][0]

            # boost push-ups slightly
            probs[push_idx] *= 1.35

            # suppress jumping jacks slightly
            probs[jack_idx] *= 0.70

            # renormalize
            probs = probs / probs.sum()

    print('\nPredictions:')
    for i, p in enumerate(probs):
        print(f'  {LABEL_MAP[i]:15s}: {p.item():.3f}')

    cls = int(probs.argmax().item())
    return cls, probs[cls].item()

def geometry_plausible(label, joints_px, conf):
    # Hard geometric rules rejecting physically impossible label assignments.
    # Returns False if the predicted label contradicts the current skeleton geometry.
    #
    # bench_press: person is lying flat — hip Y and shoulder Y should be close.
    #   If hips are clearly below shoulders (upright posture), reject.
    # pull_ups: arms must be raised — wrists must be above shoulders.
    #   If wrists are at or below shoulder level, reject.
    j = joints_px
    c = conf

    sho_vis = (c[L_SHO] + c[R_SHO]) / 2.0
    hip_vis = (c[L_HIP] + c[R_HIP]) / 2.0
    wri_vis = (c[L_WRI] + c[R_WRI]) / 2.0

    sho_mid_y = (j[L_SHO][1] + j[R_SHO][1]) / 2.0
    hip_mid_y = (j[L_HIP][1] + j[R_HIP][1]) / 2.0
    wri_mid_y = (j[L_WRI][1] + j[R_WRI][1]) / 2.0
    torso_h   = abs(hip_mid_y - sho_mid_y) + 1e-3

    if label == 'bench_press':
        if sho_vis < 0.3 or hip_vis < 0.3:
            return True  # cannot tell, do not reject
        # Lying flat: hip_y ≈ sho_y. Standing/sitting: hip_y >> sho_y (image Y down).
        # Reject if hips are substantially below shoulders — that is an upright posture.
        if hip_mid_y > sho_mid_y + 0.25 * torso_h:
            print(f'[geometry] rejecting bench_press — upright posture detected '                  f'(hip_y={hip_mid_y:.0f}, sho_y={sho_mid_y:.0f}, torso={torso_h:.0f})')
            return False

    if label == 'pull_ups':
        if sho_vis < 0.3 or wri_vis < 0.3:
            return True  # cannot tell, do not reject
        # Pull-up: wrists must be above (lower image-Y than) shoulders.
        # Reject if wrists are at or below shoulder level.
        if wri_mid_y > sho_mid_y - 0.05 * torso_h:
            print(f'[geometry] rejecting pull_ups — wrists not above shoulders '                  f'(wri_y={wri_mid_y:.0f}, sho_y={sho_mid_y:.0f})')
            return False

    if label == 'jumping_jacks':

        required = [L_SHO, R_SHO, L_HIP, R_HIP]

        if all(c[j] >= 0.2 for j in required):

            sho_mid = (j[L_SHO] + j[R_SHO]) / 2.0
            hip_mid = (j[L_HIP] + j[R_HIP]) / 2.0

            dx = abs(sho_mid[0] - hip_mid[0])
            dy = abs(sho_mid[1] - hip_mid[1])

            # Jumping jacks should be upright.
            # If torso is more horizontal than vertical,
            # reject the prediction.
            if dx > dy:
                print('[geometry] rejecting jumping_jacks — horizontal torso detected')
                return False

    return True

def pushup_geometry_hint(joints_px, conf):
    """
    Detects push-up-like posture using simple geometry.

    Conditions:
    - torso roughly horizontal
    - shoulders low in frame
    - wrists near shoulder height
    """

    required = [L_SHO, R_SHO, L_WRI, R_WRI, L_HIP, R_HIP]

    if any(conf[j] < 0.2 for j in required):
        return False

    sho_mid = (joints_px[L_SHO] + joints_px[R_SHO]) / 2.0
    hip_mid = (joints_px[L_HIP] + joints_px[R_HIP]) / 2.0
    wri_mid = (joints_px[L_WRI] + joints_px[R_WRI]) / 2.0

    dx = abs(sho_mid[0] - hip_mid[0])
    dy = abs(sho_mid[1] - hip_mid[1])

    # torso horizontal
    horizontal_torso = dx > dy

    # wrists near shoulder level
    wrist_close = abs(wri_mid[1] - sho_mid[1]) < 120

    return horizontal_torso and wrist_close

# ── Joint extraction ──────────────────────────────────────────────────────────
def extract_joints(landmarks, frame_shape):
    h, w   = frame_shape[:2]
    joints = np.zeros((NUM_JOINTS, 3), dtype=np.float32)
    for mp_idx, penn_idx in MP_TO_PENN.items():
        lm  = landmarks.landmark[mp_idx]
        vis = lm.visibility
        # Zero out position entirely if visibility too low — model sees (0,0)
        # which is closer to trained padding than a hallucinated coordinate.
        if vis >= 0.15:
            joints[int(penn_idx)] = [lm.x * w, lm.y * h, vis]
        else:
            joints[int(penn_idx)] = [0.0, 0.0, 0.0]
    return joints

def normalise_skeleton(joints_px):
    """
    Hip-origin normalisation — matches PennAction training preprocessing.
    Hip midpoint is the origin; torso height (hip→shoulder) is the scale unit.

    In this space:
      - Hips are always at (0, 0)
      - Shoulders are at approximately (0, -1.0) when standing upright
      - Knees/ankles move relative to hips, preserving leg bend signal
    """
    hip_mid = (joints_px[L_HIP] + joints_px[R_HIP]) / 2.0
    sho_mid = (joints_px[L_SHO] + joints_px[R_SHO]) / 2.0
    torso_h = np.linalg.norm(sho_mid - hip_mid)
    if torso_h < 10.0:   # torso must be at least 10px — sanity check
        return np.zeros_like(joints_px)
    return ((joints_px - hip_mid) / torso_h).astype(np.float32)

# ── Main inference loop ───────────────────────────────────────────────────────
def load_model():
    model = STGCNFineTuned(num_classes=NUM_CLASSES, A3=A3_np).to(DEVICE)
    model.load_state_dict(torch.load(CHECKPOINT_PATH, map_location=DEVICE))
    model.eval()
    n = sum(p.numel() for p in model.parameters()) / 1e6
    print(f'Model loaded — {n:.2f}M params on {DEVICE}')
    return model

def run_inference(source=0, output_path=None, max_frames=None,
                      ground_truth=None, ground_truth_reps=None,
                      gui_state=None, stop_event=None):
    model   = load_model()
    mp_pose = mp.solutions.pose
    pose    = mp_pose.Pose(min_detection_confidence=0.5,
                           min_tracking_confidence=0.6,   # raised — reduces skeleton jumping
                           smooth_landmarks=True)          # MediaPipe temporal smoothing
    cap     = cv2.VideoCapture(source)

    WINDOW_W = 1280
    WINDOW_H = 720
    cv2.namedWindow('Kinexis', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('Kinexis', WINDOW_W, WINDOW_H)

    if not cap.isOpened():
        raise RuntimeError(f'Cannot open source: {source}')

    if gui_state is not None:
        gui_state.update(running=True)

    fps = cap.get(cv2.CAP_PROP_FPS) or 30

    writer = None
    if output_path:
        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
        writer = cv2.VideoWriter(output_path, cv2.VideoWriter.fourcc(*'mp4v'),
                                 fps, (WINDOW_W, WINDOW_H))

    buffer           = SlidingWindowBuffer(CLIP_LEN)
    rep_counter      = RepCounter()
    acc_tracker      = SessionAccuracyTracker(ground_truth=ground_truth,
                                              ground_truth_reps=ground_truth_reps)
    current_exercise = None
    current_conf     = 0.0
    # Smaller window → faster label switching without a sanity check as crutch.
    pred_history     = collections.deque(maxlen=10)
    feedback         = [('good', 'Warming up...')]
    deviations       = {}
    form_score       = 100.0
    frame_idx        = 0
    bad_frame_streak = 0

    print('Running — press Q to quit')
    if ground_truth:
        print(f'Supervised mode — ground truth: {ground_truth}')
    if ground_truth_reps:
        print(f'Expected reps: {ground_truth_reps}')

    try:
        while True:
            # ── GUI stop hook ─────────────────────────────────────────────────
            if stop_event is not None and stop_event.is_set():
                break
            ret, frame = cap.read()
            if not ret or (max_frames and frame_idx >= max_frames):
                break
            frame     = cv2.resize(frame, (WINDOW_W, WINDOW_H))
            frame_idx += 1

            rgb     = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = pose.process(rgb)

            if results.pose_landmarks:
                joints    = extract_joints(results.pose_landmarks, frame.shape)
                joints_px = joints[:, :2].copy()
                conf      = joints[:, 2]

                ankle_vis = (conf[L_ANK] + conf[R_ANK]) / 2.0
                knee_vis  = (conf[L_KNE] + conf[R_KNE]) / 2.0
                wrist_vis = (conf[L_WRI] + conf[R_WRI]) / 2.0
                elbow_vis = (conf[L_ELB] + conf[R_ELB]) / 2.0
                sho_vis   = (conf[L_SHO] + conf[R_SHO]) / 2.0

                # Floor / elevated exercises often lose ankle visibility.
                # Use upper-body gating for exercises dominated by arm motion.

                if current_exercise in ['bench_press', 'push_ups']:

                    frame_ok = (
                        wrist_vis >= 0.20 and
                        elbow_vis >= 0.20 and
                        sho_vis   >= 0.20
                    )

                else:
                    frame_ok = (
                        ankle_vis >= MIN_ANKLE_VIS and
                        knee_vis  >= MIN_KNEE_VIS
                    )

                xy_norm = normalise_skeleton(joints_px)

                # Skeleton stability check: if the normalised skeleton is all-zero
                # (torso too small / person too far), skip this frame entirely.
                # This suppresses the "jumping skeleton" when MediaPipe briefly
                # loses the person and snaps to a spurious detection.
                skeleton_valid = np.any(xy_norm != 0.0)

                if frame_ok and skeleton_valid:
                    bad_frame_streak = 0
                    buffer.push(xy_norm, conf)
                else:
                    bad_frame_streak += 1
                    # Do NOT flush the buffer on ankle loss — mid-squat the ankles
                    # often briefly leave frame. Wiping the buffer loses all temporal
                    # context right when the model needs it most. Just skip the frame.
                    print(f'[{frame_idx}] Skipping frame '
                          f'(ankle={ankle_vis:.2f}, knee={knee_vis:.2f}, '
                          f'skel_valid={skeleton_valid}, streak={bad_frame_streak})')

                if frame_idx % INFER_EVERY == 0:
                    cls, c = classify(model, buffer)
                    if cls is not None and c > CONFIDENCE_THRESHOLD:
                        pred = LABEL_MAP[cls]
                        # Heuristic override:
                        # horizontal torso + floor posture strongly suggests push-up
                        if pred == 'jumping_jacks':
                            if pushup_geometry_hint(joints_px, conf):
                                pred = 'push_ups'

                        # Geometry filter: reject labels that contradict skeleton pose.
                        # Prevents bench_press/pull_ups confusion — both upper-body
                        # dominant but bench_press requires a flat (horizontal) torso.
                        if geometry_plausible(pred, joints_px, conf):
                            pred_history.append(pred)
                        acc_tracker.update(pred, c)
                        if pred_history:
                            stable = collections.Counter(pred_history).most_common(1)[0][0]
                            if stable != current_exercise:
                                print(f'[{frame_idx}] Exercise: {current_exercise} → {stable}')
                                rep_counter.reset(stable)
                                current_exercise = stable
                        current_conf = c

                angles = get_angles(joints_px)

                if current_exercise:
                    deviations, form_score = get_angle_deviations(current_exercise, angles)

                reps, rep_state = 0, STATE_INIT
                if current_exercise:
                    reps      = rep_counter.update(current_exercise, angles, current_conf)
                    rep_state = rep_counter.states.get(current_exercise, STATE_INIT)

                if frame_idx % 10 == 0 and current_exercise:
                    feedback = get_form_feedback(current_exercise, angles, rep_state)
                    # ── Push to GUI ───────────────────────────────────────────────────
                if gui_state is not None:
                    gui_state.update(
                        exercise   = current_exercise or "—",
                        confidence = float(current_conf),
                        reps       = int(reps),
                        rep_state  = str(rep_state),
                        form_score = float(form_score),
                        feedback   = list(feedback),
                        angles     = dict(angles),
                        deviations = dict(deviations),
                        frame_count= int(frame_idx),   # or whatever your frame counter var is
                    )

                draw_skeleton(frame, joints_px, conf)
                draw_overlay(frame, current_exercise, current_conf,
                             reps, feedback, rep_state, acc_tracker.accuracy())
                # draw_right_panel(frame, current_exercise, angles,
                #                  deviations, form_score, reps)
                rep_acc_val = None
                if ground_truth_reps is not None and current_exercise:
                    rep_data = acc_tracker.rep_accuracy(reps)
                    if rep_data:
                        rep_acc_val = rep_data[0]

                stability = 0
                if current_exercise:
                    stability = rep_counter.stable_conf_count.get(current_exercise, 0)

                draw_right_panel(
                    frame,
                    current_exercise,
                    angles,
                    deviations,
                    form_score,
                    reps,
                    confidence=current_conf,
                    rep_accuracy=rep_acc_val,
                    stability=stability
                )
                draw_visibility_warning(frame, ankle_vis, current_exercise)
            else:
                cv2.putText(frame, 'No pose detected', (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,0,200), 2)

            if writer: writer.write(frame)
            cv2.imshow('Kinexis', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                print('Stopped.')
                break

    finally:
        cap.release()
        if writer: writer.release()
        cv2.destroyAllWindows()
        pose.close()
        print(f'Frames processed: {frame_idx}')
        final_reps = 0
        if current_exercise:
            final_reps = rep_counter.reps[current_exercise]
            print(f'Exercise : {current_exercise}')
            print(f'Reps     : {final_reps}')
        print(acc_tracker.summary(final_reps=final_reps))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Kinexis local inference')
    parser.add_argument('--source',            default='0',
                        help='0=webcam or path to video file')
    parser.add_argument('--output',            default=None,
                        help='path to save annotated video')
    parser.add_argument('--max_frames',        default=None, type=int)
    parser.add_argument('--ground_truth',      default=None,
                        choices=list(LABEL_MAP.values()),
                        help='known exercise label for supervised accuracy')
    parser.add_argument('--ground_truth_reps', default=None, type=int,
                        help='actual rep count for rep accuracy calculation')
    args = parser.parse_args()

    source = int(args.source.strip()) if args.source.strip().isdigit() else args.source
    run_inference(source=source, output_path=args.output,
                  max_frames=args.max_frames, ground_truth=args.ground_truth,
                  ground_truth_reps=args.ground_truth_reps)