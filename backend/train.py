# backend/train.py

""" "
STGCN++ Fine-tuning
================================
Architecture reverse-engineered from checkpoint keys/shapes.

MSTCN per branch (checkpoint keys decoded):
  branches.B.0        Conv2d(ch, bc, 1)       ← pointwise projection
  branches.B.1        BatchNorm2d(bc)          ← BN after projection
  branches.B.3.conv   Conv2d(bc, bc, (3,1))   ← depthwise-style temporal conv  [branches 0..3]
  (branch 4 same as 1-3 but different dilation)
  branches.5          Conv2d(ch, bc, 1)        ← final branch: simple 1x1 only (no BN, no conv)
  transform.0         BatchNorm2d(ch)          ← BN before merging
  transform.2         Conv2d(ch, ch, 1)        ← merge projection
  tcn.bn              BatchNorm2d(ch)          ← final BN after TCN

UnitGCN (same-channel blocks have `down` too):
  All 10 blocks have gcn.down.0 (Conv) + gcn.down.1 (BN)

Checkpoint branch channel sizes per output channel (ch):
  ch=64:  branch0=14, branches1-4=10, branch5=10  → sum=14+10*4+10=64
  ch=128: branch0=23, branches1-4=21, branch5=21  → sum=23+21*4+21=128
  ch=256: branch0=46, branches1-4=42, branch5=42  → sum=46+42*4+42=256
"""

import sys, os, pickle, random, urllib.request
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import Counter, OrderedDict
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torch.optim import Adam, SGD
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm

print("Python:", sys.version)

# ============================================================================
# PATH CONFIGURATION - UPDATED FOR LOCAL USE
# ============================================================================
# Get the backend directory (where this script is located)
BACKEND_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT = BACKEND_DIR
WORK_DIR = os.path.join(BACKEND_DIR, "work_dir")
MODELS_DIR = os.path.join(BACKEND_DIR, "models")
DATA_DIR = os.path.join(BACKEND_DIR, "data")
PKL_PATH = os.path.join(DATA_DIR, "penn_action_pyskl.pkl")

# Create directories if they don't exist
os.makedirs(WORK_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(DATA_DIR, exist_ok=True)

print(f"Backend directory: {BACKEND_DIR}")
print(f"Models directory: {MODELS_DIR}")
print(f"Data directory: {DATA_DIR}")
print(f"Pickle exists: {os.path.exists(PKL_PATH)}")
print("CUDA:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))

# Load data
with open(PKL_PATH, "rb") as f:
    data = pickle.load(f)
print(
    f'Total:{len(data["annotations"])}  Train:{len(data["split"]["train"])}  Test:{len(data["split"]["test"])}'
)

LABEL_MAP = {
    0: "pull_ups",
    1: "push_ups",
    2: "bench_press",
    3: "jumping_jacks",
    4: "sit_ups",
    5: "jump_rope",
    6: "squats",
}


# ─────────────────────────────────────────────────────────────────────────────
# DATASET
# ─────────────────────────────────────────────────────────────────────────────
class PennActionDataset(Dataset):
    def __init__(self, pkl_path, split="train", clip_len=150, augment=True):
        with open(pkl_path, "rb") as f:
            raw = pickle.load(f)
        ids = set(raw["split"][split])
        self.samples = [a for a in raw["annotations"] if a["frame_dir"] in ids]
        self.clip_len = clip_len
        self.augment = augment
        print(f"[{split}] {len(self.samples)} videos")

    def _sample_indices(self, T):
        if T >= self.clip_len:
            s = (
                random.randint(0, T - self.clip_len)
                if self.augment
                else (T - self.clip_len) // 2
            )
            return list(range(s, s + self.clip_len))
        return list(range(T)) + [T - 1] * (self.clip_len - T)

    def _time_warp(self, arr):
        T = arr.shape[1]
        if T < 20:
            return arr
        sl = random.randint(T // 5, T // 2)
        ss = random.randint(0, T - sl)
        nl = max(2, int(sl * random.choice([0.75, 1.25])))
        seg = arr[:, ss : ss + sl, :]
        st = torch.from_numpy(seg).unsqueeze(0).permute(0, 1, 3, 2).float()
        sr = F.interpolate(
            st, size=(seg.shape[2], nl), mode="bilinear", align_corners=False
        )
        sr = sr.permute(0, 1, 3, 2).squeeze(0).numpy()
        arr = np.concatenate([arr[:, :ss, :], sr, arr[:, ss + sl :, :]], axis=1)
        return arr[:, np.linspace(0, arr.shape[1] - 1, self.clip_len).astype(int), :]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, i):
        ann = self.samples[i]
        kp = ann["keypoint"][0]
        conf = ann["keypoint_score"][0]
        T = ann["total_frames"]
        idx = self._sample_indices(T)
        kp = kp[idx]
        conf = conf[idx]
        xy = np.transpose(kp, (2, 0, 1)).astype(np.float32)
        conf = conf[np.newaxis, :, :].astype(np.float32)
        if self.augment:
            xy += np.random.randn(*xy.shape).astype(np.float32) * 0.01
            if random.random() < 0.3:
                c = np.concatenate([xy, conf], axis=0)
                c = self._time_warp(c)
                xy, conf = c[:2], c[2:3]
            if random.random() < 0.3:
                nT = max(10, int(self.clip_len * random.uniform(0.8, 1.2)))
                i2 = np.linspace(0, self.clip_len - 1, nT).astype(int)
                xy = xy[:, i2, :]
                conf = conf[:, i2, :]
                i3 = np.linspace(0, xy.shape[1] - 1, self.clip_len).astype(int)
                xy = xy[:, i3, :]
                conf = conf[:, i3, :]
        return torch.from_numpy(xy), torch.from_numpy(conf), int(ann["label"])


# ─────────────────────────────────────────────────────────────────────────────
# GRAPH
# ─────────────────────────────────────────────────────────────────────────────
NUM_JOINTS = 13
EDGES = [
    (0, 1),
    (0, 2),
    (1, 3),
    (3, 5),
    (2, 4),
    (4, 6),
    (1, 7),
    (2, 8),
    (7, 9),
    (9, 11),
    (8, 10),
    (10, 12),
]


def build_adj_3subset(n, edges):
    A = np.zeros((3, n, n), dtype=np.float32)
    A[0] = np.eye(n, dtype=np.float32)
    centre = {1, 2, 7, 8}
    for i, j in edges:
        if j in centre:
            A[1, j, i] = 1.0
            A[2, i, j] = 1.0
        else:
            A[1, i, j] = 1.0
            A[2, j, i] = 1.0
    for k in range(3):
        rs = A[k].sum(1, keepdims=True).clip(min=1e-6)
        A[k] = A[k] / rs
    return A


A3_np = build_adj_3subset(NUM_JOINTS, EDGES)


# ─────────────────────────────────────────────────────────────────────────────
# UnitGCN — ALL blocks have `down` (matches checkpoint)
# ─────────────────────────────────────────────────────────────────────────────
class UnitGCN(nn.Module):
    """
    Checkpoint keys per block:
      gcn.A  (3,V,V)
      gcn.bn  BN(out_ch)
      gcn.conv  Conv2d(in_ch, out_ch*3, 1)
      gcn.down.0  Conv2d(in_ch, out_ch, 1)
      gcn.down.1  BN(out_ch)
    Note: ALL 10 blocks have down (even same-channel ones).
    """

    def __init__(self, in_ch, out_ch, A3):
        super().__init__()
        self.register_buffer("A", torch.from_numpy(A3))
        self.bn = nn.BatchNorm2d(out_ch)
        self.relu = nn.ReLU(inplace=True)
        self.conv = nn.Conv2d(in_ch, out_ch * 3, 1)
        self.down = nn.Sequential(nn.Conv2d(in_ch, out_ch, 1), nn.BatchNorm2d(out_ch))

    def forward(self, x):
        N, C, T, V = x.shape
        y = self.conv(x)
        out_ch = y.shape[1] // 3
        y = y.view(N, 3, out_ch, T, V)
        y = torch.einsum("nkctv,kvw->nctw", y, self.A)
        return self.relu(self.bn(y) + self.down(x))


# ─────────────────────────────────────────────────────────────────────────────
# MSTCN — exact branch structure from checkpoint
# ─────────────────────────────────────────────────────────────────────────────
def _branch_channels(ch):
    """Return (bc0, bc) matching checkpoint channel split."""
    if ch == 64:
        return 14, 10
    if ch == 128:
        return 23, 21
    if ch == 256:
        return 46, 42
    raise ValueError(f"Unexpected channel count {ch}")


class _ConvModule(nn.Module):
    """Wrapper so state_dict key is ...3.conv.weight not ...3.weight"""

    def __init__(self, in_c, out_c, kernel, padding, dilation):
        super().__init__()
        self.conv = nn.Conv2d(in_c, out_c, kernel, padding=padding, dilation=dilation)

    def forward(self, x):
        return self.conv(x)


class MSTCN(nn.Module):
    """Matches checkpoint structure exactly"""

    def __init__(self, ch):
        super().__init__()
        bc0, bc = _branch_channels(ch)

        def make_branch(in_ch, out_ch, kernel=(3, 1), padding=(1, 0), dilation=(1, 1)):
            return nn.Sequential(
                nn.Conv2d(in_ch, out_ch, 1),  # idx 0
                nn.BatchNorm2d(out_ch),  # idx 1
                nn.ReLU(inplace=True),  # idx 2 (no params)
                _ConvModule(
                    out_ch,
                    out_ch,
                    kernel,  # idx 3 → key: 3.conv.*
                    padding=padding,
                    dilation=dilation,
                ),
            )

        self.branches = nn.ModuleList(
            [
                make_branch(ch, bc0, (3, 1), (1, 0), (1, 1)),  # branch 0 dilation=1
                make_branch(ch, bc, (3, 1), (2, 0), (2, 1)),  # branch 1 dilation=2
                make_branch(ch, bc, (3, 1), (3, 0), (3, 1)),  # branch 2 dilation=3
                make_branch(ch, bc, (3, 1), (4, 0), (4, 1)),  # branch 3 dilation=4
                make_branch(ch, bc, (3, 1), (1, 0), (1, 1)),  # branch 4 dilation=1
                nn.Conv2d(ch, bc, 1),  # branch 5 (simple 1x1)
            ]
        )
        self.transform = nn.Sequential(
            nn.BatchNorm2d(ch),  # idx 0
            nn.ReLU(inplace=True),  # idx 1 (no params)
            nn.Conv2d(ch, ch, 1),
        )  # idx 2
        self.bn = nn.BatchNorm2d(ch)

    def forward(self, x):
        outs = [b(x) for b in self.branches]
        y = torch.cat(outs, dim=1)
        y = self.transform(y)
        return self.bn(y)


# ─────────────────────────────────────────────────────────────────────────────
# STGCNBlock
# ─────────────────────────────────────────────────────────────────────────────
class STGCNBlock(nn.Module):
    def __init__(self, in_ch, out_ch, A3):
        super().__init__()
        self.gcn = UnitGCN(in_ch, out_ch, A3)
        self.tcn = MSTCN(out_ch)
        self.act = nn.ReLU(inplace=True)
        if in_ch != out_ch:
            self.residual = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, 1, bias=False), nn.BatchNorm2d(out_ch)
            )
        else:
            self.residual = nn.Identity()

    def forward(self, x):
        return self.act(self.tcn(self.gcn(x)) + self.residual(x))


# ─────────────────────────────────────────────────────────────────────────────
# Backbone
# ─────────────────────────────────────────────────────────────────────────────
class STGCNBackbone(nn.Module):
    def __init__(self, in_channels=2, A3=A3_np):
        super().__init__()
        self.data_bn = nn.BatchNorm1d(in_channels * NUM_JOINTS)
        cfg = [
            (in_channels, 64),
            (64, 64),
            (64, 64),
            (64, 64),
            (64, 128),
            (128, 128),
            (128, 128),
            (128, 256),
            (256, 256),
            (256, 256),
        ]
        self.gcn = nn.ModuleList([STGCNBlock(ic, oc, A3) for ic, oc in cfg])
        self.pool = nn.AdaptiveAvgPool2d(1)

    def forward(self, x):
        N, C, T, V = x.shape
        x = x.permute(0, 1, 3, 2).contiguous().view(N, C * V, T)
        x = self.data_bn(x)
        x = x.view(N, C, V, T).permute(0, 1, 3, 2).contiguous()
        for layer in self.gcn:
            x = layer(x)
        return self.pool(x).view(N, -1)


class ConfidenceEncoder(nn.Module):
    def __init__(self, num_joints=13, out_dim=32):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.mlp = nn.Sequential(
            nn.Linear(num_joints, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(True),
            nn.Linear(32, out_dim),
            nn.Sigmoid(),
        )

    def forward(self, conf):
        c = conf.squeeze(1).permute(0, 2, 1)
        return self.mlp(self.pool(c).squeeze(-1))


class STGCNFineTuned(nn.Module):
    def __init__(self, num_classes=7, A3=A3_np):
        super().__init__()
        self.backbone = STGCNBackbone(in_channels=2, A3=A3)
        self.conf_enc = ConfidenceEncoder(num_joints=NUM_JOINTS, out_dim=32)
        self.head = nn.Sequential(
            nn.Linear(256 + 32, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(True),
            nn.Dropout(0.4),
            nn.Linear(128, num_classes),
        )

    def forward(self, xy, conf):
        return self.head(torch.cat([self.backbone(xy), self.conf_enc(conf)], dim=1))


# ─────────────────────────────────────────────────────────────────────────────
# WEIGHT LOADING
# ─────────────────────────────────────────────────────────────────────────────
PRETRAIN_URL = "https://download.openmmlab.com/mmaction/pyskl/ckpt/stgcnpp/stgcnpp_ntu60_xsub_hrnet/j.pth"
PRETRAIN_PATH = os.path.join(MODELS_DIR, "stgcnpp_ntu60_pretrained.pth")

# Download pretrained weights if not exists
if not os.path.exists(PRETRAIN_PATH):
    print("Downloading pretrained weights (45MB)...")
    urllib.request.urlretrieve(PRETRAIN_URL, PRETRAIN_PATH)
    print("Done.")

# COCO-17 → Penn-13 joint subset
COCO17_TO_PENN13 = [0, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]


def load_pretrained_weights(model, ckpt_path):
    ckpt = torch.load(ckpt_path, map_location="cpu")
    src = ckpt.get("state_dict", ckpt)
    src_bb = {
        k[len("backbone.") :]: v for k, v in src.items() if k.startswith("backbone.")
    }

    dst = model.backbone.state_dict()
    new_state = OrderedDict()
    loaded = adapted = skipped = 0

    for dst_key, dst_val in dst.items():
        if dst_key not in src_bb:
            new_state[dst_key] = dst_val
            skipped += 1
            continue

        src_val = src_bb[dst_key]

        # data_bn: (51,)→(26,)
        if "data_bn" in dst_key and src_val.shape != dst_val.shape:
            s = src_val.view(3, 17)
            new_state[dst_key] = s[:2, COCO17_TO_PENN13].reshape(-1)
            adapted += 1

        # gcn.A: keep our computed adjacency
        elif ".gcn.A" in dst_key:
            new_state[dst_key] = dst_val
            skipped += 1

        # first block input channels 3→2
        elif "gcn.0.gcn.conv.weight" in dst_key:
            new_state[dst_key] = src_val[:, :2, :, :]
            adapted += 1
        elif "gcn.0.gcn.down.0.weight" in dst_key:
            new_state[dst_key] = src_val[:, :2, :, :]
            adapted += 1

        # same shape → direct copy
        elif src_val.shape == dst_val.shape:
            new_state[dst_key] = src_val
            loaded += 1

        # unresolvable
        else:
            new_state[dst_key] = dst_val
            skipped += 1

    model.backbone.load_state_dict(new_state, strict=True)
    total = loaded + adapted + skipped
    print(f"\n✓ Pretrained loading:")
    print(f"  Directly loaded  : {loaded}")
    print(f"  Adapted          : {adapted}")
    print(f"  Kept random init : {skipped}")
    print(f"  Total backbone   : {total}")
    print(f"  → {(loaded+adapted)/total*100:.1f}% from pretrained")
    return model


# ─────────────────────────────────────────────────────────────────────────────
# BUILD MODEL
# ─────────────────────────────────────────────────────────────────────────────
print("\nCreating model...")
model = STGCNFineTuned(num_classes=7)

# Quick key-count sanity check before loading
ckpt = torch.load(PRETRAIN_PATH, map_location="cpu")
src = ckpt.get("state_dict", ckpt)
src_bb = {k[len("backbone.") :]: v for k, v in src.items() if k.startswith("backbone.")}
dst = model.backbone.state_dict()
exact_matches = sum(
    1 for k, v in dst.items() if k in src_bb and src_bb[k].shape == v.shape
)
print(f"Exact shape matches before loading: {exact_matches}/{len(dst)}")

model = load_pretrained_weights(model, PRETRAIN_PATH)

dummy_xy = torch.randn(2, 2, 150, 13)
dummy_conf = torch.randn(2, 1, 150, 13)
out = model(dummy_xy, dummy_conf)
print(f"Output shape: {out.shape} ✓")

# ─────────────────────────────────────────────────────────────────────────────
# PHASE 1: freeze backbone
# ─────────────────────────────────────────────────────────────────────────────
for p in model.backbone.parameters():
    p.requires_grad = False
for p in model.conf_enc.parameters():
    p.requires_grad = True
for p in model.head.parameters():
    p.requires_grad = True
frozen = sum(p.numel() for p in model.backbone.parameters())
trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"\nFrozen   : {frozen/1e6:.3f}M (backbone)")
print(f"Trainable: {trainable/1e6:.4f}M (conf_enc + head)")

# ─────────────────────────────────────────────────────────────────────────────
# TRAINING
# ─────────────────────────────────────────────────────────────────────────────
train_ids = set(data["split"]["train"])
train_labels = [a["label"] for a in data["annotations"] if a["frame_dir"] in train_ids]
label_counts = Counter(train_labels)
total_train = len(train_labels)
class_weights = torch.tensor(
    [total_train / (7 * label_counts.get(i, 1)) for i in range(7)], dtype=torch.float32
)

print("\nClass weights:")
for i, w in enumerate(class_weights):
    print(f"  {LABEL_MAP[i]}: {w:.3f}")

sample_weights = torch.tensor(
    [class_weights[l].item() for l in train_labels], dtype=torch.float32
)

CLIP_LEN = 150
BATCH = 16
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
train_ds = PennActionDataset(PKL_PATH, "train", CLIP_LEN, True)
val_ds = PennActionDataset(PKL_PATH, "test", CLIP_LEN, False)
sampler = WeightedRandomSampler(sample_weights, len(train_ds), replacement=True)


def collate_fn(batch):
    return (
        torch.stack([b[0] for b in batch]),
        torch.stack([b[1] for b in batch]),
        torch.tensor([b[2] for b in batch], dtype=torch.long),
    )


train_dl = DataLoader(
    train_ds,
    BATCH,
    sampler=sampler,
    num_workers=2,
    drop_last=True,
    collate_fn=collate_fn,
)
val_dl = DataLoader(val_ds, BATCH, shuffle=False, num_workers=2, collate_fn=collate_fn)

criterion = nn.CrossEntropyLoss(weight=class_weights.to(DEVICE), label_smoothing=0.1)
model = model.to(DEVICE)

PHASE1_EPOCHS, PHASE2_EPOCHS = 25, 80
optimizer = Adam(
    list(model.conf_enc.parameters()) + list(model.head.parameters()),
    lr=1e-3,
    weight_decay=1e-4,
)
scheduler = CosineAnnealingLR(optimizer, T_max=PHASE1_EPOCHS + PHASE2_EPOCHS)
best_acc = 0.0

print(f'\n{"="*65}')
print(f"Fine-tuning on {DEVICE}  Phase1={PHASE1_EPOCHS}ep  Phase2={PHASE2_EPOCHS}ep")
print(f'{"="*65}\n')

for epoch in range(PHASE1_EPOCHS + PHASE2_EPOCHS):
    if epoch == PHASE1_EPOCHS:
        print("\n" + "=" * 65 + "\nPHASE 2 — Unfreezing backbone\n" + "=" * 65 + "\n")
        for p in model.backbone.parameters():
            p.requires_grad = True
        print(
            f"  Trainable: {sum(p.numel() for p in model.parameters() if p.requires_grad)/1e6:.3f}M"
        )
        optimizer = SGD(
            model.parameters(), lr=5e-5, momentum=0.9, weight_decay=1e-4, nesterov=True
        )
        scheduler = CosineAnnealingLR(optimizer, T_max=PHASE2_EPOCHS, eta_min=1e-7)

    phase = "HEAD" if epoch < PHASE1_EPOCHS else "FULL"
    model.train()
    tl = tc = tt = 0
    for xy, conf, y in tqdm(train_dl, leave=False, desc=f"[{phase}] E{epoch+1}"):
        xy, conf, y = xy.to(DEVICE), conf.to(DEVICE), y.to(DEVICE)
        optimizer.zero_grad()
        logits = model(xy, conf)
        loss = criterion(logits, y)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 40)
        optimizer.step()
        tl += loss.item() * y.size(0)
        tc += (logits.argmax(1) == y).sum().item()
        tt += y.size(0)

    model.eval()
    vl = vc = vt = 0
    with torch.no_grad():
        for xy, conf, y in val_dl:
            xy, conf, y = xy.to(DEVICE), conf.to(DEVICE), y.to(DEVICE)
            logits = model(xy, conf)
            vl += criterion(logits, y).item() * y.size(0)
            vc += (logits.argmax(1) == y).sum().item()
            vt += y.size(0)

    scheduler.step()
    print(
        f"[{phase}] E{epoch+1:3d}/{PHASE1_EPOCHS+PHASE2_EPOCHS} | "
        f"Train {tl/tt:.4f}/{tc/tt:.2%} | Val {vl/vt:.4f}/{vc/vt:.2%}"
    )
    if vc / vt > best_acc:
        best_acc = vc / vt
        # Save to MODELS_DIR instead of WORK_DIR
        torch.save(model.state_dict(), os.path.join(MODELS_DIR, "best_model.pth"))
        print(f"  ✓ New best: {best_acc:.2%}")

print(f'\n{"="*65}\nDone! Best val: {best_acc:.2%}\n{"="*65}')

# ─────────────────────────────────────────────────────────────────────────────
# EVALUATION — Confusion Matrix, Classification Report, Per-Class Accuracy
# ─────────────────────────────────────────────────────────────────────────────
import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report

LABEL_LIST = [
    "pull_ups",
    "push_ups",
    "bench_press",
    "jumping_jacks",
    "sit_ups",
    "jump_rope",
    "squats",
]

# Load best model for evaluation
eval_model = STGCNFineTuned(num_classes=7).to(DEVICE)
eval_model.load_state_dict(
    torch.load(os.path.join(MODELS_DIR, "best_model.pth"), map_location=DEVICE)
)
eval_model.eval()
print("\nBest model loaded for evaluation.")

eval_ds = PennActionDataset(PKL_PATH, "test", CLIP_LEN, augment=False)
eval_dl = DataLoader(
    eval_ds, BATCH, shuffle=False, num_workers=2, collate_fn=collate_fn
)

all_preds, all_labels = [], []
with torch.no_grad():
    for xy, conf, y in eval_dl:
        logits = eval_model(xy.to(DEVICE), conf.to(DEVICE))
        all_preds.extend(logits.argmax(1).cpu().numpy())
        all_labels.extend(y.numpy())

all_preds = np.array(all_preds)
all_labels = np.array(all_labels)

# ── Confusion Matrix ──────────────────────────────────────────────────────────
cm = confusion_matrix(all_labels, all_preds, labels=list(range(7)))
cm_pct = cm / (cm.sum(axis=1, keepdims=True) + 1e-8) * 100

fig, ax = plt.subplots(figsize=(10, 8))
sns.heatmap(
    cm_pct,
    annot=True,
    fmt=".1f",
    cmap="Blues",
    xticklabels=LABEL_LIST,
    yticklabels=LABEL_LIST,
    linewidths=0.5,
    ax=ax,
)
ax.set_title("Confusion Matrix — Best Model (%)")
ax.set_xlabel("Predicted")
ax.set_ylabel("True")
plt.xticks(rotation=30, ha="right")
plt.tight_layout()
# Save to MODELS_DIR
plt.savefig(os.path.join(MODELS_DIR, "confusion_matrix.png"), dpi=150)
plt.show()
print(f"Saved: {os.path.join(MODELS_DIR, 'confusion_matrix.png')}")

# ── Classification Report ─────────────────────────────────────────────────────
report = classification_report(
    all_labels, all_preds, target_names=LABEL_LIST, output_dict=True, zero_division=0
)
report_df = pd.DataFrame(report).transpose()
print("\nClassification Report:\n")
print(report_df.to_string())
report_df.to_csv(os.path.join(MODELS_DIR, "classification_report.csv"))
print(f"Saved: {os.path.join(MODELS_DIR, 'classification_report.csv')}")

# ── Per-Class Accuracy Bar Chart ──────────────────────────────────────────────
class_acc = [(all_preds[all_labels == i] == i).mean() * 100 for i in range(7)]
acc_df = pd.DataFrame({"Class": LABEL_LIST, "Accuracy (%)": class_acc})
print("\nPer-Class Accuracy:\n")
print(acc_df.to_string(index=False))

plt.figure(figsize=(10, 5))
bars = plt.bar(
    acc_df["Class"],
    acc_df["Accuracy (%)"],
    color=["#2ecc71" if v >= 70 else "#e74c3c" for v in class_acc],
)
plt.axhline(70, color="gray", ls="--", label="70% target")
plt.title("Per-Class Accuracy (green ≥ 70%)")
plt.ylabel("Accuracy (%)")
plt.ylim(0, 100)
plt.xticks(rotation=25)
for bar, val in zip(bars, class_acc):
    plt.text(
        bar.get_x() + bar.get_width() / 2,
        val + 1,
        f"{val:.1f}%",
        ha="center",
        fontsize=9,
    )
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(MODELS_DIR, "per_class_accuracy.png"), dpi=150)
plt.show()
print(f"Saved: {os.path.join(MODELS_DIR, 'per_class_accuracy.png')}")

# ── Most confused pairs ───────────────────────────────────────────────────────
print("\nMost confused predictions:")
for i in range(7):
    row = cm[i].copy()
    row[i] = 0
    j = row.argmax()
    print(
        f"  {LABEL_LIST[i]:15s} → confused with {LABEL_LIST[j]:15s} ({row[j]} samples)"
    )

# ─────────────────────────────────────────────────────────────────────────────
# CONFIG / METADATA FILE
# ─────────────────────────────────────────────────────────────────────────────
meta = {
    "best_checkpoint": os.path.join(MODELS_DIR, "best_model.pth"),
    "best_val_accuracy": round(float(best_acc), 4),
    "num_classes": 7,
    "clip_len": CLIP_LEN,
    "num_joints": NUM_JOINTS,
    "in_channels_xy": 2,
    "in_channels_conf": 1,
    "phase1_epochs": PHASE1_EPOCHS,
    "phase2_epochs": PHASE2_EPOCHS,
    "phase1_lr": 1e-3,
    "phase2_lr": 5e-5,
    "batch_size": BATCH,
    "label_smoothing": 0.1,
    "pretrained_checkpoint": PRETRAIN_PATH,
    "pretrained_url": PRETRAIN_URL,
    "label_map": {str(k): v for k, v in LABEL_MAP.items()},
    "per_class_accuracy": {LABEL_LIST[i]: round(class_acc[i], 2) for i in range(7)},
    "mediapipe_to_pennaction": {
        "0": 0,  # nose/head
        "11": 1,  # left_shoulder
        "12": 2,  # right_shoulder
        "13": 3,  # left_elbow
        "14": 4,  # right_elbow
        "15": 5,  # left_wrist
        "16": 6,  # right_wrist
        "23": 7,  # left_hip
        "24": 8,  # right_hip
        "25": 9,  # left_knee
        "26": 10,  # right_knee
        "27": 11,  # left_ankle
        "28": 12,  # right_ankle
    },
    "inference_note": (
        "forward(xy, conf) where xy.shape=(N,2,T,13) and conf.shape=(N,1,T,13)"
    ),
}

meta_path = os.path.join(MODELS_DIR, "model_meta.json")
with open(meta_path, "w") as f:
    json.dump(meta, f, indent=2)
print(f"\nSaved config: {meta_path}")
print(json.dumps(meta, indent=2))

print("\n" + "=" * 65)
print("TRAINING COMPLETE!")
print(f"All outputs saved to: {MODELS_DIR}")
print("=" * 65)
