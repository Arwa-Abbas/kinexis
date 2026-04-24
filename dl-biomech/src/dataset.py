import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from config import *


class ExerciseDataset(Dataset):
    """
    Loads ST-GCN tensors from STGCN_DIR matched to labels.csv.

    labels.csv format:
        video_name, exercise, quality_label
        squat_correct_001, squat, correct
        squat_knee_valgus_002, squat, knee_valgus
    """

    def __init__(self, split: str = "train", augment: bool = False):
        self.augment = augment
        df = pd.read_csv(LABELS_CSV)

        # Split by subject or random (here: random, 70/15/15)
        train_df, temp_df = train_test_split(
            df, test_size=0.30, random_state=42, stratify=df["quality_label"]
        )
        val_df, test_df = train_test_split(temp_df, test_size=0.50, random_state=42)

        splits = {"train": train_df, "val": val_df, "test": test_df}
        self.df = splits[split].reset_index(drop=True)

        self.exercise_map = EXERCISE_CLASSES
        self.quality_map = QUALITY_CLASSES

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        tensor_path = os.path.join(STGCN_DIR, row["video_name"] + "_tensor.npy")

        # Load: (3, 150, 18)
        X = np.load(tensor_path).astype(np.float32)

        if self.augment:
            X = self._augment(X)

        # Add person dimension: (3, 150, 18) → (3, 150, 18, 1)
        X = X[:, :, :, np.newaxis]

        ex_label = self.exercise_map[row["exercise"]]
        qua_label = self.quality_map[row["quality_label"]]

        return {
            "input": torch.FloatTensor(X),
            "exercise_label": torch.LongTensor([ex_label]),
            "quality_label": torch.LongTensor([qua_label]),
        }

    def _augment(self, X: np.ndarray) -> np.ndarray:
        """Left-right flip + Gaussian noise for data augmentation."""
        # Mirror skeleton: swap left/right joints
        FLIP_PAIRS = [
            (1, 2),
            (3, 4),
            (5, 6),
            (7, 8),
            (9, 10),
            (11, 12),
            (13, 14),
            (15, 16),
        ]
        if np.random.random() > 0.5:
            X = X.copy()
            for l, r in FLIP_PAIRS:
                X[:, :, l], X[:, :, r] = X[:, :, r].copy(), X[:, :, l].copy()
            X[0] = -X[0]  # flip x-axis
        # Add small Gaussian noise
        X += np.random.normal(0, 0.01, X.shape)
        return X
