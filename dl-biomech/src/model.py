import sys
import os
import torch
import torch.nn as nn
from config import *

# Add ST-GCN to Python path
sys.path.insert(0, os.path.join(BASE_DIR, "st-gcn"))
from net.st_gcn import Model as STGCN_Base


class BiomechSTGCN(nn.Module):
    """
    ST-GCN with dual classification heads:
      1. exercise_head → predict exercise type (squat, lunge, etc.)
      2. quality_head  → predict form quality / error type
    Uses transfer learning from pre-trained kinetics checkpoint.
    """

    def __init__(
        self,
        num_exercise_classes: int = 6,
        num_quality_classes: int = 6,
        pretrained_path: str = None,
    ):
        super().__init__()

        # Base ST-GCN (graph structure for 18 joints)
        self.backbone = STGCN_Base(
            in_channels=3,
            num_class=400,  # original Kinetics classes (we replace head)
            graph_args={"layout": "openpose", "strategy": "spatial"},
            edge_importance_weighting=True,
        )

        # Load pre-trained weights (transfer learning)
        if pretrained_path and os.path.exists(pretrained_path):
            state = torch.load(pretrained_path, map_location="cpu")
            # Load everything except the final FC layer
            filtered = {k: v for k, v in state.items() if "fcn" not in k}
            self.backbone.load_state_dict(filtered, strict=False)
            print(f"Loaded pre-trained weights from {pretrained_path}")

        # Freeze backbone (only fine-tune heads)
        for name, param in self.backbone.named_parameters():
            if "fcn" not in name:
                param.requires_grad = False

        # Feature dimension from backbone
        backbone_out_dim = 256

        # Dual classification heads (trainable)
        self.exercise_head = nn.Sequential(
            nn.Linear(backbone_out_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, num_exercise_classes),
        )
        self.quality_head = nn.Sequential(
            nn.Linear(backbone_out_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, num_quality_classes),
        )

    def forward(self, x):
        """x: (N, C, T, V, M)"""
        features = self.backbone.extract_feature(x)  # (N, 256)
        return {
            "exercise": self.exercise_head(features),
            "quality": self.quality_head(features),
        }

    def unfreeze_all(self):
        """Call after initial fine-tuning to train full network."""
        for param in self.parameters():
            param.requires_grad = True
        print("All layers unfrozen for full fine-tuning.")
