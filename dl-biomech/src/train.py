import torch
import torch.nn as nn
import os
from torch.utils.data import DataLoader
from tqdm import tqdm
from config import *
from dataset import ExerciseDataset
from model import BiomechSTGCN


def train():
    # ── Data ────────────────────────────────────────────────
    train_ds = ExerciseDataset(split="train", augment=True)
    val_ds = ExerciseDataset(split="val")
    train_dl = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)
    val_dl = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)

    print(f"Train: {len(train_ds)} | Val: {len(val_ds)} | Device: {DEVICE}")

    # ── Model ───────────────────────────────────────────────
    model = BiomechSTGCN(
        num_exercise_classes=len(EXERCISE_CLASSES),
        num_quality_classes=len(QUALITY_CLASSES),
        pretrained_path=PRETRAINED_PT,
    ).to(DEVICE)

    # ── Loss + Optimizer ────────────────────────────────────
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
    )
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=15, gamma=0.5)

    # ── Training Loop ───────────────────────────────────────
    best_val_loss = float("inf")
    patience_ctr = 0

    for epoch in range(NUM_EPOCHS):
        # ── Train ──
        model.train()
        train_loss = 0.0

        for batch in tqdm.tqdm(train_dl, desc=f"Epoch {epoch+1}/{NUM_EPOCHS}"):
            X = batch["input"].to(DEVICE)
            ex_y = batch["exercise_label"].squeeze().to(DEVICE)
            qu_y = batch["quality_label"].squeeze().to(DEVICE)

            optimizer.zero_grad()
            out = model(X)

            loss = criterion(out["exercise"], ex_y) + criterion(out["quality"], qu_y)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        # ── Validate ──
        model.eval()
        val_loss, val_correct = 0.0, 0
        with torch.no_grad():
            for batch in val_dl:
                X = batch["input"].to(DEVICE)
                ex_y = batch["exercise_label"].squeeze().to(DEVICE)
                qu_y = batch["quality_label"].squeeze().to(DEVICE)
                out = model(X)
                val_loss += (
                    criterion(out["exercise"], ex_y) + criterion(out["quality"], qu_y)
                ).item()
                preds = out["quality"].argmax(dim=1)
                val_correct += (preds == qu_y).sum().item()

        avg_val = val_loss / len(val_dl)
        val_acc = val_correct / len(val_ds)
        print(f"  Val Loss: {avg_val:.4f}  |  Quality Acc: {val_acc:.3f}")
        scheduler.step()

        # ── Save best / Early stop ──
        if avg_val < best_val_loss:
            best_val_loss = avg_val
            patience_ctr = 0
            os.makedirs(CHECKPOINTS_DIR, exist_ok=True)
            torch.save(
                model.state_dict(), os.path.join(CHECKPOINTS_DIR, "best_model.pt")
            )
            print(f"  ✓ Best model saved (val_loss={avg_val:.4f})")
        else:
            patience_ctr += 1
            if patience_ctr >= PATIENCE:
                print(f"Early stopping at epoch {epoch+1}")
                break


if __name__ == "__main__":
    train()
