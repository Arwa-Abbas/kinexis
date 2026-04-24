import cv2
import torch
import numpy as np
import mediapipe as mp
import argparse
import threading
from collections import deque
from config import *
from features import extract_frame_features, count_reps
from rules import RulesEngine
from feedback import draw_skeleton, VoiceFeedback
from extract_keypoints import keypoints_to_stgcn_tensor
from model import BiomechSTGCN


def run_realtime(exercise: str = "squat", source: int = 0):
    # ── Load model ──────────────────────────────────────────
    model = BiomechSTGCN(
        num_exercise_classes=len(EXERCISE_CLASSES),
        num_quality_classes=len(QUALITY_CLASSES),
    ).to(DEVICE)

    ckpt = os.path.join(CHECKPOINTS_DIR, "best_model.pt")
    if os.path.exists(ckpt):
        model.load_state_dict(torch.load(ckpt, map_location=DEVICE))
        print("Loaded trained model.")
    else:
        print("No checkpoint found — running without ST-GCN (rules only).")
    model.eval()

    # ── Setup BlazePose ─────────────────────────────────────
    mp_pose = mp.solutions.pose
    pose = mp_pose.Pose(min_detection_confidence=0.7, min_tracking_confidence=0.5)

    # ── Setup components ────────────────────────────────────
    rules = RulesEngine()
    voice = VoiceFeedback()
    cap = cv2.VideoCapture(source)
    kp_buf = deque(maxlen=SEQUENCE_LEN)  # rolling keypoint buffer

    rep_count = 0
    knee_angles: list = []
    frame_num = 0

    # ST-GCN runs async every 0.5s to avoid blocking
    quality_label = "..."
    stgcn_lock = threading.Lock()

    def run_stgcn_async(buf_snapshot):
        nonlocal quality_label
        if len(buf_snapshot) < 30:
            return
        kp_arr = np.array(list(buf_snapshot))  # (T, 33, 4)
        tensor = keypoints_to_stgcn_tensor(kp_arr)
        X = torch.FloatTensor(tensor[np.newaxis, :, :, :, np.newaxis]).to(DEVICE)
        with torch.no_grad():
            out = model(X)
            q_idx = out["quality"].argmax().item()
        with stgcn_lock:
            quality_label = [k for k, v in QUALITY_CLASSES.items() if v == q_idx][0]

    print(f"Starting real-time demo | Exercise: {exercise} | Press Q to quit")

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)  # mirror for user-facing cam
        frame = cv2.resize(frame, (FRAME_WIDTH, FRAME_HEIGHT))
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # ── BlazePose ──
        results = pose.process(rgb)
        if results.pose_landmarks:
            lms = results.pose_landmarks.landmark
            kp_row = np.array([[lm.x, lm.y, lm.z, lm.visibility] for lm in lms])
            kp_buf.append(kp_row)

            # ── Features ──
            feats = extract_frame_features(lms)
            if feats["left_knee"]:
                knee_angles.append(feats["left_knee"])
                rep_count = count_reps(knee_angles)

            # ── Rules engine ──
            errors = rules.check(feats)

            # ── Voice feedback ──
            for err in rules.new_errors_to_speak():
                voice.speak(err["voice"])
                rules.last_spoken = err["name"]
                break  # speak one error at a time

            # ── ST-GCN every 15 frames ──
            if frame_num % 15 == 0:
                buf_snap = kp_buf.copy()
                threading.Thread(
                    target=run_stgcn_async, args=(buf_snap,), daemon=True
                ).start()

            # ── Draw ──
            frame = draw_skeleton(frame, lms, feats, errors, exercise, rep_count)

        # ── HUD ──
        cv2.putText(
            frame,
            f"ST-GCN: {quality_label}",
            (10, FRAME_HEIGHT - 60),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (200, 200, 200),
            1,
        )
        cv2.imshow("Biomechanical Feedback System — Q to quit", frame)
        frame_num += 1

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()
    print(f"Session ended. Total reps counted: {rep_count}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--exercise", default="squat", choices=list(EXERCISE_CLASSES.keys())
    )
    parser.add_argument(
        "--source", default=0, type=int, help="Camera index (0=default) or video path"
    )
    args = parser.parse_args()
    run_realtime(exercise=args.exercise, source=args.source)
