from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
import mediapipe as mp
import cv2
import numpy as np
import uvicorn
import tempfile
import os

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

mp_pose = mp.solutions.pose
pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)


def calculate_angle(a, b, c):
    a = np.array([a.x, a.y])
    b = np.array([b.x, b.y])
    c = np.array([c.x, c.y])
    radians = np.arctan2(c[1] - b[1], c[0] - b[0]) - np.arctan2(
        a[1] - b[1], a[0] - b[0]
    )
    angle = np.abs(radians * 180.0 / np.pi)
    if angle > 180.0:
        angle = 360 - angle
    return angle


@app.post("/analyze-video")
async def analyze_video(file: UploadFile = File(...)):
    # Save video temporarily
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp_file:
        content = await file.read()
        tmp_file.write(content)
        video_path = tmp_file.name

    try:
        # Process video
        cap = cv2.VideoCapture(video_path)
        left_knees = []
        right_knees = []

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            result = pose.process(rgb)

            if result.pose_landmarks:
                lm = result.pose_landmarks.landmark

                left_knee = calculate_angle(
                    lm[mp_pose.PoseLandmark.LEFT_HIP.value],
                    lm[mp_pose.PoseLandmark.LEFT_KNEE.value],
                    lm[mp_pose.PoseLandmark.LEFT_ANKLE.value],
                )
                right_knee = calculate_angle(
                    lm[mp_pose.PoseLandmark.RIGHT_HIP.value],
                    lm[mp_pose.PoseLandmark.RIGHT_KNEE.value],
                    lm[mp_pose.PoseLandmark.RIGHT_ANKLE.value],
                )

                left_knees.append(left_knee)
                right_knees.append(right_knee)

        cap.release()

        if not left_knees:
            return {"error": "No pose detected in video"}

        avg_left = np.mean(left_knees)
        avg_right = np.mean(right_knees)
        min_knee = min(min(left_knees), min(right_knees))
        max_knee = max(max(left_knees), max(right_knees))

        # Feedback based on min knee angle (deepest squat)
        if min_knee > 140:
            feedback = "❌ Squat not deep enough! Aim for 90 degrees"
        elif min_knee < 70:
            feedback = "⚠️ Too deep! Control your descent"
        else:
            feedback = "✅ Good squat depth!"

        return {
            "left_knee": round(avg_left, 1),
            "right_knee": round(avg_right, 1),
            "min_knee": round(min_knee, 1),
            "max_knee": round(max_knee, 1),
            "feedback": feedback,
            "reps": len([x for x in left_knees if x < 120]) // 2,  # Approximate reps
            "symmetry": round(100 - abs(avg_left - avg_right) / avg_left * 100, 1),
        }

    finally:
        os.unlink(video_path)


@app.get("/health")
async def health():
    return {"status": "ok"}


if __name__ == "__main__":
    print("🚀 AI Service running on http://localhost:8001")
    uvicorn.run(app, host="127.0.0.1", port=8001)
