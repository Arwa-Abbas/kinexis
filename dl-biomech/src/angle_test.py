# this is only use for testing purpose

import cv2
import mediapipe as mp
import numpy as np

# Initialize MediaPipe
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)
mp_draw = mp.solutions.drawing_utils


def calculate_angle(a, b, c):
    """Calculate angle between three points"""
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


cap = cv2.VideoCapture(0)

print("Press 'q' to quit")
print("Do a squat! Watch your knee angle.")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = pose.process(rgb)

    if result.pose_landmarks:
        landmarks = result.pose_landmarks.landmark

        # Get knee angle (left knee)
        hip = landmarks[mp_pose.PoseLandmark.LEFT_HIP.value]
        knee = landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value]
        ankle = landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value]

        knee_angle = calculate_angle(hip, knee, ankle)

        # Draw skeleton
        mp_draw.draw_landmarks(frame, result.pose_landmarks, mp_pose.POSE_CONNECTIONS)

        # Show knee angle
        cv2.putText(
            frame,
            f"Knee Angle: {int(knee_angle)} deg",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 0),
            2,
        )

        # Feedback based on angle
        if knee_angle > 140:
            feedback = "Go deeper!"
            color = (0, 0, 255)  # Red
        elif knee_angle < 70:
            feedback = "Too deep!"
            color = (0, 0, 255)
        else:
            feedback = "Good form!"
            color = (0, 255, 0)  # Green

        cv2.putText(frame, feedback, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

        # Draw circle at knee
        knee_pixel = (int(knee.x * frame.shape[1]), int(knee.y * frame.shape[0]))
        cv2.circle(frame, knee_pixel, 8, (0, 255, 255), -1)

    cv2.imshow("Knee Angle Tracker - Press Q to quit", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
pose.close()
print("Test complete!")
