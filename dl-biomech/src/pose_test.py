# this is only use for testing purpose to check if media pipe is working or not

import cv2
import mediapipe as mp

# Initialize MediaPipe Pose
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)
mp_draw = mp.solutions.drawing_utils

# Open webcam
cap = cv2.VideoCapture(0)

print("Press 'q' to quit")
print("Stand in front of camera...")

while True:
    # Read frame
    ret, frame = cap.read()
    if not ret:
        break

    # Flip horizontally (mirror view)
    frame = cv2.flip(frame, 1)

    # Convert BGR to RGB
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Process the frame
    result = pose.process(rgb)

    # Draw skeleton if pose detected
    if result.pose_landmarks:
        # Draw landmarks and connections
        mp_draw.draw_landmarks(frame, result.pose_landmarks, mp_pose.POSE_CONNECTIONS)

        # Count visible keypoints
        visible = 0
        for lm in result.pose_landmarks.landmark:
            if lm.visibility > 0.5:
                visible += 1

        # Show keypoint count
        cv2.putText(
            frame,
            f"Keypoints: {visible}/33",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0),
            2,
        )
    else:
        cv2.putText(
            frame,
            "No pose detected",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 0, 255),
            2,
        )

    # Show the frame
    cv2.imshow("Pose Detection - Press Q to quit", frame)

    # Press 'q' to quit
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# Cleanup
cap.release()
cv2.destroyAllWindows()
pose.close()

print("Test complete!")
