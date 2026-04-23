import cv2
import mediapipe as mp

mp_hands = mp.tasks.vision.HandLandmarker
mp_base_options = mp.tasks.BaseOptions
mp_vision = mp.tasks.vision

# Create hand landmarker
base_options = mp_base_options(model_asset_path="hand_landmarker.task")
options = mp_vision.HandLandmarkerOptions(
    base_options=base_options,
    num_hands=1
)

detector = mp_hands.create_from_options(options)

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)

    result = detector.detect(mp_image)

    if result.hand_landmarks:
        for hand_landmarks in result.hand_landmarks:
            for landmark in hand_landmarks:
                h, w, _ = frame.shape
                cx, cy = int(landmark.x * w), int(landmark.y * h)
                cv2.circle(frame, (cx, cy), 5, (0, 255, 0), -1)

    cv2.imshow("Hand Detection", frame)

    if cv2.waitKey(1) == 27:
        break

cap.release()
cv2.destroyAllWindows()
