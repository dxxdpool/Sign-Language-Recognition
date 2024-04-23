import cv2
import mediapipe as mp
import numpy as np
from mediapipe.framework.formats import landmark_pb2
from mediapipe.python import solutions
import keras.models as km
import time

classes = ['a', 'b', 'c', 'd', 'f', 'i', 's', 'u', 'v', 'w', 'y']

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=2, min_detection_confidence=0.6)

# Drawing specs
mp_drawing = mp.solutions.drawing_utils
drawing_styles = mp.solutions.drawing_styles

model = km.load_model('SignModel.keras')

def display_fps(frame, start_time):
    current_time = time.time()
    fps = 1.0 / (current_time - start_time)
    fps_text = f"FPS: {fps:.2f}"
    cv2.putText(frame, fps_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

def draw_landmarks_on_black_bg(landmarks, width, height):
    # Create a black background
    black_bg = np.zeros((height, width, 3), dtype=np.uint8)

    # Draw landmarks on the black background
    hand_landmarks_proto = landmark_pb2.NormalizedLandmarkList()
    hand_landmarks_proto.landmark.extend([
        landmark_pb2.NormalizedLandmark(x=landmark.x, y=landmark.y, z=landmark.z) for landmark in landmarks.landmark
    ])
    solutions.drawing_utils.draw_landmarks(
        black_bg,
        hand_landmarks_proto,
        solutions.hands.HAND_CONNECTIONS,
        solutions.drawing_styles.get_default_hand_landmarks_style(),
        solutions.drawing_styles.get_default_hand_connections_style()
    )

    return black_bg

import os

def main():
    # Open the default camera
    cap = cv2.VideoCapture(0)

    # Check if camera opened successfully
    if not cap.isOpened():
        print("Error opening video stream")
        return

    start_time = time.time()

    while True:
        # Capture frame-by-frame
        ret, frame = cap.read()
        cv2.GaussianBlur(frame, (3, 3), 1, frame, 1)

        # Display FPS on the frame
        display_fps(frame, start_time)

        # Check if the frame is empty
        if not ret:
            print("End of video stream")
            break

        # Define the region of interest (ROI)
        x, y, w, h = 10, 100, 300, 300
        roi_frame = frame[y:y+h, x:x+w]

        roi_frame_rgb = cv2.cvtColor(roi_frame, cv2.COLOR_BGR2RGB)

        # Detect hands in the ROI
        results = hands.process(roi_frame_rgb)

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                # Draw landmarks on black background
                black_bg = draw_landmarks_on_black_bg(hand_landmarks, w, h)
                # Display the black background with landmarks
                # cv2.imshow("Black Background with Landmarks", black_bg)
                black_bg = np.expand_dims(black_bg, axis=0)

                predicted_probabilities = model.predict(black_bg)
                print(type(predicted_probabilities))
                print(predicted_probabilities)
                predicted_index = np.argmax(model.predict(black_bg))

                letter = classes[predicted_index]
                font = cv2.FONT_HERSHEY_SIMPLEX
                org = (50, 50)  # Position of the text
                font_scale = 1
                color = (255, 0, 0)  # Blue color in BGR
                thickness = 2
                cv2.putText(frame, letter, org, font, font_scale, color, thickness, cv2.LINE_AA)

        # Draw rectangle on the main frame
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 0), 2)

        # Display the main camera frame with ROI rectangle
        cv2.imshow("Main Camera", frame)

        # Press 'q' to exit
        if cv2.waitKey(1) == ord('q'):  # Exit if count is 100
            break

    # When everything done, release the video capture object
    cap.release()

    # Close all windows
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()


