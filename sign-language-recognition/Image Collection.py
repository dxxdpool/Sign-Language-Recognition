import cv2
import mediapipe as mp
import numpy as np
from mediapipe.framework.formats import landmark_pb2
from mediapipe.python import solutions

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=2, min_detection_confidence=0.6)

# Drawing specs
mp_drawing = mp.solutions.drawing_utils
drawing_styles = mp.solutions.drawing_styles

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

    count = 0  # Counter to keep track of the number of saved images
    saving_image = False  # Flag to indicate if an image is currently being saved
    save_dir = "ImageData/f"

    # Create directory if it does not exist
    os.makedirs(save_dir, exist_ok=True)

    while True:
        # Capture frame-by-frame
        ret, frame = cap.read()
        cv2.GaussianBlur(frame, (3, 3), 1, frame, 1)

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
                cv2.imshow("Black Background with Landmarks", black_bg)

                # Save the black background image when 'a' key is pressed
                key = cv2.waitKey(1)
                if key == ord('a'):
                    if not saving_image:  # Check if saving process is not already running
                        saving_image = True
                        print("Saving image...")
                    count += 1
                    filename = os.path.join(save_dir, f"{count}.png")
                    cv2.imwrite(filename, black_bg)
                    print(f"Saved {filename}")

                # Reset saving flag when 'a' key is released
                elif key == ord('a') + 1:
                    saving_image = False

                # Check if count is 130
                if count == 130:
                    print("Reached maximum count of 100 images. Exiting...")
                    break

        # Display message on the main screen if an image is being saved
        if saving_image:
            cv2.putText(frame, "Saving image...", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        # Draw rectangle on the main frame
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 0), 2)

        # Display the main camera frame with ROI rectangle
        cv2.imshow("Main Camera", frame)

        # Press 'q' to exit
        if cv2.waitKey(1) == ord('q') or count == 130:  # Exit if count is 130
            break

    # When everything done, release the video capture object
    cap.release()

    # Close all windows
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
