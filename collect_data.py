import cv2
import os
import numpy as np
import mediapipe as mp

# Import from our other files
from config import DATA_PATH, actions, no_sequences, sequence_length
from utils import mediapipe_detection, draw_styled_landmarks, extract_keypoints, mp_holistic

# --- 1. Create Folders ---
# We use a dictionary to track the starting folder for each action
action_counts = {}

for action in actions:
    action_path = os.path.join(DATA_PATH, action)
    if not os.path.exists(action_path):
        os.makedirs(action_path)

    dir_content = os.listdir(action_path)
    # Get the highest folder number currently existing
    if dir_content:
        valid_indices = [int(i) for i in dir_content if i.isdigit()]
        dirmax = np.max(valid_indices) if valid_indices else -1  # start at -1 so next is 0
    else:
        dirmax = -1  # start at -1 so next is 0

    action_counts[action] = dirmax + 1  # The folder number we will start creating from

    # Create the new folders (e.g., 0 to 29)
    for sequence in range(no_sequences):
        folder_num = action_counts[action] + sequence
        try:
            os.makedirs(os.path.join(DATA_PATH, action, str(folder_num)))
        except:
            pass

# --- 2. Collect Data ---
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
    # STANDBY LOOP
    while True:
        ret, frame = cap.read()
        image, results = mediapipe_detection(frame, holistic)
        draw_styled_landmarks(image, results)

        cv2.putText(image, "Get Ready! Press 's' to start.", (50, 200),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 4, cv2.LINE_AA)
        cv2.imshow('OpenCV Feed', image)

        if cv2.waitKey(10) & 0xFF == ord('s'):
            break
        if cv2.waitKey(10) & 0xFF == ord('q'):
            cap.release()
            cv2.destroyAllWindows()
            exit()

    # RECORDING LOOP
    for action in actions:
        start_folder = action_counts[action]

        for sequence in range(no_sequences):
            # We must use the correct folder number here
            current_folder = start_folder + sequence

            for frame_num in range(sequence_length):
                ret, frame = cap.read()
                image, results = mediapipe_detection(frame, holistic)
                draw_styled_landmarks(image, results)

                # Display status
                if frame_num == 0:
                    cv2.putText(image, 'STARTING COLLECTION', (120, 200),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 4, cv2.LINE_AA)
                    cv2.waitKey(2000)

                cv2.putText(image, 'Collecting: {} Video #{}'.format(action, sequence), (15, 12),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA)

                cv2.imshow('OpenCV Feed', image)

                # Export keypoints
                keypoints = extract_keypoints(results)

                # Use the calculated folder number
                npy_path = os.path.join(DATA_PATH, action, str(current_folder), str(frame_num))
                np.save(npy_path, keypoints)

                if cv2.waitKey(10) & 0xFF == ord('q'):
                    break

    cap.release()
    cv2.destroyAllWindows()