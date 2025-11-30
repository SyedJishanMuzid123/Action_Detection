import cv2
import numpy as np
import os
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
import mediapipe as mp

# Local imports
from config import actions
from utils import mediapipe_detection, draw_styled_landmarks, extract_keypoints, mp_holistic

# --- 1. Re-build Model Architecture ---
model = Sequential()
model.add(LSTM(64, return_sequences=True, activation='relu', input_shape=(30, 1662)))
model.add(LSTM(128, return_sequences=True, activation='relu'))
model.add(LSTM(64, return_sequences=False, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(actions.shape[0], activation='softmax'))

# Load the weights
model.load_weights('action.h5')
print("Model loaded successfully.")

# --- 2. Visualization Logic ---
colors = [(245, 117, 16), (117, 245, 16), (16, 117, 245)]


def prob_viz(res, actions, input_frame, colors):
    output_frame = input_frame.copy()
    for num, prob in enumerate(res):
        # dynamic check to prevent index errors
        color = colors[num] if num < len(colors) else (255, 255, 255)

        cv2.rectangle(output_frame, (0, 60 + num * 40), (int(prob * 100), 90 + num * 40), color, -1)
        cv2.putText(output_frame, actions[num], (0, 85 + num * 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2,
                    cv2.LINE_AA)
    return output_frame


# --- 3. Real-time Detection Loop ---
sequence = []
predictions = []
threshold = 0.5

# New variable for single-sign display
current_action = " "

cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
    while cap.isOpened():

        ret, frame = cap.read()
        if not ret:
            break

        # Make detections
        image, results = mediapipe_detection(frame, holistic)

        # Draw landmarks
        draw_styled_landmarks(image, results)

        # Prediction logic
        keypoints = extract_keypoints(results)
        sequence.append(keypoints)
        sequence = sequence[-30:]  # Keep only the last 30 frames

        if len(sequence) == 30:
            res = model.predict(np.expand_dims(sequence, axis=0))[0]
            predictions.append(np.argmax(res))

            # Stability logic: Check if the last 10 frames predicted the same class
            if np.unique(predictions[-10:])[0] == np.argmax(res):
                # Threshold logic: Only update if confidence is high
                if res[np.argmax(res)] > threshold:
                    # UPDATED: Just set the variable, don't append to a list
                    current_action = actions[np.argmax(res)]
                else:
                    # Optional: clear the text if confidence drops?
                    # For now, let's keep the last valid detection or set to " "
                    pass

                    # Visualize probabilities
            image = prob_viz(res, actions, image, colors)

        # Draw the current action on screen
        cv2.rectangle(image, (0, 0), (640, 40), (245, 117, 16), -1)

        # UPDATED: Display current_action instead of ' '.join(sentence)
        cv2.putText(image, current_action, (3, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

        # Show to screen
        cv2.imshow('OpenCV Feed', image)

        # Break gracefully
        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()