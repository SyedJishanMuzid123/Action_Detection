import os
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical

# Import config
from config import DATA_PATH, actions, no_sequences, sequence_length

label_map = {label: num for num, label in enumerate(actions)}

sequences, labels = [], []

print("Loading data...")

for action in actions:
    for sequence in range(no_sequences):  # Note: This assumes folder names are 0 to 29
        window = []
        for frame_num in range(sequence_length):
            # We need to handle cases where files might be missing or folder names differ
            # But based on your code, we look for standard paths:
            file_path = os.path.join(DATA_PATH, action, str(sequence), "{}.npy".format(frame_num))

            if os.path.exists(file_path):
                res = np.load(file_path)
                window.append(res)
            else:
                print(f"Warning: File missing at {file_path}")

        # Only append if we have a full window of frames
        if len(window) == sequence_length:
            sequences.append(window)
            labels.append(label_map[action])

print(f"Data Loaded. Sequences: {len(sequences)}")

X = np.array(sequences)
y = to_categorical(labels).astype(int)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.05)

print("---------------------------------------")
print(f"X_train shape: {X_train.shape}")
print(f"y_train shape: {y_train.shape}")
print(f"X_test shape: {X_test.shape}")
print(f"y_test shape: {y_test.shape}")
print("---------------------------------------")
print("Data preprocessing complete. You are ready to build the LSTM model.")