import numpy as np
import os
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.callbacks import TensorBoard, Callback # Import Callback
from sklearn.metrics import multilabel_confusion_matrix, accuracy_score

# Import config
from config import DATA_PATH, actions, no_sequences, sequence_length

# --- 1. Custom Callback for 90% Accuracy ---
class StopAtAccuracy(Callback):
    def on_epoch_end(self, epoch, logs={}):
        # Check if accuracy is greater than or equal to 90% (0.9)
        if(logs.get('categorical_accuracy') >= 0.9):
            print("\nReached 90% accuracy! Saving model and stopping training.")
            self.model.save('action.h5')
            self.model.stop_training = True

# --- 2. Load and Preprocess Data ---
label_map = {label:num for num, label in enumerate(actions)}

sequences, labels = [], []

print("Loading data...")
for action in actions:
    for sequence in range(no_sequences):
        window = []
        for frame_num in range(sequence_length):
            res = np.load(os.path.join(DATA_PATH, action, str(sequence), "{}.npy".format(frame_num)))
            window.append(res)
        sequences.append(window)
        labels.append(label_map[action])

X = np.array(sequences)
y = to_categorical(labels).astype(int)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.05)

# --- 3. Build LSTM Model ---
log_dir = os.path.join('Logs')
tb_callback = TensorBoard(log_dir=log_dir)

# Initialize our custom callback
stop_callback = StopAtAccuracy()

model = Sequential()
model.add(LSTM(64, return_sequences=True, activation='relu', input_shape=(30, 1662)))
model.add(LSTM(128, return_sequences=True, activation='relu'))
model.add(LSTM(64, return_sequences=False, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(actions.shape[0], activation='softmax'))

model.compile(optimizer='Adam', loss='categorical_crossentropy', metrics=['categorical_accuracy'])

# --- 4. Train Model ---
print("Starting training... Training will stop automatically at 90% accuracy.")

# Add stop_callback to the list
model.fit(X_train, y_train, epochs=2000, callbacks=[tb_callback, stop_callback])

# --- 5. Evaluation ---
# (Only runs if training finishes or stops early)
try:
    # If the callback didn't trigger (rare), we save here manually
    model.save('action.h5')
except:
    pass

model.summary()