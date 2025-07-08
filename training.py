import os
import cv2
import numpy as np
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.model_selection import train_test_split

IMG_SIZE = 224
MAX_SEQ_LENGTH = 20
NUM_FEATURES = 2048

# Feature extractor (InceptionV3)
def build_feature_extractor():
    base_model = keras.applications.InceptionV3(
        weights="imagenet", include_top=False, pooling="avg", input_shape=(IMG_SIZE, IMG_SIZE, 3)
    )
    inputs = keras.Input((IMG_SIZE, IMG_SIZE, 3))
    preprocessed = keras.applications.inception_v3.preprocess_input(inputs)
    outputs = base_model(preprocessed)
    return keras.Model(inputs, outputs, name="feature_extractor")

feature_extractor = build_feature_extractor()

def crop_center_square(frame):
    y, x = frame.shape[0:2]
    min_dim = min(y, x)
    start_x = (x // 2) - (min_dim // 2)
    start_y = (y // 2) - (min_dim // 2)
    return frame[start_y:start_y+min_dim, start_x:start_x+min_dim]

def load_video(path, max_frames=MAX_SEQ_LENGTH, resize=(IMG_SIZE, IMG_SIZE)):
    cap = cv2.VideoCapture(path)
    frames = []
    while len(frames) < max_frames:
        ret, frame = cap.read()
        if not ret:
            break
        frame = crop_center_square(frame)
        frame = cv2.resize(frame, resize)
        frame = frame[:, :, [2, 1, 0]]  # BGR to RGB
        frames.append(frame)
    cap.release()
    return np.array(frames)

def prepare_video(path):
    frames = load_video(path)
    frame_features = np.zeros((MAX_SEQ_LENGTH, NUM_FEATURES), dtype="float32")
    length = min(MAX_SEQ_LENGTH, len(frames))
    for i in range(length):
        frame_features[i] = feature_extractor.predict(frames[i:i+1])[0]
    return frame_features

# Load all data
def load_dataset(base_dir):
    features = []
    labels = []
    for label, category in enumerate(["real", "fake"]):
        folder = os.path.join(base_dir, category)
        for filename in os.listdir(folder):
            video_path = os.path.join(folder, filename)
            print(f"Processing: {video_path}")
            feat = prepare_video(video_path)
            features.append(feat)
            labels.append(label)
    return np.array(features), np.array(labels)

# Load data
X, y = load_dataset("dataset")
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, stratify=y)

# Model architecture
inputs = keras.Input(shape=(MAX_SEQ_LENGTH, NUM_FEATURES))
x = layers.GlobalAveragePooling1D()(inputs)
x = layers.Dense(64, activation='relu')(x)
x = layers.Dropout(0.3)(x)
outputs = layers.Dense(1, activation='sigmoid')(x)
model = keras.Model(inputs, outputs)

model.compile(loss="binary_crossentropy", optimizer="adam", metrics=["accuracy"])

# Train
model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=10, batch_size=4)

# Save model
model.save("model.h5")
print("✅ Model saved as model.h5")
