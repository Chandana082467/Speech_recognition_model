import os
import librosa
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

DATASET_PATH = "dataset"

X = []
y = []

print("Loading dataset...")

for root, dirs, files in os.walk(DATASET_PATH):
    for file in files:
        if file.endswith(".wav"):
            path = os.path.join(root, file)
            label = os.path.basename(os.path.dirname(path))

            audio, sr = librosa.load(path, duration=3, offset=0.5)
            mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=13)
            mfcc = np.mean(mfcc.T, axis=0)

            X.append(mfcc)
            y.append(label)

print("Training model...")

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = Pipeline([
    ("scaler", StandardScaler()),
    ("svm", SVC(kernel="rbf"))
])

model.fit(X_train, y_train)

accuracy = model.score(X_test, y_test)
print("Accuracy:", accuracy)

joblib.dump(model, "model.joblib")
print("Model saved to model.joblib")

def predict(file_path):
    audio, sr = librosa.load(file_path, duration=3, offset=0.5)
    mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=13)
    mfcc = np.mean(mfcc.T, axis=0)
    mfcc = np.array(mfcc).reshape(1, -1)
    return model.predict(mfcc)[0]

result = predict(r"C:\Users\chand\Downloads\Speech-Emotion-Recognition-main\Speech-Emotion-Recognition-main\dataset\here.wav")
print("Prediction:", result)
