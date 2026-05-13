import streamlit as st
import numpy as np
import joblib
import librosa
from io import BytesIO


def extract_features(audio, sr):
    mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=13)
    return np.mean(mfcc.T, axis=0)


def preprocess_speech(wav_bytes):
    audio, sr = librosa.load(BytesIO(wav_bytes), sr=None, mono=True, duration=3, offset=0.5)
    if np.max(np.abs(audio)) > 0:
        audio = audio / np.max(np.abs(audio))
    return extract_features(audio, sr)


# Load the pre-trained model pipeline
model = joblib.load("model.joblib")

st.title("Speech Emotion Recognition")
st.header("Predict emotion from a WAV audio file")

uploaded_file = st.file_uploader("Choose a speech file", type="wav")

if uploaded_file is not None:
    features = preprocess_speech(uploaded_file.read())
    features = np.expand_dims(features, axis=0)

    prediction = model.predict(features)
    emotion = prediction[0]

    st.write(f"The predicted emotion is: **{emotion}**")
