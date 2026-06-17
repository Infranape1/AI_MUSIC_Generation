import librosa
import numpy as np
import joblib
import os

BASE_DIR = os.path.dirname(__file__)

model = joblib.load(os.path.join(BASE_DIR, "genre_model.pkl"))
scaler = joblib.load(os.path.join(BASE_DIR, "scaler.pkl"))
le = joblib.load(os.path.join(BASE_DIR, "label_encoder.pkl"))


def extract_features(file_path):
    try:
        audio, sr = librosa.load(file_path, duration=30)

        mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=40)
        mfcc_mean = np.mean(mfcc.T, axis=0)

        stft = np.abs(librosa.stft(audio))

        chroma = librosa.feature.chroma_stft(S=stft, sr=sr)
        chroma_mean = np.mean(chroma.T, axis=0)

        contrast = librosa.feature.spectral_contrast(S=stft, sr=sr)
        contrast_mean = np.mean(contrast.T, axis=0)

        return np.hstack([mfcc_mean, chroma_mean, contrast_mean])

    except Exception as e:
        print("Feature Error:", str(e))
        return None


def predict_genre(file):
    try:
        features = extract_features(file)

        if features is None:
            return "Unknown"

        features = features.reshape(1, -1)
        features = scaler.transform(features)

        prediction = model.predict(features)

        return le.inverse_transform(prediction)[0]

    except Exception as e:
        print("Prediction Error:", str(e))
        return "Unknown"