import pickle
import numpy as np
import soundfile
import librosa
import sys

def extract_feature(file_name, **kwargs):
    mfcc = kwargs.get("mfcc")
    chroma = kwargs.get("chroma")
    mel = kwargs.get("mel")

    with soundfile.SoundFile(file_name) as sound_file:
        X = sound_file.read(dtype="float32")
        if len(X.shape) > 1:
            X = librosa.to_mono(X.T)
        sample_rate = sound_file.samplerate
        result = np.array([])

        if mfcc:
            mfccs = np.mean(librosa.feature.mfcc(y=X, sr=sample_rate, n_mfcc=40).T, axis=0)
            result = np.hstack((result, mfccs))
        if chroma:
            stft = np.abs(librosa.stft(X))
            chroma_feat = np.mean(librosa.feature.chroma_stft(S=stft, sr=sample_rate).T, axis=0)
            result = np.hstack((result, chroma_feat))
        if mel:
            mel_feat = np.mean(librosa.feature.melspectrogram(y=X, sr=sample_rate).T, axis=0)
            result = np.hstack((result, mel_feat))
    return result

# LOAD MODEL
model_path = "result/mlp_classifier.model"
with open(model_path, "rb") as f:
    model = pickle.load(f)

print("[+] Model loaded from", model_path)

# PREDICTION
def predict_emotion(file_path):
    features = extract_feature(file_path, mfcc=True, chroma=True, mel=True)
    features = features.reshape(1, -1)  # reshape pour sklearn
    prediction = model.predict(features)
    return prediction[0]

# TEST FILE
if len(sys.argv) < 2:
    print("Usage: python test_voice.py path_to_file.wav")
    sys.exit(1)

file_to_test = sys.argv[1]
emotion = predict_emotion(file_to_test)
print(f"[+] Predicted emotion for '{file_to_test}': {emotion}")
