import os
import glob
import numpy as np
import soundfile
import librosa
import pickle
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score

# Feature Extraction
def extract_feature(file_name, **kwargs):
    """
    Extract features from an audio file.
    Features supported:
      - MFCC
      - Chroma
      - MEL
      - Contrast
      - Tonnetz
    Example:
        features = extract_feature(path, mfcc=True, chroma=True, mel=True)
    """
    mfcc = kwargs.get("mfcc", False)
    chroma = kwargs.get("chroma", False)
    mel = kwargs.get("mel", False)
    contrast = kwargs.get("contrast", False)
    tonnetz = kwargs.get("tonnetz", False)

    with soundfile.SoundFile(file_name) as sound_file:
        X = sound_file.read(dtype="float32")

        # Some files may be stereo -> convert to mono
        if len(X.shape) > 1:
            X = librosa.to_mono(X)

        sample_rate = sound_file.samplerate

        if chroma or contrast:
            stft = np.abs(librosa.stft(X))

        result = np.array([])

        if mfcc:
            mfccs = np.mean(librosa.feature.mfcc(y=X, sr=sample_rate, n_mfcc=40).T, axis=0)
            result = np.hstack((result, mfccs))

        if chroma:
            chroma_features = np.mean(librosa.feature.chroma_stft(S=stft, sr=sample_rate).T, axis=0)
            result = np.hstack((result, chroma_features))

        if mel:
            mel_features = np.mean(librosa.feature.melspectrogram(y=X, sr=sample_rate).T, axis=0)
            result = np.hstack((result, mel_features))

        if contrast:
            contrast_features = np.mean(librosa.feature.spectral_contrast(S=stft, sr=sample_rate).T, axis=0)
            result = np.hstack((result, contrast_features))

        if tonnetz:
            tonnetz_features = np.mean(
                librosa.feature.tonnetz(y=librosa.effects.harmonic(X), sr=sample_rate).T, axis=0
            )
            result = np.hstack((result, tonnetz_features))

    return result

# Emotions found in the dataset
AVAILABLE_EMOTIONS = {"Angry", "Happy", "Sad", "Neutral", "Surprise"}

def load_data(dataset_path="./emotion_speech_dataset", test_size=0.25):
    """
    Load dataset
    """
    X, y = [], []

    # Iterate through all emotion subfolders inside each speaker directory
    for speaker_folder in sorted(glob.glob(os.path.join(dataset_path, "*"))):
        if not os.path.isdir(speaker_folder):
            continue

        for emotion in AVAILABLE_EMOTIONS:
            emotion_folder = os.path.join(speaker_folder, emotion)
            if not os.path.isdir(emotion_folder):
                continue

            for file_path in glob.glob(os.path.join(emotion_folder, "*.wav")):
                try:
                    features = extract_feature(file_path, mfcc=True, chroma=True, mel=True)
                    X.append(features)
                    y.append(emotion)
                except Exception as e:
                    print(f"[WARN] Skipping {file_path}: {e}")

    print(f"[INFO] Total samples loaded: {len(X)}")

    return train_test_split(np.array(X), y, test_size=test_size, random_state=7)

# Model Training
if __name__ == "__main__":
    X_train, X_test, y_train, y_test = load_data(test_size=0.20)

    print(f"[+] Number of training samples: {len(X_train)}")
    print(f"[+] Number of testing samples: {len(X_test)}")
    print(f"[+] Number of features: {X_train.shape[1]}")

    # Best hyperparameters from grid search
    model_params = {
        "alpha": 0.01,
        "batch_size": 256,
        "epsilon": 1e-08,
        "hidden_layer_sizes": (300,),
        "learning_rate": "adaptive",
        "max_iter": 500,
    }

    model = MLPClassifier(**model_params)

    print("[*] Training the model...")
    model.fit(X_train, y_train)

    # Test and evaluate
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)

    print("Accuracy: {:.2f}%".format(accuracy * 100))

    # Save the trained model
    os.makedirs("result", exist_ok=True)
    pickle.dump(model, open("result/mlp_classifier.model", "wb"))
    print("[+] Model saved to result/mlp_classifier.model")
