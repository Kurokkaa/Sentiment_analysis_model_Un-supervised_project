import os
import glob
import numpy as np
import soundfile
import librosa
import pickle
import torch
import torch.nn.functional as F
from transformers import Wav2Vec2FeatureExtractor, Wav2Vec2ForSequenceClassification
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score

# --- CONFIGURATION WAV2VEC ---
MODEL_ID = "superb/wav2vec2-base-superb-er" 

print("[*] Loading Wav2Vec2 model and feature extractor...")
feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(MODEL_ID)
emotion_model = Wav2Vec2ForSequenceClassification.from_pretrained(MODEL_ID)

ID_TO_EMOTION = {0: "Neutral", 1: "Happy", 2: "Angry", 3: "Sad"}

def predict_emotion_wav2vec(file_path, threshold=0.6):
    """
    Prédit l'émotion et retourne le label si le score dépasse le seuil (threshold).
    """
    speech, sr = librosa.load(file_path, sr=16000)
    inputs = feature_extractor(speech, sampling_rate=16000, return_tensors="pt", padding=True)
    
    with torch.no_grad():
        logits = emotion_model(**inputs).logits
        # On calcule les probabilités avec Softmax
        probs = F.softmax(logits, dim=-1)
        
    confidence, prediction = torch.max(probs, dim=-1)
    
    # Si le modèle n'est pas assez sûr de lui, on rejette l'échantillon
    if confidence.item() < threshold:
        return None
        
    return ID_TO_EMOTION.get(prediction.item())

# --- EXTRACTION DE FEATURES (MLP) ---
def extract_feature(file_name, **kwargs):
    mfcc = kwargs.get("mfcc", False)
    chroma = kwargs.get("chroma", False)
    mel = kwargs.get("mel", False)

    try:
        with soundfile.SoundFile(file_name) as sound_file:
            X = sound_file.read(dtype="float32")
            if len(X.shape) > 1:
                X = librosa.to_mono(X)
            sample_rate = sound_file.samplerate

            result = np.array([])
            if mfcc:
                mfccs = np.mean(librosa.feature.mfcc(y=X, sr=sample_rate, n_mfcc=40).T, axis=0)
                result = np.hstack((result, mfccs))
            if chroma:
                stft = np.abs(librosa.stft(X))
                chroma_features = np.mean(librosa.feature.chroma_stft(S=stft, sr=sample_rate).T, axis=0)
                result = np.hstack((result, chroma_features))
            if mel:
                mel_features = np.mean(librosa.feature.melspectrogram(y=X, sr=sample_rate).T, axis=0)
                result = np.hstack((result, mel_features))
            return result
    except Exception as e:
        return None

def load_data(dataset_old="./emotion_speech_dataset", dataset_new="./dataset_1", test_size=0.20):
    X, y = [], []


    print(f"[*] Processing original dataset: {dataset_old}")
    available_emotions = {"Angry", "Happy", "Sad", "Neutral", "Surprise"}
    
    for speaker_folder in glob.glob(os.path.join(dataset_old, "*")):
        if not os.path.isdir(speaker_folder): continue
        for emotion in available_emotions:
            emotion_folder = os.path.join(speaker_folder, emotion)
            for file_path in glob.glob(os.path.join(emotion_folder, "*.wav")):
                features = extract_feature(file_path, mfcc=True, chroma=True, mel=True)
                if features is not None:
                    X.append(features)
                    y.append(emotion)

    print(f"[*] Labeling new dataset: {dataset_new}...")
    new_files = glob.glob(os.path.join(dataset_new, "**/*.wav"), recursive=True)
    added_count = 0
    
    for file_path in new_files:
        label = predict_emotion_wav2vec(file_path)
        if label:
            features = extract_feature(file_path, mfcc=True, chroma=True, mel=True)
            if features is not None:
                X.append(features)
                y.append(label)
                added_count += 1
    
    print(f"[+] Added {added_count} samples from dataset_1 (rejected low confidence samples)")
    return train_test_split(np.array(X), y, test_size=test_size, random_state=7)

if __name__ == "__main__":
    X_train, X_test, y_train, y_test = load_data()

    print(f"[+] Training samples: {len(X_train)}")
    print(f"[+] Testing samples: {len(X_test)}")
    print(f"[+] Features dimension: {X_train.shape[1]}")

    model_params = {
        "alpha": 0.01,
        "batch_size": 256,
        "hidden_layer_sizes": (300,),
        "learning_rate": "adaptive",
        "max_iter": 500,
    }

    model = MLPClassifier(**model_params)
    print("[*] Training the MLP model...")
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print("Final Accuracy: {:.2f}%".format(accuracy * 100))

    # Save
    os.makedirs("result", exist_ok=True)
    pickle.dump(model, open("result/mlp_classifier_hybrid.model", "wb"))
    print("[+] Model saved to result/mlp_classifier_hybrid.model")