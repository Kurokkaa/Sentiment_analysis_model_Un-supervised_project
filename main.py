import sys
import pickle
import numpy as np
import soundfile
import librosa
from pathlib import Path
import shutil
import whisper
from pydub import AudioSegment 
from tqdm import tqdm

WHISPER_MODEL_NAME = "medium.en"                
MODEL_PATH = "result/mlp_classifier.model"      
TEMP_DIR = Path("temp_segments")                

def extract_feature(file_name, **kwargs):
    """Extrait les features (MFCC, Chroma, Mel) du fichier audio."""
    mfcc = kwargs.get("mfcc")
    chroma = kwargs.get("chroma")
    mel = kwargs.get("mel")

    try:
        with soundfile.SoundFile(file_name) as sound_file:
            X = sound_file.read(dtype="float32")
            if len(X.shape) > 1:
                # Convertir en mono si stéréo
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
    except Exception as e:
        print(f"Error extracting features from {file_name}: {e}")
        return np.array([])
        
    return result

print(f"Loading Whisper model: {WHISPER_MODEL_NAME}...")
try:
    whisper_model = whisper.load_model(WHISPER_MODEL_NAME)
except Exception as e:
    print(f"Error loading Whisper model: {e}")
    sys.exit(1)

print(f"Loading emotion classifier model from {MODEL_PATH}...")
try:
    with open(MODEL_PATH, "rb") as f:
        emotion_model = pickle.load(f)
except Exception as e:
    print(f"Error loading emotion classifier: {e}")
    sys.exit(1)

TEMP_DIR.mkdir(exist_ok=True)


def predict_emotion(file_path):
    """Predict emotion from audio file"""
    features = extract_feature(file_path, mfcc=True, chroma=True, mel=True)
    if features.size == 0:
        return "[Erreur feature]"
        
    features = features.reshape(1, -1)
    
    try:
        prediction = emotion_model.predict(features)
        return prediction[0]
    except Exception as e:
        return "[Erreur prédiction]"

def process_audio_file(file_path):
    """Transcrition et prédiction"""
    audio_path = Path(file_path)
    if not audio_path.exists():
        print(f"Erreur: Fichier non trouvé à {file_path}")
        return

    output_txt_path = audio_path.with_suffix(".txt")
    print(f"\n--- Processing '{audio_path.name}' ---")
    
    print("Transcribing audio and getting segments with Whisper...")
    try:
        result = whisper_model.transcribe(str(audio_path), fp16=False)
        segments = result.get("segments", [])
    except Exception as e:
        print(f"Erreur durant la transcription Whisper: {e}")
        return

    try:
        audio = AudioSegment.from_file(audio_path)
    except Exception as e:
        print(f"Erreur lors du chargement audio avec pydub. Assurez-vous que FFmpeg est installé et accessible. Erreur: {e}")
        return

    txt_output = ""
    
    print(f"Processing {len(segments)} segments for emotion prediction...")
    for i, segment in enumerate(tqdm(segments, desc="Segment Analysis")):
        start_ms = int(segment['start'] * 1000)
        end_ms = int(segment['end'] * 1000)
        segment_text = segment['text'].strip()
        
        if not segment_text:
            continue

        segment_audio = audio[start_ms:end_ms]
        temp_file_path = TEMP_DIR / f"{audio_path.stem}_{i:03d}.wav"
        
        try:
            segment_audio.export(temp_file_path, format="wav")
            
            # Predict emotion
            emotion = predict_emotion(temp_file_path)
            
            # Clean up temporary file
            temp_file_path.unlink() 
            
        except Exception as e:
            emotion = "[Erreur découpage]"
            if temp_file_path.exists():
                temp_file_path.unlink()

        # Format 
        line = f"[{emotion}]{segment_text}\n"
        txt_output += line

    # 3. SAVE RESULTS
    output_txt_path.write_text(txt_output, encoding="utf-8")
    
    # Nettoyage
    try:
        shutil.rmtree(TEMP_DIR)
    except OSError as e:
        print(f"Erreur lors de la suppression du répertoire temporaire {TEMP_DIR}: {e}")
        
    print(f"\n✅ Analyse Complète. Résultat sauvegardé dans: '{'./'}'")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python main.py path_to_audio_file.wav")
        sys.exit(1)
        
    file_to_test = sys.argv[1]
    process_audio_file(file_to_test)