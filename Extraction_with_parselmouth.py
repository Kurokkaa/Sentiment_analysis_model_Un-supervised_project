import parselmouth
import pandas as pd
import os

# --CALCUL OF SPEECH RATE-- 

def get_speech_rate_from_textgrid(sound, silence_threshold=-25, min_silence_duration=0.1):
    """
    Compute speech rate (number of sounding segments per second) based on an automatic Praat segmentation (silence vs. speech).

    """
    tg = parselmouth.praat.call(
        sound,
        "To TextGrid (silences)",
        50,                # intensity threshold (dB)
        0.1,               # minimum pitch (s)
        silence_threshold, # silence threshold (dB)
        min_silence_duration,  # minimum silence (s)
        0.1,               # minimum sounding (s)
        "silent", "sounding"
    )

    n_intervals = parselmouth.praat.call(tg, "Get number of intervals", 1)
    nb_voiced_segments = 0
    total_voiced_time = 0.0

    for i in range(1, n_intervals + 1):
        label = parselmouth.praat.call(tg, "Get label of interval", 1, i)
        if label == "sounding":
            nb_voiced_segments += 1
            start_time = parselmouth.praat.call(tg, "Get start time of interval", 1, i)
            end_time = parselmouth.praat.call(tg, "Get end time of interval", 1, i)
            total_voiced_time += (end_time - start_time)

    total_duration = sound.get_total_duration()
    speech_rate = nb_voiced_segments / total_duration if total_duration > 0 else 0
    voiced_ratio = total_voiced_time / total_duration if total_duration > 0 else 0

    return {
        "speech_rate_segments_per_sec": speech_rate,
        "voiced_time_ratio": voiced_ratio,
        "nb_voiced_segments": nb_voiced_segments,
        "total_duration_sec": total_duration
    }

# --EXTRACTION OF FEATURES--

def extract_praat_features(path):

    snd = parselmouth.Sound(path)

    features = {}

    try:
        # --- Pitch ---
        pitch = snd.to_pitch(pitch_floor=75, pitch_ceiling=300)
        features["mean_f0_Hz"] = parselmouth.praat.call(pitch, "Get mean", 0, 0, "Hertz")
    except Exception as e:
        print(f"Pitch not detected for {os.path.basename(path)} : {e}")
        features["mean_f0_Hz"] = None
        

    try:
        # --- Intensity ---
        intensity = snd.to_intensity()
        features["mean_intensity_dB"] = parselmouth.praat.call(intensity, "Get mean", 0, 0, "energy")
    except Exception as e:
        print(f"Intensity not computed for {os.path.basename(path)} : {e}")
        features["mean_intensity_dB"] = None

    try:
        # --- Jitter & Shimmer ---
        point_process = parselmouth.praat.call(snd, "To PointProcess (periodic, cc)", 75, 300)
        features["jitter_local"] = parselmouth.praat.call(point_process, "Get jitter (local)", 0, 0, 0.0001, 0.02, 1.3)
        features["shimmer_local"] = parselmouth.praat.call([snd, point_process], "Get shimmer (local)", 0, 0, 0.0001, 0.02, 1.3, 1.6)
    except Exception as e:
        print(f"Jitter/Shimmer not computed for {os.path.basename(path)} : {e}")
        features["jitter_local"] = None
        features["shimmer_local"] = None

    try:
        # --- Harmonicité (HNR) ---
        harmonicity = snd.to_harmonicity_cc(time_step=0.01, minimum_pitch=75)
        features["hnr_dB"] = parselmouth.praat.call(harmonicity, "Get mean", 0, 0)
    except Exception as e:
        print(f"HNR not computed for {os.path.basename(path)} : {e}")
        features["hnr_dB"] = None
    
    try:
        # --- Speech Rate via Praat ---
        speech_stats = get_speech_rate_from_textgrid(snd)
        features.update(speech_stats)
    except Exception as e:
        print(f"Speech rate not computed for {os.path.basename(path)} : {e}")
        features["speech_rate_segments_per_sec"] = None
        features["voiced_time_ratio"] = None
        features["nb_voiced_segments"] = None
        features["total_duration_sec"] = None

    return features


# === CONFIGURATION ===
folder = r"C:\Users\aubin\OneDrive\Bureau\M1_NLP\Interdisciplinary_project\RAVDESS\SURPRISE_RAVDESS"
output_csv = r"C:\Users\aubin\OneDrive\Bureau\M1_NLP\Interdisciplinary_project\resultats_surprise_ravdess.csv"

# === EXTRACTION ===
data = []

for filename in os.listdir(folder):
    if filename.lower().endswith(".wav"):
        file_path = os.path.join(folder, filename)
        try:
            feats = extract_praat_features(file_path)
            feats["fichier"] = filename
            data.append(feats)
            print(f"{filename} traité")
        except Exception as e:
            print(f"Error on {filename}: {e}")

# === CSV SAVING ===
df = pd.DataFrame(data)

# Main columns (reorder only those that exist)
cols = ["fichier", "mean_f0_Hz", "mean_intensity_dB", "jitter_local", "shimmer_local", "hnr_dB", "speech_rate_segments_per_sec", "voiced_time_ratio",
    "nb_voiced_segments", "total_duration_sec"]
existing_cols = [c for c in cols if c in df.columns]
df = df[existing_cols]

# Save file
df.to_csv(output_csv, index=False, encoding="utf-8")

print(f"\nExtraction successfully completed!")
print(f"Results saved to: {output_csv}")
print(f"Number of processed files: {len(df)}")

