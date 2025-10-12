from pathlib import Path
import shutil
import whisper
from tqdm import tqdm
import csv
import time

# ======= SETTINGS =======
ROOT_DIR = Path("Emotion Speech Dataset")       # source dataset
OUTPUT_DIR = Path("Transcriptions")             # where new data will be saved
LOG_FILE = OUTPUT_DIR / "transcriptions_log.csv" # master CSV
MODEL_NAME = "medium.en"                        # medium.en model
AUDIO_EXTS = {".wav"}

# ======= LOAD MODEL =======
print(f"Loading Whisper model: {MODEL_NAME} (this may take a few mins first time)...")
model = whisper.load_model(MODEL_NAME)

# ======= FIND ALL AUDIO FILES RECURSIVELY =======
audio_files = [p for p in ROOT_DIR.rglob("*") if p.suffix.lower() in AUDIO_EXTS]
print(f"Found {len(audio_files)} audio files under '{ROOT_DIR}'.")

# ======= PREPARE OUTPUT FOLDERS =======
OUTPUT_DIR.mkdir(exist_ok=True)

# ======= PREPARE CSV LOG FILE =======
if not LOG_FILE.exists():
    with open(LOG_FILE, mode="w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["emotion", "file_name", "file_path", "transcript"])

# ======= PROCESS =======
start = time.time()

for audio_path in tqdm(audio_files, desc="Transcribing"):
    # Identify emotion from folder name 
    emotion = audio_path.parent.name
    relative_folder = audio_path.parent.relative_to(ROOT_DIR)
    out_folder = OUTPUT_DIR / relative_folder
    out_folder.mkdir(parents=True, exist_ok=True)

    # Output paths
    out_txt = out_folder / f"{audio_path.stem}.txt"
    out_audio = out_folder / audio_path.name

    # Skip if transcript already exists
    if out_txt.exists():
        continue

    # ======= TRANSCRIBE =======
    try:
        result = model.transcribe(str(audio_path), fp16=False)
        text = result.get("text", "").strip()
    except Exception as e:
        print(f"Error transcribing {audio_path.name}: {e}")
        text = "[ERROR: transcription failed]"

    # ======= SAVE TRANSCRIPT & COPY AUDIO =======
    out_txt.write_text(text, encoding="utf-8")
    if not out_audio.exists():
        shutil.copy2(audio_path, out_audio)

    # ======= LOG ENTRY TO CSV =======
    with open(LOG_FILE, mode="a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            emotion,
            audio_path.name,
            str(out_audio.relative_to(OUTPUT_DIR)),
            text
        ])

elapsed = time.time() - start
print(f"\n Transcribed {len(audio_files)} files in {elapsed/60:.2f} min.")
print(f"All results saved under '{OUTPUT_DIR}/'")
print(f"Combined log file: '{LOG_FILE}'")
