from pathlib import Path
import pandas as pd
from jiwer import wer, cer
import re

# ====== Paths ======
REF_ROOT = Path("Emotion Speech Dataset")   # Folder containing the actual transcripts
PRED_ROOT = Path("Transcriptions")          # Folder with Whisper outputs
OUTPUT_FILE = "whisper_evaluation_results.csv"

results = []

for group_dir in sorted(REF_ROOT.glob("*/")):  
    group_name = group_dir.name
    ref_file = group_dir / f"{group_name}.txt"
    if not ref_file.exists():
        print(f"Missing reference file for {group_name}")
        continue

    # Read all reference lines
    ref_lines = ref_file.read_text(encoding="utf-8").strip().splitlines()

    # Traverse all emotion subfolders 
    for emotion_dir in group_dir.iterdir():
        if not emotion_dir.is_dir():
            continue  #skip .txt files
        pred_emotion_dir = PRED_ROOT / group_name / emotion_dir.name
        if not pred_emotion_dir.exists():
            print(f"Missing transcription folder for {group_name}/{emotion_dir.name}")
            continue

        for pred_txt in pred_emotion_dir.glob("*.txt"):
            audio_id = pred_txt.stem 

            # --- Find matching reference line
            matching_lines = [l for l in ref_lines if l.startswith(audio_id)]
            if not matching_lines:
                print(f" No reference line found for {audio_id}")
                continue

            ref_line = matching_lines[0].strip()

            # --- Handle tab-separated reference lines like:
            if "\t" in ref_line:
                parts = ref_line.split("\t")
                if len(parts) >= 2:
                    ref_text = parts[1].strip()  
                else:
                    print(f"Malformed reference line (tab issue): {ref_line}")
                    ref_text = ""
            else:
                # fallback for space-separated files
                parts = ref_line.split(maxsplit=1)
                ref_text = parts[1].rsplit(" ", 1)[0] if len(parts) > 1 else ""

            # --- Predicted text
            pred_text = pred_txt.read_text(encoding="utf-8").strip()

            # --- Compute metrics
            file_wer = wer(ref_text.lower(), pred_text.lower())
            file_cer = cer(ref_text.lower(), pred_text.lower())
            acc = (1 - file_wer) * 100

            # --- Add to results
            results.append({
                "group": group_name,
                "file_name": pred_txt.name,
                "reference_text": ref_text,
                "predicted_text": pred_text,
                "WER": round(file_wer, 3),
                "CER": round(file_cer, 3),
                "Accuracy (%)": round(acc, 2)
            })

# ====== Export results ======
df = pd.DataFrame(results, columns=[
    "group", "file_name", "reference_text", "predicted_text", "WER", "CER", "Accuracy (%)"
])
df.to_csv(OUTPUT_FILE, index=False, encoding="utf-8")

# ====== Print summary ======
print("\n Evaluation Complete!")
print(f"Results saved to '{OUTPUT_FILE}'")
print(f"Total files evaluated: {len(df)}")
print(f"Average Accuracy: {df['Accuracy (%)'].mean():.2f}%")
print("\nSample results:")
print(df.head())

