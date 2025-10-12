import torch
import torchaudio
import soundfile as sf
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor
import os
import pandas as pd
from tqdm import tqdm
import jiwer
import librosa
import argparse
from pathlib import Path

class Wav2Vec2Transcriber:
    def __init__(self, model_name="facebook/wav2vec2-base-960h"):
        """Initialize Wav2Vec2 model and processor"""
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")

        # Load model and processor
        print(f"Loading model: {model_name}")
        self.processor = Wav2Vec2Processor.from_pretrained(model_name)
        self.model = Wav2Vec2ForCTC.from_pretrained(model_name)
        self.model.to(self.device)
        self.model.eval()
        print("Model loaded successfully!")

    def load_audio(self, audio_path):
        """Load and preprocess audio file"""
        try:
            # Load audio with soundfile
            audio, sample_rate = sf.read(audio_path)

            # Convert to mono if stereo
            if len(audio.shape) > 1:
                audio = audio.mean(axis=1)

            # Convert to 16kHz if needed
            if sample_rate != 16000:
                audio = librosa.resample(audio, orig_sr=sample_rate, target_sr=16000)
                sample_rate = 16000

            return audio, sample_rate
        except Exception as e:
            print(f"Error loading {audio_path}: {e}")
            return None, None

    def transcribe_audio(self, audio_path):
        """Transcribe a single audio file"""
        audio, sample_rate = self.load_audio(audio_path)
        if audio is None:
            return ""

        # Handle very long audio by chunking
        max_length = 16000 * 30  # 30 seconds max
        if len(audio) > max_length:
            # Process in chunks and combine
            chunks = [audio[i:i+max_length] for i in range(0, len(audio), max_length)]
            transcriptions = []

            for chunk in chunks:
                if len(chunk) > 1000:  # Skip very short chunks
                    chunk_transcription = self._transcribe_chunk(chunk, sample_rate)
                    if chunk_transcription:
                        transcriptions.append(chunk_transcription)

            return " ".join(transcriptions)
        else:
            return self._transcribe_chunk(audio, sample_rate)

    def _transcribe_chunk(self, audio, sample_rate):
        """Transcribe a single audio chunk"""
        try:
            # Preprocess audio
            inputs = self.processor(audio, sampling_rate=sample_rate, return_tensors="pt", padding=True)

            with torch.no_grad():
                # Move to device
                inputs = {k: v.to(self.device) for k, v in inputs.items()}

                # Get model predictions
                logits = self.model(**inputs).logits

                # Decode predictions
                predicted_ids = torch.argmax(logits, dim=-1)
                transcription = self.processor.batch_decode(predicted_ids)[0]

            return transcription.lower().strip()
        except Exception as e:
            print(f"Error transcribing chunk: {e}")
            return ""

    def transcribe_batch(self, audio_files):
        """Transcribe multiple audio files"""
        transcriptions = {}

        for audio_file in tqdm(audio_files, desc="Transcribing"):
            try:
                transcription = self.transcribe_audio(audio_file)
                filename = os.path.basename(audio_file)
                transcriptions[filename] = transcription
            except Exception as e:
                print(f"Error processing {audio_file}: {e}")
                transcriptions[os.path.basename(audio_file)] = ""

        return transcriptions

def load_ground_truth(transcription_file):
    """Load ground truth transcriptions"""
    ground_truth = {}

    try:
        with open(transcription_file, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue

                if '|' in line:
                    parts = line.split('|', 1)
                    if len(parts) == 2:
                        filename, transcription = parts
                        ground_truth[filename.strip()] = transcription.strip().lower()
                    else:
                        print(f"Warning: Invalid format in line {line_num}: {line}")
                else:
                    print(f"Warning: No separator '|' found in line {line_num}: {line}")
    except FileNotFoundError:
        print(f"Error: Transcription file '{transcription_file}' not found!")
        return {}
    except Exception as e:
        print(f"Error reading transcription file: {e}")
        return {}

    print(f"Loaded {len(ground_truth)} ground truth transcriptions")
    return ground_truth

def normalize_text(text):
    """Basic text normalization"""
    import re

    # Convert to lowercase
    text = text.lower()

    # Remove extra whitespace
    text = ' '.join(text.split())

    # Remove punctuation (optional - comment out if you want to keep punctuation)
    text = re.sub(r'[^\w\s]', '', text)

    return text.strip()

def evaluate_transcriptions(predicted, ground_truth, normalize=True):
    """Evaluate transcription accuracy"""
    results = []

    matched_files = 0
    for filename in predicted.keys():
        if filename in ground_truth:
            matched_files += 1
            pred_text = predicted[filename]
            true_text = ground_truth[filename]

            # Apply normalization if requested
            if normalize:
                pred_text_norm = normalize_text(pred_text)
                true_text_norm = normalize_text(true_text)
            else:
                pred_text_norm = pred_text
                true_text_norm = true_text

            # Skip empty transcriptions
            if not true_text_norm:
                print(f"Warning: Empty ground truth for {filename}")
                continue

            try:
                # Calculate WER and CER
                wer = jiwer.wer(true_text_norm, pred_text_norm)
                cer = jiwer.cer(true_text_norm, pred_text_norm)

                results.append({
                    'filename': filename,
                    'ground_truth': true_text,
                    'prediction': pred_text,
                    'ground_truth_normalized': true_text_norm,
                    'prediction_normalized': pred_text_norm,
                    'wer': wer,
                    'cer': cer,
                    'length_ground_truth': len(true_text_norm.split()),
                    'length_prediction': len(pred_text_norm.split())
                })
            except Exception as e:
                print(f"Error calculating metrics for {filename}: {e}")

    print(f"Matched files: {matched_files}/{len(predicted)}")
    return results

def analyze_results(results_df):
    """Analyze and print evaluation results"""
    if results_df.empty:
        print("No results to analyze!")
        return

    print("\n" + "="*60)
    print("EVALUATION RESULTS SUMMARY")
    print("="*60)

    # Overall statistics
    avg_wer = results_df['wer'].mean()
    avg_cer = results_df['cer'].mean()
    median_wer = results_df['wer'].median()
    median_cer = results_df['cer'].median()

    print(f"Average WER: {avg_wer:.3f} ({avg_wer*100:.1f}%)")
    print(f"Average CER: {avg_cer:.3f} ({avg_cer*100:.1f}%)")
    print(f"Median WER: {median_wer:.3f} ({median_wer*100:.1f}%)")
    print(f"Median CER: {median_cer:.3f} ({median_cer*100:.1f}%)")

    # WER distribution
    print(f"\nWER Distribution:")
    print(f"Perfect (WER = 0): {(results_df['wer'] == 0).sum()}/{len(results_df)} files ({(results_df['wer'] == 0).mean()*100:.1f}%)")
    print(f"Excellent (WER < 0.1): {(results_df['wer'] < 0.1).sum()}/{len(results_df)} files ({(results_df['wer'] < 0.1).mean()*100:.1f}%)")
    print(f"Good (WER < 0.2): {(results_df['wer'] < 0.2).sum()}/{len(results_df)} files ({(results_df['wer'] < 0.2).mean()*100:.1f}%)")
    print(f"Fair (WER < 0.5): {(results_df['wer'] < 0.5).sum()}/{len(results_df)} files ({(results_df['wer'] < 0.5).mean()*100:.1f}%)")
    print(f"Poor (WER >= 0.5): {(results_df['wer'] >= 0.5).sum()}/{len(results_df)} files ({(results_df['wer'] >= 0.5).mean()*100:.1f}%)")

    # Best and worst performers
    print(f"\nBest performing files (lowest WER):")
    best_files = results_df.nsmallest(3, 'wer')[['filename', 'wer', 'cer']]
    for _, row in best_files.iterrows():
        print(f"  {row['filename']}: WER={row['wer']:.3f}, CER={row['cer']:.3f}")

    print(f"\nWorst performing files (highest WER):")
    worst_files = results_df.nlargest(3, 'wer')[['filename', 'wer', 'cer']]
    for _, row in worst_files.iterrows():
        print(f"  {row['filename']}: WER={row['wer']:.3f}, CER={row['cer']:.3f}")

def main():
    parser = argparse.ArgumentParser(description='Evaluate transcription accuracy using Wav2Vec2')
    parser.add_argument('--audio_folder', default='audio_files', help='Folder containing audio files')
    parser.add_argument('--transcription_file', default='transcriptions.txt', help='File containing ground truth transcriptions')
    parser.add_argument('--output_folder', default='results', help='Output folder for results')
    parser.add_argument('--model_name', default='facebook/wav2vec2-base-960h', help='Wav2Vec2 model to use')
    parser.add_argument('--no_normalize', action='store_true', help='Skip text normalization')

    args = parser.parse_args()

    # Create output directory
    Path(args.output_folder).mkdir(exist_ok=True)

    # Check if audio folder exists
    if not os.path.exists(args.audio_folder):
        print(f"Error: Audio folder '{args.audio_folder}' not found!")
        return

    # Check if transcription file exists
    if not os.path.exists(args.transcription_file):
        print(f"Error: Transcription file '{args.transcription_file}' not found!")
        return

    # Initialize transcriber
    print("Initializing Wav2Vec2 transcriber...")
    transcriber = Wav2Vec2Transcriber(args.model_name)

    # Get audio files
    audio_extensions = ['.wav', '.mp3', '.flac', '.m4a']
    audio_files = []

    for ext in audio_extensions:
        audio_files.extend(Path(args.audio_folder).glob(f'*{ext}'))
        audio_files.extend(Path(args.audio_folder).glob(f'*{ext.upper()}'))

    audio_files = [str(f) for f in audio_files]

    if not audio_files:
        print(f"No audio files found in '{args.audio_folder}'!")
        return

    print(f"Found {len(audio_files)} audio files")

    # Load ground truth
    print("Loading ground truth transcriptions...")
    ground_truth = load_ground_truth(args.transcription_file)

    if not ground_truth:
        print("No ground truth transcriptions loaded!")
        return

    # Transcribe audio files
    print("Starting transcription...")
    predictions = transcriber.transcribe_batch(audio_files)

    # Save raw predictions
    pred_file = os.path.join(args.output_folder, 'wav2vec2_predictions.txt')
    with open(pred_file, 'w', encoding='utf-8') as f:
        for filename, transcription in predictions.items():
            f.write(f"{filename}|{transcription}\n")
    print(f"Raw predictions saved to: {pred_file}")

    # Evaluate results
    print("Evaluating results...")
    results = evaluate_transcriptions(predictions, ground_truth, normalize=not args.no_normalize)

    if not results:
        print("No matching files found for evaluation!")
        return

    # Save detailed results
    df = pd.DataFrame(results)
    csv_file = os.path.join(args.output_folder, 'evaluation_results.csv')
    df.to_csv(csv_file, index=False)
    print(f"Detailed results saved to: {csv_file}")

    # Analyze and display results
    analyze_results(df)

    print(f"\nAll results saved in '{args.output_folder}' folder")

if __name__ == "__main__":
    main()
