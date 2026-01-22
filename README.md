# (Un)supervised Project: Speech Emotion & Transcription Pipeline

This project was developed for the **(Un)supervised Project** course. It provides an end-to-end pipeline to transcribe audio files and predict the emotional tone of spoken segments using a hybrid machine learning approach.

---

## Datasets

The system is train on ESD and RAVD

## 📌 Project Overview

The system processes long audio files to generate transcribed text labeled with specific emotions: **Neutral**, **Happy**, **Angry**, or **Sad**.

The project consists of two main components:
1.  **Hybrid Training (`sentiments_analysis.py`)**: Uses a pre-trained **Wav2Vec2** model (`superb/wav2vec2-base-superb-er`) to "pseudo-label" new data. This data is combined with the **ESD** (Emotion Speech Dataset) and **RAVDESS** datasets to train a fast **MLP (Multi-Layer Perceptron)** classifier.
2.  **Inference Pipeline (`main.py`)**: Utilizes **OpenAI Whisper** for transcription and the trained MLP model to predict emotions on a per-segment basis.



---

## 🚀 Features

* **Transcription**: Powered by the `whisper-medium.en` model for high-accuracy English speech-to-text.
* **Emotion Detection**: Classifies segments into four distinct emotional states.
* **Hybrid Audio Analysis**: Extracts **MFCC**, **Chroma**, and **Mel Spectrogram** features for robust classification.
* **Segment-Level Slicing**: Automatically divides audio based on Whisper's timestamps for precise analysis.

---

## 🛠️ Tech Stack

* **Speech Processing**: `Librosa`, `Soundfile`, `Pydub`.
* **Deep Learning**: `PyTorch`, `Transformers` (Hugging Face).
* **Transcription**: `OpenAI Whisper`.
* **Machine Learning**: `Scikit-learn` (MLP Classifier).

---

## 📂 Project Structure

```text
├── sentiments_analysis.py  # Script for training and pseudo-labeling
├── main.py                 # Script for transcription and inference
├── dataset_1/              # Unlabeled data for pseudo-labeling
├── emotion_speech_dataset/ # Labeled dataset for supervised training
├── result/                 # Storage for the final trained model
└── temp_segments/          # Temporary cache for audio segment processing
