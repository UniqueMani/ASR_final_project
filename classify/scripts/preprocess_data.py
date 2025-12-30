"""
Data Preprocessing Script
Functions:
1. Convert all audio to 16kHz, Mono, WAV format
2. Segment long audio into 3-5 second chunks
3. Balance data (downsampling)
4. Split train/test dataset with 1:1:1 ratio
"""

import os
import glob
import numpy as np
import soundfile as sf
import librosa
from pathlib import Path
from tqdm import tqdm
import random
from sklearn.model_selection import train_test_split

# Set random seeds
random.seed(42)
np.random.seed(42)

# Configuration parameters
TARGET_SR = 16000  # Target sampling rate
MONO = True  # Mono channel
SEGMENT_LENGTH = (3, 5)  # Segment length range (seconds)
TRAIN_RATIO = 0.8  # Training set ratio

# Dataset path configuration
DATASETS = {
    'mandarin': '../../SHALCAS22A/corpus',
    'tibetan': '../../xbmu_amdo31/data/wav',
    'uyghur': '../../data_thuyg20/data'
}

# Output path
OUTPUT_DIR = '../data/processed'


def load_audio(file_path):
    """Load audio file"""
    try:
        audio, sr = librosa.load(file_path, sr=TARGET_SR, mono=MONO)
        return audio, sr
    except Exception as e:
        print(f"Error loading {file_path}: {e}")
        return None, None


def segment_audio(audio, sr, min_len=3, max_len=5):
    """
    Segment audio into 3-5 second chunks
    """
    segments = []
    audio_len = len(audio) / sr

    if audio_len < min_len:
        # If audio is too short, pad it
        target_len = int(min_len * sr)
        if len(audio) < target_len:
            audio = np.pad(audio, (0, target_len - len(audio)), mode='constant')
        segments.append(audio[:target_len])
    else:
        # Segment the audio
        segment_len = random.uniform(min_len, max_len)
        segment_samples = int(segment_len * sr)

        for start in range(0, len(audio), segment_samples):
            end = start + segment_samples
            if end <= len(audio):
                segments.append(audio[start:end])
            elif len(audio) - start >= int(min_len * sr):
                # Keep the last segment if it's longer than minimum length
                segments.append(audio[start:])

    return segments


def process_dataset(language, audio_files, max_samples=None):
    """
    Process a single dataset
    """
    print(f"\nProcessing {language} dataset...")
    all_segments = []

    # Downsample if needed
    if max_samples and len(audio_files) > max_samples:
        audio_files = random.sample(audio_files, max_samples)

    for audio_file in tqdm(audio_files, desc=f"Processing {language}"):
        audio, sr = load_audio(audio_file)
        if audio is None:
            continue

        # Segment audio
        segments = segment_audio(audio, sr,
                                min_len=SEGMENT_LENGTH[0],
                                max_len=SEGMENT_LENGTH[1])
        all_segments.extend(segments)

    print(f"{language} processing complete, total {len(all_segments)} segments")
    return all_segments


def save_segments(segments, language, split):
    """
    Save audio segments
    """
    output_path = Path(OUTPUT_DIR) / split / language
    output_path.mkdir(parents=True, exist_ok=True)

    for idx, segment in enumerate(tqdm(segments, desc=f"Saving {language} {split}")):
        filename = f"{language}_{split}_{idx:06d}.wav"
        filepath = output_path / filename
        sf.write(filepath, segment, TARGET_SR)

    print(f"{language} {split} saved: {len(segments)} files")


def main():
    print("=" * 60)
    print("Starting Data Preprocessing")
    print("=" * 60)

    # Collect all audio files
    dataset_files = {}
    for lang, path in DATASETS.items():
        # Recursively find all wav files
        files = glob.glob(os.path.join(path, '**/*.wav'), recursive=True)
        dataset_files[lang] = files
        print(f"{lang}: {len(files)} original files")

    # Find the minimum dataset size (Uyghur)
    min_size = min(len(files) for files in dataset_files.values())
    print(f"\nMinimum dataset size: {min_size}")
    print(f"Downsampling other datasets to balance data")

    # Process each dataset
    processed_data = {}
    for lang, files in dataset_files.items():
        # Downsample to minimum dataset size
        segments = process_dataset(lang, files, max_samples=min_size)
        processed_data[lang] = segments

    # Balance datasets: ensure equal number of segments per language
    min_segments = min(len(segments) for segments in processed_data.values())
    print(f"\nBalanced segments per language: {min_segments}")

    for lang in processed_data:
        if len(processed_data[lang]) > min_segments:
            processed_data[lang] = random.sample(processed_data[lang], min_segments)

    # Split train and test sets
    print("\nSplitting train and test sets...")
    for lang, segments in processed_data.items():
        # 80/20 split
        train_segments, test_segments = train_test_split(
            segments, train_size=TRAIN_RATIO, random_state=42
        )

        print(f"\n{lang}:")
        print(f"  Train set: {len(train_segments)} segments")
        print(f"  Test set: {len(test_segments)} segments")

        # Save
        save_segments(train_segments, lang, 'train')
        save_segments(test_segments, lang, 'test')

    print("\n" + "=" * 60)
    print("Data Preprocessing Complete!")
    print("=" * 60)

    # Generate dataset statistics
    print("\nFinal Dataset Statistics:")
    for split in ['train', 'test']:
        print(f"\n{split.upper()}:")
        for lang in ['mandarin', 'tibetan', 'uyghur']:
            path = Path(OUTPUT_DIR) / split / lang
            count = len(list(path.glob('*.wav')))
            print(f"  {lang}: {count} files")


if __name__ == '__main__':
    main()
