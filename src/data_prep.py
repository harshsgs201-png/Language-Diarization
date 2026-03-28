"""
Data preparation module for Language Diarization.
Handles audio resampling and Mel-spectrogram extraction.
"""

import os
import numpy as np
import librosa
import soundfile as sf
from pathlib import Path
from tqdm import tqdm


def resample_audio(audio_path, target_sr=16000):
    """
    Resample audio file to target sample rate.

    Args:
        audio_path: Path to audio file
        target_sr: Target sample rate (default: 16000 Hz)

    Returns:
        Resampled audio array
    """
    y, sr = librosa.load(audio_path, sr=None)
    if sr != target_sr:
        y = librosa.resample(y, orig_sr=sr, target_sr=target_sr)
    return y, target_sr


def extract_mel_spectrogram(audio_path, n_mels=64, n_fft=2048, hop_length=512, target_sr=16000):
    """
    Extract Mel-spectrogram from audio file.

    Args:
        audio_path: Path to audio file
        n_mels: Number of Mel bands
        n_fft: FFT size
        hop_length: Hop length for STFT
        target_sr: Sample rate

    Returns:
        Mel-spectrogram as numpy array
    """
    y, sr = librosa.load(audio_path, sr=target_sr)
    mel_spec = librosa.feature.melspectrogram(
        y=y, sr=sr, n_mels=n_mels, n_fft=n_fft, hop_length=hop_length
    )
    mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
    return mel_spec_db


def process_dataset(input_dir, output_dir, n_mels=64):
    """
    Process all audio files in a directory.

    Args:
        input_dir: Input directory with audio files
        output_dir: Output directory for saved spectrograms
        n_mels: Number of Mel bands
    """
    os.makedirs(output_dir, exist_ok=True)

    audio_files = list(Path(input_dir).glob('**/*.wav'))

    for audio_file in tqdm(audio_files, desc="Processing audio files"):
        try:
            mel_spec = extract_mel_spectrogram(str(audio_file), n_mels=n_mels)

            # Save as .npy file
            output_path = Path(output_dir) / f"{audio_file.stem}_melspec.npy"
            output_path.parent.mkdir(parents=True, exist_ok=True)
            np.save(str(output_path), mel_spec)

        except Exception as e:
            print(f"Error processing {audio_file}: {e}")


if __name__ == "__main__":
    # Example usage
    input_dir = "data/interim"
    output_dir = "data/processed"
    process_dataset(input_dir, output_dir)
