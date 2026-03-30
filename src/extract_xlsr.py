import os
import sys

# CRITICAL: Set HF cache to D: drive BEFORE importing transformers (C: is full!)
os.environ['HF_HOME'] = 'D:/hf_cache'
os.makedirs(os.environ['HF_HOME'], exist_ok=True)

import glob
import torch
import numpy as np
from tqdm import tqdm
from transformers import Wav2Vec2FeatureExtractor, Wav2Vec2Model
from scipy.io import wavfile
import librosa
import shutil
import gc

# --- Configuration ---
RAW_AUDIO_BASE = "data/raw/Corpus"  # Root directory for audio files
PROCESSED_DIR = "data/processed"     # Where your _labels.npy files live
OUTPUT_DIR = "data/processed"        # We will save _emb.npy here too

SR = 16000
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Verify HF cache location

print(f"HF Cache: {os.environ['HF_HOME']}")
print(f"Device: {DEVICE}")
print(f"Loading XLSR-53 Model...")

# Download and load model
try:
    processor = Wav2Vec2FeatureExtractor.from_pretrained("facebook/wav2vec2-large-xlsr-53")
    model = Wav2Vec2Model.from_pretrained("facebook/wav2vec2-large-xlsr-53").to(DEVICE)
    model.eval()
    print("[SUCCESS] Model loaded successfully")
except Exception as e:
    print(f"[ERROR] Error loading model: {e}")
    print("[ERROR] Check that you have ~2GB free disk space")
    exit(1)

def find_audio_file(file_id):
    """
    Search for an audio file by filename across all audio directories.
    file_id: e.g., "AD09002" or "CH03001"
    
    Returns: Full path to audio file, or None if not found
    """
    # Search in all subdirectories of RAW_AUDIO_BASE
    pattern = os.path.join(RAW_AUDIO_BASE, "**", f"{file_id}.wav")
    matches = glob.glob(pattern, recursive=True)
    
    if matches:
        return matches[0]  # Return first match
    return None

def load_audio_scipy(audio_path, target_sr=16000):
    """Load and resample audio using scipy + librosa (no torchaudio/FFmpeg needed)."""
    try:
        sr, audio_data = wavfile.read(audio_path)
        
        # Convert to float and normalize
        if audio_data.dtype != np.float32:
            audio_data = audio_data.astype(np.float32)
            if audio_data.max() > 1.0:
                audio_data = audio_data / 32768.0  # 16-bit normalization
        
        # Convert to mono if stereo
        if len(audio_data.shape) > 1:
            audio_data = np.mean(audio_data, axis=1)
        
        # Resample if needed
        if sr != target_sr:
            audio_data = librosa.resample(audio_data, orig_sr=sr, target_sr=target_sr)
        
        return audio_data, target_sr
    except Exception as e:
        return None, None

@torch.no_grad()
def process_embeddings():
    """Main extraction loop: process all Phase 1 labels and extract XLSR embeddings."""
    label_files = sorted(glob.glob(os.path.join(PROCESSED_DIR, "*_labels.npy")))
    
    # Filter out already processed files
    pending = [f for f in label_files if not os.path.exists(f.replace("_labels", "_emb"))]
    
    print(f"Total label files: {len(label_files)}")
    print(f"Already processed: {len(label_files) - len(pending)}")
    print(f"Pending extraction: {len(pending)}\n")
    
    if len(pending) == 0:
        print("✓ All files already processed!")
        print_summary(len(label_files), 0, 0)
        return
    
    success_count = 0
    mismatch_count = 0
    missing_audio_count = 0
    
    for label_path in tqdm(pending, desc="Extracting XLSR"):
        # Extract filename from label file
        file_id = os.path.splitext(os.path.basename(label_path))[0]
        file_id = file_id.replace("_labels", "")  # Remove _labels suffix
        
        output_path = os.path.join(OUTPUT_DIR, f"{file_id}_emb.npy")
        
        # Skip if already processed
        if os.path.exists(output_path):
            success_count += 1
            continue
        
        # Find the corresponding audio file
        audio_path = find_audio_file(file_id)
        if audio_path is None:
            missing_audio_count += 1
            continue

        # 1. Load and Resample Audio using scipy (no FFmpeg needed)
        audio_data, loaded_sr = load_audio_scipy(audio_path, target_sr=SR)
        if audio_data is None:
            missing_audio_count += 1
            continue
            
        # 2. Extract XLSR Features
        try:
            inputs = processor(audio_data, sampling_rate=SR, return_tensors="pt")
            inputs = {k: v.to(DEVICE) for k, v in inputs.items()}
            
            outputs = model(**inputs)
            # Shape: (1, Time, 1024) -> (Time, 1024)
            embeddings = outputs.last_hidden_state.squeeze(0).cpu().numpy()
            
            # Clean up GPU memory
            del inputs, outputs
            if DEVICE.type == 'cuda':
                torch.cuda.empty_cache()
            gc.collect()
            
        except Exception as e:
            print(f"Failed to extract features from {file_id}: {e}")
            missing_audio_count += 1
            continue
        
        # 3. Align with Phase 1 Labels
        labels = np.load(label_path)
        
        # Wav2Vec2 CNN downsampling can occasionally result in +/- 1 frame difference
        # compared to strict mathematical division. We truncate to the minimum length.
        min_len = min(embeddings.shape[0], labels.shape[0])
        
        if abs(embeddings.shape[0] - labels.shape[0]) > 2:
            # If the mismatch is large, something is wrong with the audio file length
            mismatch_count += 1
            continue
            
        aligned_embeddings = embeddings[:min_len, :]
        aligned_labels = labels[:min_len]
        
        # Overwrite the label with the perfectly aligned version
        try:
            np.save(label_path, aligned_labels)
            np.save(output_path, aligned_embeddings)
            success_count += 1
        except OSError as e:
            print(f"\nDisk Space Error: {e}")
            print("Please free up disk space and rerun this script")
            raise

def print_summary(total_labels, success_count, mismatch_count, missing_audio_count=0):
    """Print extraction summary."""
    print("\n" + "="*60)
    print("PHASE 2 EXTRACTION SUMMARY")
    print("="*60)
    print(f"Successfully Extracted: {success_count}")
    print(f"Already Existed: {total_labels - success_count - mismatch_count - missing_audio_count}")
    print(f"Large Mismatches Skipped: {mismatch_count}")
    print(f"Missing Audio Files: {missing_audio_count}")
    print(f"Total Processed: {success_count + mismatch_count + missing_audio_count}")
    print("="*60)
    
    # Final check
    embeddings = glob.glob(os.path.join(OUTPUT_DIR, "*_emb.npy"))
    print(f"\nFinal: {len(embeddings)} embeddings saved")

if __name__ == "__main__":
    try:
        process_embeddings()
    except Exception as e:
        print(f"\nExtraction failed: {e}")
        print("\nTroubleshooting:")
        print("1. Free up at least 2GB disk space")
        print("2. Delete ~/.cache/huggingface_xlsr/ to clear model cache")
        print("3. Rerun this script")
