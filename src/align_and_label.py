"""
Text-Based Alignment & Frame-Level Labeling Pipeline
Fast processing without requiring Whisper audio transcription
"""

import json
import pandas as pd
import numpy as np
from pathlib import Path
from tqdm import tqdm
import warnings

warnings.filterwarnings('ignore')

# Configuration
CORPUS_DIR = Path("data/raw/Corpus")
INTERIM_DIR = Path("data/interim")
PROCESSED_DIR = Path("data/processed")
FRAME_RATE = 50  # 50Hz = 20ms per frame

HINDI_KEYWORDS = {
    'ा', 'ि', 'ु', 'ृ', 'े', 'ै', 'ो', 'ौ', 'ं', 'ः', 'ँ',
    'क', 'ख', 'ग', 'घ', 'ङ', 'च', 'छ', 'ज', 'झ', 'ञ',
    'ट', 'ठ', 'ड', 'ढ', 'ण', 'त', 'थ', 'द', 'ध', 'न',
    'प', 'फ', 'ब', 'भ', 'म', 'य', 'र', 'ल', 'व', 'श', 'ष', 'स', 'ह'
}


def detect_language(word: str) -> str:
    """Detect if word is Hindi or English"""
    if not word:
        return 'OTHER'
    
    for char in word:
        if char in HINDI_KEYWORDS:
            return 'HI'
    
    if word.isalpha() or word.isalnum():
        return 'EN'
    
    return 'OTHER'

def generate_word_timestamps(transcription: str, duration: float) -> pd.DataFrame:
    """Generate approximate timestamps for words by distributing across duration"""
    words = transcription.split()
    if not words or duration <= 0:
        return pd.DataFrame()
    
    word_duration = duration / len(words)
    aligned_data = []
    
    for idx, word in enumerate(words):
        start_time = idx * word_duration
        end_time = (idx + 1) * word_duration
        aligned_data.append({
            "word": word,
            "language": detect_language(word),
            "start_time": start_time,
            "end_time": end_time,
            "confidence": 0.8
        })
    
    return pd.DataFrame(aligned_data)

def create_frame_labels(aligned_df: pd.DataFrame, duration: float, frame_rate: int = 50):
    """Convert word-level labels to frame-level (50Hz = 20ms)"""
    frame_duration = 1.0 / frame_rate
    num_frames = int(np.ceil(duration / frame_duration))
    labels = np.full(num_frames, 2, dtype=np.int32)
    
    language_to_label = {'EN': 0, 'HI': 1, 'OTHER': 2, 'SILENCE': 2}
    
    for _, row in aligned_df.iterrows():
        start_frame = int(row['start_time'] / frame_duration)
        end_frame = int(np.ceil(row['end_time'] / frame_duration))
        
        start_frame = max(0, min(start_frame, num_frames - 1))
        end_frame = max(1, min(end_frame, num_frames))
        
        label_value = language_to_label.get(row['language'], 2)
        labels[start_frame:end_frame] = label_value
    
    return labels

def get_audio_duration(audio_path: Path) -> float:
    """Get audio duration from file"""
    try:
        import librosa
        duration = librosa.get_duration(path=str(audio_path))
        return float(duration)
    except:
        try:
            file_size = audio_path.stat().st_size
            audio_bytes = max(file_size - 44, 0)
            duration = audio_bytes / 32000.0
            return max(float(duration), 0.5)
        except:
            return 1.0

def process_dataset(dataset_name: str):
    """Process one dataset (adult or children)"""
    dataset_path = CORPUS_DIR / dataset_name
    json_path = dataset_path / "annotations" / "code_switched_labels.json"
    
    if not json_path.exists():
        return 0
    
    with open(json_path, 'r', encoding='utf-8') as f:
        annotations = json.load(f)
    
    print(f"\n{dataset_name}: ", end='', flush=True)
    processed = 0
    
    for ann in tqdm(annotations, desc="Processing", unit="file", leave=False):
        try:
            # Handle both 'audio' (adult) and 'audio_filepath' (children) field names
            audio_path_str = ann.get('audio') or ann.get('audio_filepath', '')
            filename = audio_path_str.split('/')[-1]
            if not filename.endswith('.wav'):
                continue
            
            file_id = filename.replace('.wav', '')
            transcription = ann.get('transcription', '').strip()
            
            if not transcription:
                continue
            
            audio_path = None
            audio_dir = dataset_path / "audio"
            for split_name in ['train_split', 'val_split', 'test_split']:
                candidate = audio_dir / split_name / filename
                if candidate.exists():
                    audio_path = candidate
                    break
            
            if not audio_path:
                continue
            
            duration = get_audio_duration(audio_path)
            aligned_df = generate_word_timestamps(transcription, duration)
            if aligned_df.empty:
                continue
            
            INTERIM_DIR.mkdir(parents=True, exist_ok=True)
            csv_path = INTERIM_DIR / f"{file_id}_aligned.csv"
            aligned_df.to_csv(csv_path, index=False)
            
            frame_labels = create_frame_labels(aligned_df, duration, FRAME_RATE)
            
            PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
            npy_path = PROCESSED_DIR / f"{file_id}_labels.npy"
            np.save(npy_path, frame_labels)
            
            metadata = {
                "file_id": file_id,
                "dataset": dataset_name,
                "duration": float(duration),
                "frame_rate": FRAME_RATE,
                "num_frames": int(len(frame_labels)),
                "label_distribution": {
                    "english": int(np.sum(frame_labels == 0)),
                    "hindi": int(np.sum(frame_labels == 1)),
                    "other": int(np.sum(frame_labels == 2))
                }
            }
            metadata_path = PROCESSED_DIR / f"{file_id}_metadata.json"
            with open(metadata_path, 'w') as f:
                json.dump(metadata, f, indent=2)
            
            processed += 1
        except:
            pass
    
    print(f"{processed} files")
    return processed

def main():
    """Main processing loop"""
    print(f"{'='*60}")
    print("Text-Based Alignment & Frame-Level Labeling")
    print(f"{'='*60}")
    
    INTERIM_DIR.mkdir(parents=True, exist_ok=True)
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    
    total = 0
    total += process_dataset('adult')
    total += process_dataset('children')
    
    print(f"\n{'='*60}")
    print(f"Total: {total} files processed")
    print(f"Output: {INTERIM_DIR} and {PROCESSED_DIR}")
    print(f"{'='*60}\n")

if __name__ == "__main__":
    main()
