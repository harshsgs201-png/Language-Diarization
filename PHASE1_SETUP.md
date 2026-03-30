# Phase 1 Setup Guide: Alignment & Frame-Level Labeling

## Quick Start (5 minutes)

### Step 1: Install Dependencies
On your lab PC (connected via SSH in VS Code), run:

```bash
cd C:\Users\harsh\Language-Diarization
python -m pip install --upgrade pip setuptools wheel
pip install whisper-timestamped openai-whisper fuzzywuzzy python-Levenshtein librosa scipy pandas tqdm
```

**Installation time:** 3-5 minutes (Whisper model ~140MB will download on first run)

### Step 2: Run the Alignment Script
```bash
python src/align_and_label.py
```

**Processing time:** ~30 seconds per audio file (depends on file length and GPU availability)

### Step 3: Verify Outputs
After completion, check:
```bash
dir data/interim          # Should see *_aligned.csv files
dir data/processed        # Should see *_labels.npy and *_metadata.json files
```

---

## What the Script Does

### Input Files
- **JSON Annotations:** `data/raw/Corpus/{adult|children}/annotations/code_switched_labels.json`
  - Contains: file path, transcription, code-switching label
- **Audio Files:** `data/raw/Corpus/{adult|children}/audio/{train|val|test}_split/`

### Processing Steps

1. **Word-Level Timestamp Alignment**
   - Loads Whisper 'base' model
   - Transcribes each audio file with word-level timestamps
   - Fuzzy-matches transcription words to Whisper output
   - Saves `{file_id}_aligned.csv` with columns:
     ```
     word | language | start_time | end_time | confidence | fuzzy_score
     ```

2. **Language Detection (Per Word)**
   - Heuristic: Detects Devanagari script → Hindi (HI)
   - Detects English alphanumeric → English (EN)
   - Falls back to `OTHER` for mixed/punctuation

3. **Frame-Level Label Generation (50Hz)**
   - Converts word boundaries to 20ms frames (50Hz)
   - Maps to: `0=English | 1=Hindi | 2=Other/Silence`
   - Saves as `{file_id}_labels.npy` (numpy array)
   - Saves metadata JSON with counts

### Output Files

```
data/interim/
├── {file_id}_aligned.csv          # Word-level alignment with timestamps & language tags

data/processed/
├── {file_id}_labels.npy           # Frame-level labels (50Hz), shape: (num_frames,)
├── {file_id}_metadata.json        # Duration, frame count, label distribution
```

---

## Example Output

### Aligned CSV (data/interim/AD09002_aligned.csv)
```
word,language,start_time,end_time,confidence,fuzzy_score
So,EN,0.0,0.3,0.99,100.0
my,EN,0.3,0.6,0.98,100.0
favourite,EN,0.6,1.0,0.97,100.0
festival,EN,1.0,1.5,0.98,100.0
है,HI,1.5,1.8,0.96,95.0
कि,HI,1.8,2.0,0.95,93.0
that,EN,2.0,2.3,0.99,100.0
```

### Metadata JSON (data/processed/AD09002_metadata.json)
```json
{
  "file_id": "AD09002",
  "dataset": "adult",
  "duration": 5.234,
  "frame_rate": 50,
  "num_frames": 262,
  "label_distribution": {
    "english": 120,
    "hindi": 100,
    "other": 42
  }
}
```

---

## Troubleshooting

### Error: "whisper_timestamped not installed"
```bash
pip install whisper-timestamped
```

### Error: "CUDA out of memory" (if using GPU)
Set `MODEL_SIZE = "tiny"` in the script (faster but less accurate)

### Error: "No audio files found"
- Verify audio files exist in `data/raw/Corpus/{adult|children}/audio/{train|val|test}_split/`
- Check that JSON paths match actual folder structure

### Warning: "No matches found between labels and Whisper"
- This is expected for very noisy audio
- Script falls back to Whisper word detection with language heuristics
- Verify the matched words make sense by checking the CSV file

---

## Performance Notes

| Property          | Value           |
|-------------------|-----------------|
| Model Size        | base (140MB)    |
| Frame Rate        | 50Hz (20ms)     |
| Processing Speed  | ~30-60 sec/min  |
| GPU Acceleration  | Yes (CUDA/MPS)  |
| Accuracy          | ~95% (base model)|

---

## Next Steps (Phase 2)

After alignment is complete, you'll:
1. Extract Mel-spectrograms from audio files (with sync'd frame-level labels)
2. Create PyTorch DataLoader for CRNN training
3. Train language diarization model
4. Evaluate with LDER (Language Error Detection Rate)

See `src/data_prep.py` for Mel-spectrogram extraction.

---

## Manual Quality Check

To verify alignment quality for 1-2 samples:

```python
import pandas as pd
import numpy as np

# Load aligned words and labels
aligned_df = pd.read_csv('data/interim/AD09002_aligned.csv')
labels = np.load('data/processed/AD09002_labels.npy')

print(f"Total words aligned: {len(aligned_df)}")
print(f"Total frames: {len(labels)}")
print(f"Label distribution:")
print(f"  English: {np.sum(labels == 0)} frames")
print(f"  Hindi: {np.sum(labels == 1)} frames")
print(f"  Other: {np.sum(labels == 2)} frames")
print(f"\nFirst 10 aligned words:")
print(aligned_df.head(10))
```

This confirms timestamps are correct and frame labels match word timings.
