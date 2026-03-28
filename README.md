# Language-Diarization

A deep learning project for automatic language diarization using CRNN (Convolutional Recurrent Neural Network) architecture.

## Project Structure

```
Language-Diarization/
├── data/
│   ├── raw/                 # Original zips and audio files
│   ├── interim/             # Chunked audio and Whisper transcripts
│   └── processed/           # Extracted Mel-spectrograms (.npy) and labels
├── src/
│   ├── data_prep.py         # Audio resampling and Mel-spec extraction
│   ├── dataset.py           # PyTorch Dataset and DataLoader classes
│   ├── model.py             # CRNN architecture
│   ├── train.py             # Training loop and loss functions
│   └── evaluate.py          # pyannote.metrics calculation (LDER, JER)
├── models/
│   └── saved_weights/       # .pt or .pth files
├── requirements.txt
└── README.md
```

## Setup

### 1. Clone the repository
```bash
git clone https://github.com/harshsgs201-png/Language-Diarization.git
cd Language-Diarization
```

### 2. Create virtual environment
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

## Usage

### Data Preparation
Prepare your audio files in `data/raw/` and extract Mel-spectrograms:
```bash
python src/data_prep.py
```

### Training
Train the model on prepared data:
```bash
python src/train.py --batch_size 32 --num_epochs 100 --learning_rate 0.001
```

### Evaluation
Evaluate the trained model:
```bash
python src/evaluate.py
```

## Requirements
- Python 3.9+
- PyTorch 2.0+
- Librosa for audio processing
- pyannote for evaluation metrics

See `requirements.txt` for full dependencies.

## Model Architecture

The model uses a CRNN architecture:
- **Convolutional layers**: Extract spatial features from Mel-spectrograms
- **LSTM layers**: Capture temporal dependencies
- **Dense output layer**: Prediction layer

## Notes
- Ensure your data follows the expected format (16kHz, mono audio)
- GPU recommended for faster training
- All model weights are saved to `models/saved_weights/`