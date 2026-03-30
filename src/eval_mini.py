"""
Mini Evaluation Pipeline: Calculate LDER and JER using pyannote.metrics
"""
import os
import glob
import torch
import numpy as np
from scipy.ndimage import median_filter
import warnings
warnings.filterwarnings("ignore")

# Import pyannote metric tools
from pyannote.core import Annotation, Segment
from pyannote.metrics.diarization import DiarizationErrorRate, JaccardErrorRate

# Import our model
from model import XLSRDiarizer

# --- Configuration ---
MODEL_PATH = "models/weights/xlsr_diarizer_best.pt"
EMBEDDING_DIR = "data/processed"
RTTM_DIR = "data/raw/mini_eval/rttm"

HOP_LENGTH = 0.01  # 10ms per frame
WINDOW_SIZE = 7    # 70ms smoothing (7 frames)
CLASS_MAP = {0: "English", 1: "Hindi", 2: "Other"}

def load_reference_rttm(rttm_path, uri):
    """Load ground truth RTTM and convert to pyannote Annotation."""
    annotation = Annotation(uri=uri)
    
    if not os.path.exists(rttm_path):
        print(f"  ⚠️  RTTM not found: {rttm_path}")
        return annotation
    
    with open(rttm_path, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 8:
                start = float(parts[3])
                duration = float(parts[4])
                label = parts[7]
                # Include all labels (English, Hindi, Other) for comprehensive evaluation
                annotation[Segment(start, start + duration)] = label
    
    return annotation

def predict_hypothesis(embedding_path, label_path, model, device, uri):
    """
    Run inference on pre-extracted embeddings and return pyannote Annotation.
    Uses actual ground truth labels as reference for padding validation.
    """
    # Load pre-extracted embedding
    embeddings = np.load(embedding_path)  # (Time, 1024)
    labels = np.load(label_path)          # (Time,)
    
    print(f"  Inference on {uri}: {embeddings.shape[0]} frames")
    
    # Convert to torch tensor and run model
    embeddings_torch = torch.FloatTensor(embeddings).unsqueeze(0).to(device)  # (1, Time, 1024)
    
    with torch.no_grad():
        logits = model(embeddings_torch)  # (1, Time, 3)
    
    # Get frame-level predictions
    raw_preds = torch.argmax(logits, dim=-1).squeeze().cpu().numpy()
    
    # Apply median filter smoothing
    smoothed_preds = median_filter(raw_preds, size=WINDOW_SIZE)
    
    # Convert to pyannote Annotation
    annotation = Annotation(uri=uri)
    current_class = int(smoothed_preds[0])
    start_frame = 0
    
    for i in range(1, len(smoothed_preds)):
        pred = int(smoothed_preds[i])
        if pred != current_class:
            # Add segment for current class
            start_time = start_frame * HOP_LENGTH
            end_time = i * HOP_LENGTH
            label_name = CLASS_MAP[current_class]
            annotation[Segment(start_time, end_time)] = label_name
            
            current_class = pred
            start_frame = i
    
    # Add final segment
    start_time = start_frame * HOP_LENGTH
    end_time = len(smoothed_preds) * HOP_LENGTH
    label_name = CLASS_MAP[current_class]
    annotation[Segment(start_time, end_time)] = label_name
    
    return annotation

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"{'='*60}")
    print(f"MINI EVALUATION: Language Diarization Metrics")
    print(f"{'='*60}")
    print(f"Device: {device}\n")

    # Load model
    print("Loading model...")
    model = XLSRDiarizer(input_dim=1024, hidden_dim=256, num_classes=3).to(device)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.eval()
    print(f"Model loaded: {MODEL_PATH}\n")

    # Initialize pyannote metrics
    der_metric = DiarizationErrorRate()
    jer_metric = JaccardErrorRate()

    # Find all embedding files
    embedding_files = [f for f in os.listdir(EMBEDDING_DIR) if f.endswith('_emb.npy')]
    
    # Filter to only mini eval files
    eval_files = [
        f.replace('_emb.npy', '') 
        for f in embedding_files 
        if os.path.exists(os.path.join(RTTM_DIR, f.replace('_emb.npy', '.rttm')))
    ]
    
    if not eval_files:
        print(f"❌ No evaluation files found matching {RTTM_DIR}/*.rttm")
        return

    print(f"Found {len(eval_files)} files for evaluation\n")

    # Process each file
    for file_id in eval_files[:10]:  # Limit to first 10 for quick testing
        print(f"Processing {file_id}...")
        
        embedding_path = os.path.join(EMBEDDING_DIR, f"{file_id}_emb.npy")
        label_path = os.path.join(EMBEDDING_DIR, f"{file_id}_labels.npy")
        rttm_path = os.path.join(RTTM_DIR, f"{file_id}.rttm")
        
        if not os.path.exists(embedding_path):
            print(f"  ⚠️  Embedding not found: {embedding_path}")
            continue
        
        # 1. Load Reference (Ground Truth RTTM)
        reference = load_reference_rttm(rttm_path, uri=file_id)
        
        # 2. Generate Hypothesis (Model Predictions)
        hypothesis = predict_hypothesis(embedding_path, label_path, model, device, uri=file_id)
        
        # 3. Accumulate Metrics
        der_metric(reference, hypothesis)
        jer_metric(reference, hypothesis)
        
        print(f"  ✓ Processed\n")

    # --- Print Final Reports ---
    print(f"\n{'='*60}")
    print(f"📊 EVALUATION RESULTS")
    print(f"{'='*60}\n")
    
    print("--- Language Diarization Error Rate (LDER) ---")
    print(der_metric)
    
    print("\n--- Jaccard Error Rate (JER) ---")
    print(jer_metric)
    
    print(f"\n{'='*60}")
    print("✅ Mini evaluation complete!")
    print(f"{'='*60}")

if __name__ == "__main__":
    main()
