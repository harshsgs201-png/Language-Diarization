"""
SMOKE TEST: Single-file inference pipeline
Tests: Load pre-extracted embeddings → Diarizer prediction → RTTM output
"""
import torch
import numpy as np
from scipy.ndimage import median_filter
import warnings
warnings.filterwarnings("ignore")

# Import our model
from model import XLSRDiarizer

# --- Configuration ---
MODEL_PATH = "models/weights/xlsr_diarizer_best.pt"
TEST_EMBEDDING = "data/processed/AD09001_emb.npy"
TEST_LABEL = "data/processed/AD09001_labels.npy"
OUTPUT_RTTM = "outputs/test_output.rttm"

HOP_LENGTH = 0.01  # 10ms → 0.01s per frame
WINDOW_SIZE = 7  # 70ms smoothing (7 frames)
CLASS_MAP = {0: "English", 1: "Hindi", 2: "Other"}

def load_test_embedding(emb_path, label_path):
    """Load pre-extracted XLSR embeddings and labels"""
    print(f"  Loading embedding: {emb_path}...")
    embeddings = np.load(emb_path)  # (366, 1024)
    labels = np.load(label_path)    # (366,)
    
    print(f"  Embedding shape: {embeddings.shape}")
    print(f"  Label shape: {labels.shape}")
    print(f"  Label distribution: {np.bincount(labels.astype(int))}")
    
    # Convert to torch tensor
    embeddings = torch.FloatTensor(embeddings).unsqueeze(0)  # (1, 366, 1024)
    
    return embeddings

def frames_to_rttm(smoothed_preds):
    """Convert frame predictions to RTTM segments (10ms per frame)"""
    segments = []
    current_class = int(smoothed_preds[0])
    start_frame = 0
    
    for i, pred in enumerate(smoothed_preds):
        pred = int(pred)
        if pred != current_class:
            segments.append({
                "start": start_frame * HOP_LENGTH,
                "end": i * HOP_LENGTH,
                "label": CLASS_MAP[current_class]
            })
            current_class = pred
            start_frame = i
    
    # Add final segment
    segments.append({
        "start": start_frame * HOP_LENGTH,
        "end": len(smoothed_preds) * HOP_LENGTH,
        "label": CLASS_MAP[current_class]
    })
    
    return segments

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n{'='*60}")
    print(f"SMOKE TEST: XLSR Diarizer Inference Pipeline")
    print(f"{'='*60}")
    print(f"Device: {device}")
    print(f"Model: {MODEL_PATH}")
    print(f"Test embedding: {TEST_EMBEDDING}\n")
    
    # Step 1: Load pre-extracted XLSR embeddings
    print("Step 1: Loading pre-extracted XLSR Embeddings...")
    embeddings = load_test_embedding(TEST_EMBEDDING, TEST_LABEL)
    
    # Step 2: Load diarizer model
    print("\nStep 2: Loading XLSRDiarizer model...")
    model = XLSRDiarizer(input_dim=1024, hidden_dim=256, num_classes=3).to(device)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.eval()
    print(f"  Model loaded: {MODEL_PATH}")
    
    # Step 3: Inference
    print("\nStep 3: Running inference...")
    embeddings = embeddings.to(device)
    with torch.no_grad():
        logits = model(embeddings)  # (1, 366, 3)
    
    # Get predictions (argmax over class dimension)
    raw_preds = torch.argmax(logits, dim=-1).squeeze().cpu().numpy()
    print(f"  Raw predictions shape: {raw_preds.shape}")
    print(f"  Raw predictions (first 20): {raw_preds[:20]}")
    
    # Get prediction probabilities
    probs = torch.softmax(logits, dim=-1).squeeze().cpu().numpy()
    print(f"  Max probability (first 20): {probs[:20].max(axis=1)}")
    
    # Step 4: Smoothing
    print("\nStep 4: Applying median filter smoothing...")
    smoothed_preds = median_filter(raw_preds, size=WINDOW_SIZE)
    print(f"  Smoothed predictions (first 20): {smoothed_preds[:20]}")
    
    # Step 5: Convert to RTTM segments
    print("\nStep 5: Converting to RTTM format...")
    segments = frames_to_rttm(smoothed_preds)
    
    # Print preview
    print(f"\n{'='*60}")
    print("PREVIEW: First 10 Segments")
    print(f"{'='*60}")
    for i, seg in enumerate(segments[:10]):
        duration = seg['end'] - seg['start']
        print(f"  [{seg['start']:6.2f}s → {seg['end']:6.2f}s] ({duration:6.2f}s) : {seg['label']}")
    
    # Step 6: Save RTTM
    print(f"\nStep 6: Saving RTTM...")
    with open(OUTPUT_RTTM, 'w') as f:
        file_id = "AD09001"
        for seg in segments:
            if seg['label'] != "Other":  # Skip "Other" in RTTM
                duration = seg['end'] - seg['start']
                f.write(f"SPEAKER {file_id} 1 {seg['start']:.3f} {duration:.3f} <NA> <NA> {seg['label']} <NA> <NA>\n")
    
    print(f"\n{'='*60}")
    print(f"✅ SMOKE TEST PASSED")
    print(f"{'='*60}")
    print(f"Output saved: {OUTPUT_RTTM}")
    print(f"Total segments: {len(segments)}")
    print(f"Duration: {segments[-1]['end']:.2f}s")
    
    # Show actual RTTM content
    print(f"\n{'='*60}")
    print("RTTM Output Preview:")
    print(f"{'='*60}")
    with open(OUTPUT_RTTM, 'r') as f:
        lines = f.readlines()
        for line in lines[:10]:
            print(line.strip())

if __name__ == "__main__":
    main()
