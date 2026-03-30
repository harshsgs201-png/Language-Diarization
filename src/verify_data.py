"""
Phase 2 Verification: Sanity check for XLSR embeddings before training.
"""

import os
import glob
import numpy as np

def verify_extraction(processed_dir="data/processed"):
    """Verify that XLSR embeddings are healthy: correct shapes, no NaNs, no zeros."""
    
    emb_files = sorted(glob.glob(os.path.join(processed_dir, "*_emb.npy")))
    lbl_files = sorted(glob.glob(os.path.join(processed_dir, "*_labels.npy")))
    
    print(f"\nFound {len(emb_files)} embeddings and {len(lbl_files)} label files.\n")
    assert len(emb_files) == len(lbl_files), "Mismatch in file counts!"
    
    # Sanity check counters
    nan_count = 0
    zero_count = 0
    shape_mismatch_count = 0
    class_counts = {0: 0, 1: 0, 2: 0}

    # Verify each file
    for emb_f, lbl_f in zip(emb_files, lbl_files):
        emb = np.load(emb_f)
        lbl = np.load(lbl_f)
        
        # 1. Shape Verification
        if emb.shape[0] != lbl.shape[0]:
            print(f"⚠️  Shape mismatch in {os.path.basename(emb_f)}: emb={emb.shape} vs lbl={lbl.shape}")
            shape_mismatch_count += 1
            continue
            
        if emb.shape[1] != 1024:
            print(f"⚠️  Embedding dimension is not 1024 in {os.path.basename(emb_f)}: {emb.shape[1]}")
            shape_mismatch_count += 1
            continue
        
        # 2. NaN and Zero Verification
        if np.isnan(emb).any():
            nan_count += 1
            print(f"⚠️  NaN values found in {os.path.basename(emb_f)}")
            
        if not np.any(emb):
            zero_count += 1
            print(f"⚠️  All-zero values found in {os.path.basename(emb_f)}")
            
        # 3. Class Distribution (only count original labels, not padding)
        unique, counts = np.unique(lbl, return_counts=True)
        for u, c in zip(unique, counts):
            if u in class_counts:  # Ignore -100 (padding marker)
                class_counts[u] += c

    # Print verification report
    print("="*60)
    print("PHASE 2 VERIFICATION REPORT")
    print("="*60)
    print(f"\n✓ Total Embeddings: {len(emb_files)}")
    print(f"✓ Total Labels: {len(lbl_files)}")
    print(f"\n📊 Data Quality Checks:")
    print(f"   Shape Matching:        {'PERFECT ✓' if shape_mismatch_count == 0 else f'{shape_mismatch_count} MISMATCHES ⚠️'}")
    print(f"   NaN Files:             {nan_count if nan_count > 0 else '0 ✓'}")
    print(f"   Zero-out Files:        {zero_count if zero_count > 0 else '0 ✓'}")
    
    print(f"\n🗣️  Class Distribution (Original, Unpadded):")
    total_frames = sum(class_counts.values())
    class_names = ["English (0)", "Hindi (1)", "Other/Silence (2)"]
    for cls in range(3):
        count = class_counts.get(cls, 0)
        pct = 100.0 * count / total_frames if total_frames > 0 else 0
        print(f"   {class_names[cls]:25s}: {count:8,} frames ({pct:6.2f}%)")
    
    print(f"\n$ Total Frames (All Classes): {total_frames:,}")
    
    # Calculate and print class weights
    print(f"\n⚖️  Recommended Class Weights (Inverse Frequency):")
    weights = []
    for cls in range(3):
        count = class_counts.get(cls, 1)
        if count > 0:
            weight = total_frames / (3 * count)
        else:
            weight = 1.0
        weights.append(weight)
    
    # Normalize
    weights_sum = sum(weights)
    normalized_weights = [w / weights_sum for w in weights]
    
    for cls, (w, nw) in enumerate(zip(weights, normalized_weights)):
        print(f"   {class_names[cls]:25s}: {nw:.6f}")
    
    print("\n" + "="*60)
    print(f"✅ PHASE 2 STATUS: {'READY FOR TRAINING ✓' if (nan_count == 0 and zero_count == 0 and shape_mismatch_count == 0) else 'NEEDS DEBUGGING ⚠️'}")
    print("="*60 + "\n")
    
    return {
        'total_embeddings': len(emb_files),
        'total_labels': len(lbl_files),
        'nan_count': nan_count,
        'zero_count': zero_count,
        'shape_mismatches': shape_mismatch_count,
        'class_counts': class_counts,
        'total_frames': total_frames,
        'class_weights': normalized_weights
    }

if __name__ == "__main__":
    verify_extraction()
