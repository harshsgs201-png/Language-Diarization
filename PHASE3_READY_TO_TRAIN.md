# Phase 3 Training - All Systems Ready! 🚀

## Current Status

### ✅ Extraction In Progress
- **Terminal**: Running `src/extract_xlsr.py`
- **Progress**: 122/5,124 files done (2.4%)
- **Rate**: 1.88 sec/file
- **ETA**: ~2.5 hours

**Why it took time to fix:**
- Initial issue: HuggingFace cache tried to download to C: drive (which was full)
- **Solution**: Redirected cache to D: drive (406+ GB free)
- **Result**: Extraction now proceeding smoothly

---

## Phase 3 Pipeline (Ready to Execute)

### Step 1: Wait for Extraction to Complete ⏳
```bash
# Currently running in background
# Check progress periodically
python -c "import os; print(len([f for f in os.listdir('data/processed') if f.endswith('_emb.npy')]))"
```

**When complete**: Should show 5,167 files

---

### Step 2: Run Verification ✅ (Already Prepared)
```bash
python src/verify_data.py
```

**What it does:**
- Loads all extracted embeddings
- Checks for NaN values (should be 0)
- Checks for all-zero files (should be 0)
- Computes actual class distribution
- Calculates optimal class weights
- **Output example**:
  ```
  English (0):        353,368 frames (37.18%)
  Hindi (1):          560,215 frames (58.94%)
  Other/Silence (2):   36,871 frames ( 3.88%)
  
  Class Weights (Inverse Frequency):
  English (0):       0.089169
  Hindi (1):         0.056245
  Other/Silence (2): 0.854586
  ```

---

### Step 3: Train Model ✅ (Already Prepared)
```bash
python src/train.py
```

**What it does:**
1. Verifies embeddings (calls verify_extraction internally)
2. Loads and splits dataset (90% train / 10% val)
3. Creates DataLoader with `padding_value=-100`
4. Initializes XLSRDiarizer model (4.2M params)
5. Creates `CrossEntropyLoss(weight=weights, ignore_index=-100)`
6. **Trains for max 20 epochs** with:
   - AdamW optimizer (lr=1e-3, weight_decay=1e-4)
   - Gradient clipping (max_norm=5.0)
   - ReduceLROnPlateau scheduler
   - Early stopping (patience=5)
7. **Saves best model** to `models/weights/xlsr_diarizer_best.pt`

**Expected Output**:
```
Training on cpu...

Step 1: Verifying Phase 2 embeddings...
   ✅ Phase 2 verification passed!

Step 2: Loading dataset...
   Total samples: 5167
   Training samples: 4650
   Validation samples: 517

Step 3: Initializing model...
   Model parameters: 4,204,548

Step 4: Setting up loss function and optimizer...
   Class weights: [0.0892, 0.0562, 0.8546]
   Optimizer: AdamW (lr=1e-3)
   Loss: CrossEntropyLoss with ignore_index=-100

TRAINING STARTED

Epoch 1/20 [Train]: 100%|██████████| 290/290 [00:45<00:00, 6.35 batches/s]
   Train Loss: 0.8234 | Val Loss: 0.7145
   ✅ Saved best model! (loss: 0.7145)
...
Epoch 12/20 [Train]: 100%|██████████| 290/290 [00:47<00:00, 6.12 batches/s]
   Train Loss: 0.2134 | Val Loss: 0.2987
   ⏹️  Early stopping triggered (patience=5)

TRAINING COMPLETED
✅ Best model saved to: models/weights/xlsr_diarizer_best.pt
   Best validation loss: 0.2987
```

**Training Time**:
- CPU: 10-12 hours
- GPU (CUDA):  30-45 minutes

---

## Key Design Decisions Explained

### 1. Padding with -100 (Not 2)
```python
# ❌ OLD (corrupts class distribution):
padded_labels = pad_sequence(labels, batch_first=True, padding_value=2)
# Now 50% of batch is "Other" labels - model overfits!

# ✅ NEW (leverages PyTorch built-in):
padded_labels = pad_sequence(labels, batch_first=True, padding_value=-100)
# CrossEntropyLoss.ignore_index=-100 automatically skips gradients from padding
```

**Why it works:**
- PyTorch's loss function is hardcoded to ignore `-100` during backprop
- Only original frames (never padding) affect gradients
- Maintains perfect class balance in training signal

---

### 2. Class Weights from Verification
```python
# Computed from ORIGINAL labels only (before padding):
class_weights = {
    0: 0.089  # English (minority, higher weight)
    1: 0.056  # Hindi (majority, lower weight)
    2: 0.855  # Other (rarity, very high weight!)
}

# Each class contributes equally to gradient:
# EN error (0.089 weight) × 353k frames = 31.5k gradient units
# HI error (0.056 weight) × 560k frames = 31.4k gradient units
# OT error (0.855 weight) × 37k frames = 31.6k gradient units
```

**Why we compute class weights:**
- Raw accuracy: Hindi always wins (59% of data)
- With weights: All languages learned equally

---

### 3. SpecAugment Masking
Applied only during training (not validation):
```python
# Time Masking: Zero out 1-20 consecutive frames
# Effect: Model robust to dropped audio / packet loss

# Feature Masking: Zero out 1-100 embedding channels
# Effect: Model learns to use multiple XLSR features, not just one
```

---

## File Checklist

| File | Status | Purpose |
|------|--------|---------|
| `src/extract_xlsr.py` | ✅ Running | Extract XLSR embeddings |
| `src/verify_data.py` | ✅ Ready | Validate embeddings & compute weights |
| `src/dataset.py` | ✅ Ready | DataLoader with `-100` padding |
| `src/model.py` | ✅ Ready | XLSRDiarizer architecture (Bi-LSTM + attention) |
| `src/train.py` | ✅ Ready | Complete training loop with verification |

---

## Next Actions (Exact Order)

1. **Monitor extraction** (run while you sleep/work)
   ```bash
   # Check progress in new terminal
   while ($true) { 
     python -c "import os; c=len([f for f in os.listdir('data/processed') if f.endswith('_emb.npy')]); print(\"`n[$(Get-Date -Format 'HH:mm:ss')] Embeddings: \$c/5167 ($('{0:P0}' -f ($c/5167)))\"); 
     Start-Sleep -Seconds 300 
   }
   ```

2. **Once extraction complete**, run:
   ```bash
   python src/verify_data.py
   ```

3. **After verification passes**, run:
   ```bash
   python src/train.py
   ```

4. **Monitor training**, expected output:
   - Loss should decrease each epoch
   - Val loss should eventually improve
   - Early stopping at epoch 10-15 typically

---

## Troubleshooting

| Problem | Solution |
|---------|----------|
| Extraction stuck | Check task manager: python.exe using CPU? |
| Extraction out of space | D: has 406GB free - you're fine! |
| Verify shows NaN | Rerun extraction (might be partial files) |
| Training too slow | This is CPU - normal for 5.1K files |
| Training stops early | Early stopping works - this is expected! |
| OOM error during training | Reduce batch_size from 16 to 8 in train.py |

---

## Summary

✅ **All code prepared and ready**
✅ **Extraction running (ETA 2.5 hours)**
⏳ **Waiting on Phase 2 to complete**

Once extraction finishes:
1. Run verify_data.py (1 min)
2. Run train.py (10-12 hours on CPU)
3. Evaluate trained model

**Status: Everything is automated - just wait for extraction!**
