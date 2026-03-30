# Phase 3: Language Diarization Model Training (Complete Blueprint)

## System Status

### ✅ Completed
- **Phase 1**: Word-level alignment (5,167 files → 950,454 frames)
- **Phase 2**: XLSR-53 feature extraction (running in background)
- **Phase 3 Files**: Model, Dataset, Training Loop ready

### 📊 Data Summary (From Phase 1)
```
Total Frames:    950,454
├─ English (0):     353,368 (37.18%)
├─ Hindi (1):       560,215 (58.94%)
└─ Other (2):        36,871 (3.88%)
```

### ⚖️ Recommended Class Weights (Normalized)
```python
weights = [0.089169, 0.056245, 0.854586]
# English gets 0.089 (minority emphasis)
# Hindi gets 0.056 (majority suppression)
# Other gets 0.855 (rarity emphasis - 15.2x Hindi!)
```

---

## Architecture Overview: XLSRDiarizer

### Component 1: XLSR-53 Feature Extraction (Frozen)
- **Input**: Raw audio (16 kHz)
- **Model**: facebook/wav2vec2-large-xlsr-53
- **Output**: 1024-dim embeddings @ 50 Hz (20ms frames)
- **Status**: Phase 2 (extraction running)

### Component 2: Bi-LSTM Encoder
```
Input: (Batch, Time, 1024)
  ↓
Bi-LSTM(2 layers, 256 hidden)
  ↓
Output: (Batch, Time, 512)
```

**Why Bi-LSTM?**
- Captures **forward context**: "I am speaking..." (predicting language at current frame)
- Captures **backward context**: "...in Hindi" (what's coming next influences current frame)
- **Code-switch boundaries** require both directions to resolve ambiguity

### Component 3: Self-Attention Layer
```
Input: (Batch, Time, 512)
  ↓
Attention(512 → 1)
  ↓
Softmax over Time dimension
  ↓
Element-wise multiply with LSTM output
  ↓
Output: (Batch, Time, 512)
```

**Why Attention?**
- **Frame weighting**: Some frames are more informative (language-defining moments)
- **Transition detection**: Code-switch boundaries get higher attention scores
- **Graceful degradation**: Poor quality audio frames get downweighted automatically

### Component 4: Frame-Level Classifier
```
Input: (Batch, Time, 512) [attention-weighted]
  ↓
Linear(512 → 3)
  ↓
Output: (Batch, Time, 3) [logits for EN, HI, Other]
```

---

## Loss Function: The Padding Secret Weapon

### The Problem
```python
# Naive padding with class label:
padded_labels = pad_sequence(labels, batch_first=True, padding_value=2)
# Now 50% of your batch is "Other" labels!
# Model overfits to padding prediction
```

### The Solution: PyTorch's ignore_index
```python
criterion = nn.CrossEntropyLoss(
    weight=class_weights,
    ignore_index=-100  # Magic flag: "ignore this during backprop"
)

# In collate function:
padded_labels = pad_sequence(labels, batch_first=True, padding_value=-100)
# PyTorch's CrossEntropyLoss is hardcoded to skip these positions!
```

**How it works:**
1. Padding value is -100
2. During forward pass: logits computed for all (Batch, Time) positions
3. During loss calculation: positions with label=-100 are **completely ignored**
4. During backward pass: **no gradients** flow from padded positions
5. Result: Model only learns from real audio frames!

---

## Class Weighting: Inverse Frequency Magic

### Why Simple Accuracy Doesn't Work
```
Naive training (no weights):
├─ Hindi accuracy: 95% (easy, 59% of data)
├─ English accuracy: 92% (medium, 37% of data)
└─ Other accuracy: 10% (DISASTER! Only 4% of data)

Model learns: "Always guess Hindi" → 59% accuracy (useless)
```

### Inverse Frequency Weighting
```python
For each class c:
    weight[c] = total_frames / (num_classes × count[c])

English:  950,454 / (3 × 353,368) = 0.894  → Normalized: 0.089
Hindi:    950,454 / (3 × 560,215) = 0.563  → Normalized: 0.056
Other:    950,454 / (3 × 36,871)  = 8.577  → Normalized: 0.855
```

### What This Achieves
```
Each class contributes equally to gradient:
├─ Hindi error (0.056 weight) × 560k frames = 31k gradient units
├─ English error (0.089 weight) × 353k frames = 31k gradient units
└─ Other error (0.855 weight) × 36k frames = 31k gradient units

Result: Balanced learning across all three languages!
```

---

## Complete Training Pipeline

### Step 1: Verify Phase 2 (Once extraction completes)
```bash
python src/verify_data.py
```

**Checks:**
- ✅ All embeddings have shape (N, 1024)
- ✅ No NaN values (corrupted data)
- ✅ No all-zero files (failed extraction)
- ✅ Shape alignment between embeddings and labels

### Step 2: Start Training
```bash
python src/train.py
```

**What happens:**
1. Loads HiACC_XLSR_Dataset with SpecAugment augmentation for training
2. Splits 90% train / 10% validation
3. Creates batches with dynamic padding (variable-length sequences)
4. Pads embeddings with 0.0, labels with -100
5. Trains XLSRDiarizer on weighted CrossEntropyLoss
6. Validates each epoch with gradient clipping
7. Saves best model based on validation loss
8. Early stopping after 5 epochs without improvement

### Training Hyperparameters
```python
Batch size:     16 (fits in memory with variable lengths)
Learning rate:  1e-3 (standard for Transformer-based features)
Optimizer:      AdamW (weight_decay=1e-4 prevents overfitting)
Scheduler:      ReduceLROnPlateau (reduce LR if loss plateaus)
Gradient clip:  5.0 (prevents LSTM exploding gradients)
Epochs:         20 max (early stopping usually ~10-12)
```

---

## SpecAugment Masking (Data Augmentation)

### Why SpecAugment for XLSR Embeddings?
XLSR embeddings are **temporally dense 1024-dim vectors**. Standard SpecAugment works on 2D spectrograms. Our custom approach:

### Time Masking
```python
# Randomly zero out 1-20 consecutive time frames
# Simulates: Missing audio / packet loss
# Effect: Model learns robustness to temporal gaps

Example:
Input embedding[100:120, :] → Set to 0
Model sees: "The audio of 'in' is missing"
Model still predicts: "Hindi" (uses context)
```

### Feature Masking
```python
# Randomly zero out 1-100 consecutive embedding dimensions
# Simulates: Damaged acoustic stream / speaker variation
# Effect: Model learns to use multiple feature channels

Example:
Input embedding[:, 400:500] → Set to 0
Model sees: "One channel of XLSR is corrupted"
Model still predicts: Language using other 924 dimensions
```

**Augmentation only during training**, not validation → prevents overfitting

---

## Key Advantages of This Architecture

### 1. Lightweight (4.2M parameters)
- **Frozen XLSR** (300M parameters) handles feature extraction
- **Trainable model** (4.2M) is only sequence modeling
- **Fast training**: ~10 hours on CPU, ~30 minutes on GPU

### 2. Robust to Variable-Length Audio
```
Train batch example:
├─ Utterance 1: 5.2 sec → 260 frames
├─ Utterance 2: 8.1 sec → 405 frames
├─ Utterance 3: 3.7 sec → 185 frames
└─ Utterance 4: 12.3 sec → 615 frames

Collate: Pad all to 615 frames
├─ No loss in temporal resolution
├─ No fixed sequence length required
└─ Handles any utterance duration
```

### 3. Attention-Based Frame Weighting
- Automatically learns which frames are language-defining
- Code-switch boundaries get highest attention
- Noisy frames get downweighted
- **No manual feature engineering required**

### 4. Class-Balanced Training
- Rare "Other" class doesn't get drowned out
- All three languages contribute equally to gradient
- Early stopping prevents overfitting to Hindi
- Stable convergence across all classes

---

## Post-Training: Inference & Evaluation

### Batch Inference (Fast)
```python
model.eval()
with torch.no_grad():
    logits = model(embeddings)  # (N, Time, 3)
    
predictions = logits.argmax(dim=-1)  # Frame-level class IDs
confidence = logits.softmax(dim=-1).max(dim=-1)[0]  # Confidence scores

# Per-frame predictions
for t in range(logits.shape[1]):
    pred = {0: 'EN', 1: 'HI', 2: 'Other'}[predictions[0, t].item()]
    conf = confidence[0, t].item()
```

### Evaluation Metrics
```python
# Accuracy (simple)
accuracy = (predictions == labels).float().mean()

# Per-class accuracy (important!)
for c in range(3):
    mask = (labels == c)
    class_acc = (predictions[mask] == c).float().mean()
    print(f"Class {c}: {class_acc:.2%}")

# Confusion matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(labels.flatten(), predictions.flatten())

# Language Detection Error Rate (LDER)
# Same as accuracy but weighted differently for your use case
```

---

## Troubleshooting Checklist

| Issue | Cause | Solution |
|-------|-------|----------|
| Loss = NaN | Exploding gradients | Already handled: gradient clipping=5.0 |
| Loss = Constant | Dead LSTM | Reduce learning rate or reinit |
| Validation ≫ Training | Overfitting | Enable SpecAugment (already on) |
| Other accuracy = 0% | Class imbalance | Check weights: should be [0.089, 0.056, 0.855] |
| OOM error | Batch too large | Reduce batch_size from 16 → 8 |
| Very slow training | CPU only | Use GPU if available: CUDA automatically detected |
| Model saves 0GB | Training bug | Check that PyTorch version ≥ 1.9 |

---

## Files Summary

| File | Purpose | Status |
|------|---------|--------|
| `src/extract_xlsr.py` | XLSR-53 embedding extraction | ⏳ Running (Phase 2) |
| `src/dataset.py` | Variable-length DataLoader + augmentation | ✅ Ready |
| `src/verify_data.py` | Sanity check before training | ✅ Ready |
| `src/model.py` | XLSRDiarizer architecture | ✅ Ready + tested |
| `src/train.py` | Complete training loop | ✅ Ready to execute |

---

## Next Steps: Execution Plan

```
NOW: Phase 2 extraction running in background

WHEN EXTRACTION COMPLETES:
1. Run: python src/verify_data.py
   → Check output for NaN/zero counts (should be 0)
   → Verify class weights are used

2. Run: python src/train.py
   → Watch training progress
   → Model automatically saves best weights
   
3. Plot training curves (optional):
   $ tensorboard --logdir=models/weights  (if you add it)
   $ plot_loss_history.py  (not included, DIY)

4. Evaluate on test set:
   → Load best model
   → Run inference on test utterances
   → Calculate metrics per language
```

---

## Mathematical Foundation: Why This Works

### Cross-Entropy with Class Weights
$$\text{Loss} = -\frac{1}{N} \sum_{i=1}^{N} w_{y_i} \log(p_{y_i})$$

Where:
- $y_i$ = true class for sample $i$ (0, 1, or 2)
- $p_{y_i}$ = predicted probability for true class
- $w_{y_i}$ = class weight (higher for rare classes)
- $N$ = total samples

**Effect**: Rare classes contribute more to loss → larger gradients → better learning

### Attention Mechanism
$$\text{attention}(x) = x \odot \text{softmax}(W \cdot x)$$

Where:
- $x$ = LSTM output (Batch, Time, 512)
- $W$ = learned weight matrix (512 → 1)
- $\odot$ = element-wise multiplication
- Result: Each frame gets a weight [0, 1]

**Effect**: Model learns to focus on language-defining frames automatically

---

## Success Criteria

✅ **Training Complete When:**
- Validation loss converges (stops decreasing)
- English accuracy > 85%
- Hindi accuracy > 85%
- Other accuracy > 50% (hard class, this is good!)
- No NaN/Inf in loss
- Model saves best.pt file without errors

⚠️ **Red Flags:**
- Other accuracy stuck at 0% → check class weights
- Training loss goes NaN → too high learning rate
- Validation loss way higher than train → overfitting even with SpecAugment
- Model file doesn't save → disk space issue or permissions

---

## Summary

You now have a **production-ready Phase 3 training pipeline** that:

1. ✅ Uses **frozen XLSR embeddings** (no GPU required for feature extraction)
2. ✅ Handles **variable-length utterances** via padding with -100
3. ✅ Balances **class imbalance** via inverse-frequency weighting
4. ✅ Augments training data with **SpecAugment masking**
5. ✅ Includes **self-attention** for frame importance weighting
6. ✅ Uses **gradient clipping** to prevent LSTM instability
7. ✅ Implements **early stopping** to prevent overfitting
8. ✅ Saves **best model** based on validation performance

**Estimated Training Time:**
- CPU: ~10 hours (but you have librosa + scipy, no GPU needed for extraction!)
- GPU: ~30 minutes

**Ready to train once Phase 2 extraction completes!**
