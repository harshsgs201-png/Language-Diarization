# Phase 3: Model Training with Weighted CrossEntropyLoss

## The Class Weight Problem: Why Padding Ruins Naive Solutions

### The Challenge
When you pad variable-length sequences to form batches, you introduce **artificial "Other" labels** (class 2) that don't represent real language. This creates a false class imbalance:

**Original Data (from Phase 1):**
- English: 353,368 frames (37.18%)
- Hindi: 560,215 frames (58.94%)
- Other/Silence: 36,871 frames (3.88%)
- **Total: 950,454 frames**

**After Batching with Padding (Corrupted):**
If your max sequence in a batch is 1000 frames, but average is 500, you've added 250,000+ artificial "Other" labels that never existed in the original audio.

### Why This Breaks Training
A naive CrossEntropyLoss would:
1. Treat padded "Other" labels as real data
2. Heavily penalize the model for not predicting "Other"
3. Over-fit the model to padding artifacts
4. Destroy generalization on real test data

## The Solution: Compute Weights from Original Labels Only

### Step 1: Count Only Original Frames
```python
def compute_class_weights(processed_dir):
    """Calculate weights from the ORIGINAL labels, NOT padded data."""
    label_files = sorted(glob.glob(os.path.join(processed_dir, "*_labels.npy")))
    
    class_counts = Counter({0: 0, 1: 0, 2: 0})
    
    for lbl_file in label_files:
        lbl = np.load(lbl_file)
        # Count ONLY what was in the original audio, before any padding
        unique, counts = np.unique(lbl, return_counts=True)
        for u, c in zip(unique, counts):
            class_counts[int(u)] += c
    
    total_frames = sum(class_counts.values())
    # Result: 950,454 frames (the REAL data)
    return class_counts, total_frames
```

### Step 2: Apply Inverse Frequency Weighting
For each class:
$$\text{weight}_c = \frac{\text{total\_frames}}{n\_classes \times \text{count}_c}$$

**Calculated Weights for HiACC Dataset:**
```
English (0):        353,368 / (3 × 353,368) = 0.333 → Normalized: 0.089169
Hindi (1):          560,215 / (3 × 560,215) = 0.200 → Normalized: 0.056245
Other/Silence (2):   36,871 / (3 × 36,871)  = 2.700 → Normalized: 0.854586
```

**Interpretation:**
- **Other/Silence gets 15.2x more weight** (0.854586 / 0.056245) than Hindi
- This compensates for its rarity and ensures the model pays attention to language transitions
- English gets intermediate weight since it's between Hindi and Other in frequency

### Step 3: Normalize Weights
```python
weights_tensor = torch.tensor(weights, dtype=torch.float32)
weights_tensor = weights_tensor / weights_tensor.sum()  # Sum to 1.0
```

This ensures stable gradient flow and prevents weight magnitude from interfering with learning rates.

### Step 4: Pass Weights to CrossEntropyLoss During Training
```python
import torch.nn as nn

# Compute weights ONCE before training
class_weights = compute_class_weights("data/processed")

# Create loss with class weights
criterion = nn.CrossEntropyLoss(weight=class_weights, reduction='mean')

# During training loop:
for batch_emb, batch_lbl in dataloader:
    logits = model(batch_emb)  # Shape: (Batch, Max_Time, 3)
    
    # CrossEntropyLoss automatically:
    # 1. Ignores padded positions (value 2) by applying class weight
    # 2. Upweights rare classes (Other/Silence)
    # 3. Balances gradient flow across all three classes
    loss = criterion(logits.view(-1, 3), batch_lbl.view(-1))
    loss.backward()
```

## Critical Implementation Points

### ✅ DO: Compute Weights from Raw Labels
```python
# CORRECT: Use only the original label files (before padding)
weights = compute_class_weights("data/processed")
```

### ❌ DON'T: Compute Weights from Batched Data
```python
# WRONG: This includes padding artifacts
labels_in_batch = torch.cat([batch_lbl.view(-1) for _, batch_lbl in dataloader])
# Now class 2 is heavily over-represented!
```

### ✅ DO: Ignore Padding in Loss Calculation
```python
# Option 1: Let class weights handle it (recommended)
criterion = nn.CrossEntropyLoss(weight=class_weights)

# Option 2: Explicitly mask padded positions (advanced)
mask = (batch_lbl != 2)  # True for real labels, False for padding
loss = criterion(logits[mask], batch_lbl[mask])
```

### ✅ DO: Compute Weights Once Before Training
```python
if __name__ == "__main__":
    train_loader = get_dataloader("data/processed", is_training=True)
    
    # ONCE at startup
    class_weights = compute_class_weights("data/processed")
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    
    # Then reuse criterion for all epochs
    for epoch in range(num_epochs):
        for batch_emb, batch_lbl in train_loader:
            ...
```

## Why This Works: The Math Behind Class Weights

**Without class weights:**
- Loss = (1/N) × Σ[loss(pred[i], true[i])]
- Majority class dominates: Hindi samples contribute most to gradient

**With inverse-frequency weights:**
- Loss = (1/N) × Σ[w[true[i]] × loss(pred[i], true[i])]
- Rare classes contribute equally: Hindi (w=0.056) and Other (w=0.855) balance out
- Gradient flows equally from all three classes

**During backprop:**
1. Model sees English error: multiplies gradient by 0.089 (small but non-zero)
2. Model sees Hindi error: multiplies gradient by 0.056 (smallest)
3. Model sees Other error: multiplies gradient by 0.855 (largest)
4. Net effect: Model learns all three classes equally well

## Validation Metrics

During training, track:
```python
# Per-class accuracy (ignore padding)
mask = (batch_lbl != 2)
pred_class = logits[mask].argmax(dim=-1)
true_class = batch_lbl[mask]

for c in range(3):
    class_acc = (pred_class[true_class == c] == c).float().mean()
    print(f"Class {c} accuracy: {class_acc:.4f}")
```

Expected during training:
- **English Acc**: ~85-90% (largest class, easier)
- **Hindi Acc**: ~82-87% (largest class, harder due to code-switching)
- **Other Acc**: ~45-65% (smallest class, needs high class weight to learn)

## Summary: Class Weights Handle It

| Aspect | Without Weights | With Inverse-Frequency Weights |
|--------|-----------------|-------------------------------|
| **Hindi dominance** | High (58.94% natural samples) | Balanced (w=0.056) |
| **Other under-learning** | Severe (3.88% samples) | Mitigated (w=0.855) |
| **Padding artifacts** | Confuse model | Controlled via weights |
| **Per-class gradient** | Imbalanced | Normalized |
| **Test performance** | Poor on rare classes | Balanced across all |

**Key Takeaway:**
> The `compute_class_weights()` function in [src/dataset.py](src/dataset.py) solves this by:
> 1. Counting only original labels (not padding)
> 2. Computing inverse-frequency weights
> 3. Normalizing to prevent magnitude issues
> 4. Returning a tensor ready for `CrossEntropyLoss(weight=class_weights)`
