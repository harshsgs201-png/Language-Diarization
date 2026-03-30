# DISK SPACE SOLUTIONS & RECOMMENDATIONS

## Current Situation
- **D: drive**: ~0 MB free (needs 2GB+ for model download)
- **Embeddings output**: ~5GB total (5,167 files × 1MB each)
- **Total needed**: 7+ GB
- **Status**: Extraction blocked

---

## Option 1: Free Up Disk Space (Recommended)

### Quick Cleanup (Windows)
```powershell
# Delete temp files
Remove-Item -Path C:\Windows\Temp\* -Recurse -Force -ErrorAction SilentlyContinue
Remove-Item -Path C:\Users\harsh\AppData\Local\Temp\* -Recurse -Force -ErrorAction SilentlyContinue

# Clear HuggingFace cache (will re-download first time)
Remove-Item -Path "C:\Users\harsh\.cache\huggingface*" -Recurse -Force -ErrorAction SilentlyContinue

# Check disk space after cleanup
dir D:\ | Measure-Object -Property Length -Sum | Select-Object @{Label="Free Space";Expression={[Math]::Round(($_.Sum)/1GB, 2)}}MB
```

### Large Files to Check
Find and delete:
- Old Python packages (D:\venv-harsh) - backup if needed
- Large video/media files in data/
- Docker images or virtual machine files
- Old project backups

---

## Option 2: Use Smaller Subset (Quick Testing)

If you can't free up 7GB, extract embeddings for just a test set first:

```bash
# Modify src/extract_xlsr.py to process only N files:
python -c "
import glob
import random
labels = sorted(glob.glob('data/processed/*_labels.npy'))
# Keep only 500 random samples for testing
test_set = random.sample(labels, min(500, len(labels)))
for f in test_set:
    print(f)
" > test_subset.txt
```

Then train on this test set to validate the pipeline works before scaling to full 5,167.

---

## Option 3: Use External Storage

If you have USB or external hard drive (7+ GB):
1. Connect external drive (e.g., F:)
2. Modify PROCESSED_DIR in extract_xlsr.py: `PROCESSED_DIR = "F:/embeddings"`
3. Run extraction to external drive
4. Move embeddings back when space frees up

---

## Option 4: Stream Embeddings During Training (No Storage)

Modify training to extract embeddings **on-the-fly** during training:
- Don't save embeddings to disk
- Extract in batch during training
- Trade compute time for disk space
- Requires rewriting dataset.py

**Not recommended** (too slow), but possible if disk space is permanent constraint.

---

## Recommended Path Forward

### Step 1: Check Current Free Space
```powershell
# PowerShell
Get-Volume -DriveLetter D | Format-Table DriveLetter, SizeRemaining, Size
Get-Volume -DriveLetter C | Format-Table DriveLetter, SizeRemaining, Size
```

### Step 2: Clean Up Non-Essential Files
- Windows Temp folder
- HuggingFace cache (will re-download, 1.27GB)
- Other large files not needed

### Step 3: Verify 7+ GB Free
```powershell
$free = (Get-Volume -DriveLetter D).SizeRemaining
Write-Host "Free space: $([Math]::Round($free/1GB, 2))GB"
# Should show >= 7
```

### Step 4: Run Extraction Again
```bash
python src/extract_xlsr.py
# Expected time: 30-60 minutes on CPU
# Will create: 5,167 *_emb.npy files in data/processed/
```

### Step 5: Verify & Train
```bash
python src/verify_data.py
# Expected time: 1 minute
# Should show: nan_count=0, zero_count=0, class_weights=[0.089, 0.056, 0.855]

python src/train.py
# Expected time: 10-12 hours on CPU, 30 min on GPU
# Will save: models/weights/xlsr_diarizer_best.pt
```

---

## What Each Step Creates

| Step | Output | Size | Keep? |
|------|--------|------|-------|
| Extract XLSR | 5,167 `*_emb.npy` files | ~5 GB | ✅ Yes (needed for training) |
| Train Model | `xlsr_diarizer_best.pt` | ~16 MB | ✅ Yes (YOUR TRAINED MODEL!) |
| HF Cache | Model weights in `~/.cache` | ~1.3 GB | ⚠️ One-time only |

**Total disk needed**: 7 GB (5 GB embeddings + 1.3 GB model + 0.7 GB buffer)

---

## Emergency: Minimal Run

If you absolutely cannot free 7GB:
1. Process **only 500 random utterances** (1GB total)
2. Train and validate pipeline on small set
3. Once training works, delete partial embeddings
4. Free up space (delete trained model, clean cache)
5. Re-run full extraction
6. Re-train on complete data

This lets you validate everything works before committing full disk space.

---

## Questions Before Running

1. **How much free space do you have right now?**
   - Check: Right-click D:\ → Properties

2. **Can you delete anything on D:?**
   - Old venvs? Old projects? Old data backups?

3. **Have you got external USB/external HD?**
   - Can process there instead

4. **Time constraint?**
   - CPU extraction: 30-60 min
   - CPU training: 10-12 hours
   - GPU training: 30-60 min

---

## After SUCCESSFUL Extraction

Once embeddings are saved, disk space becomes less critical:
- Delete HF cache (~1.3 GB freed automatically)
- Embeddings stay (~5 GB)
- Training only needs ~500 MB additional buffer

Then training becomes fast!

---

## Status

✅ **Phase 1**: Complete (5,167 labeled files)  
⏸️ **Phase 2**: Blocked on disk space (39 partial embeddings, 5,128 pending)  
✅ **Phase 3**: Code ready (model, training loop, verification)

**Critical Path**: Free disk space → Run extraction → Run training

