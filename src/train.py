"""
Phase 3: Training Script for Language Diarization Model
Trains XLSRDiarizer on frame-level language classification with weighted CrossEntropyLoss.
"""

import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from dataset import HiACC_XLSR_Dataset, collate_fn, compute_class_weights
from model import create_model
from verify_data import verify_extraction
from tqdm import tqdm
import json


def train():
    """Main training loop for language diarization."""
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n{'='*60}")
    print(f"Training on {device}...")
    print(f"{'='*60}\n")

    # --- Step 1: Verify Phase 2 Data ---
    print("Step 1: Verifying Phase 2 embeddings...")
    verification = verify_extraction("data/processed")
    
    if verification['nan_count'] > 0 or verification['zero_count'] > 0:
        print("⚠️  WARNING: Corrupted embeddings detected! Training may fail.")
        print("   Please run Phase 2 extraction script first.")
        return
    
    print("\n✅ Phase 2 verification passed!\n")

    # --- Step 2: Load Dataset ---
    print("Step 2: Loading dataset...")
    full_dataset = HiACC_XLSR_Dataset("data/processed", is_training=True)
    print(f"   Total samples: {len(full_dataset)}")
    
    # 90/10 Train/Val Split
    train_size = int(0.9 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_set, val_set = random_split(full_dataset, [train_size, val_size])
    
    # Validation shouldn't use SpecAugment
    val_set.dataset.is_training = False
    
    print(f"   Training samples: {train_size}")
    print(f"   Validation samples: {val_size}\n")

    train_loader = DataLoader(
        train_set,
        batch_size=16,
        shuffle=True,
        collate_fn=collate_fn
    )
    val_loader = DataLoader(
        val_set,
        batch_size=16,
        shuffle=False,
        collate_fn=collate_fn
    )

    # --- Step 3: Initialize Model ---
    print("Step 3: Initializing model...")
    model = create_model(device=device)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"   Model parameters: {total_params:,}\n")

    # --- Step 4: Setup Loss & Optimizer ---
    print("Step 4: Setting up loss function and optimizer...")
    
    # Get class weights from verification
    class_weights = torch.tensor(verification['class_weights'], dtype=torch.float32).to(device)
    print(f"   Class weights: {[f'{w:.4f}' for w in class_weights.tolist()]}")
    
    # CrossEntropyLoss with:
    # - weight: Class balancing via inverse frequency
    # - ignore_index=-100: PyTorch ignores padding automatically
    criterion = nn.CrossEntropyLoss(weight=class_weights, ignore_index=-100)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=1e-3,
        weight_decay=1e-4
    )
    
    # Learning rate scheduler (reduce LR on plateau)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='min',
        factor=0.5,
        patience=3
    )
    
    print(f"   Optimizer: AdamW (lr=1e-3)")
    print(f"   Loss: CrossEntropyLoss with ignore_index=-100\n")

    # --- Step 5: Training Loop ---
    print(f"{'='*60}")
    print("TRAINING STARTED")
    print(f"{'='*60}\n")
    
    epochs = 20
    best_val_loss = float('inf')
    patience_counter = 0
    max_patience = 5
    
    os.makedirs("models/weights", exist_ok=True)

    for epoch in range(epochs):
        # Training phase
        model.train()
        train_loss = 0.0
        train_batches = 0
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs} [Train]")
        for emb, lbl in pbar:
            emb, lbl = emb.to(device), lbl.to(device)
            
            # Forward pass
            optimizer.zero_grad()
            logits = model(emb)  # (Batch, Time, 3)
            
            # Loss calculation (ignore_index=-100 handles padding)
            loss = criterion(logits.view(-1, 3), lbl.view(-1))
            
            # Backward pass
            loss.backward()
            
            # Gradient clipping (prevents exploding gradients in LSTM)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            
            optimizer.step()
            
            train_loss += loss.item()
            train_batches += 1
            pbar.set_postfix({'loss': f'{loss.item():.4f}'})
        
        avg_train_loss = train_loss / train_batches
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        val_batches = 0
        
        with torch.no_grad():
            pbar = tqdm(val_loader, desc=f"Epoch {epoch+1}/{epochs} [Val]")
            for emb, lbl in pbar:
                emb, lbl = emb.to(device), lbl.to(device)
                
                logits = model(emb)
                loss = criterion(logits.view(-1, 3), lbl.view(-1))
                
                val_loss += loss.item()
                val_batches += 1
                pbar.set_postfix({'loss': f'{loss.item():.4f}'})
        
        avg_val_loss = val_loss / val_batches
        
        # Logging
        print(f"\n   Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}")
        
        # Learning rate scheduling
        scheduler.step(avg_val_loss)
        
        # Save best model
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            patience_counter = 0
            model_path = "models/weights/xlsr_diarizer_best.pt"
            torch.save(model.state_dict(), model_path)
            print(f"   ✅ Saved best model! (loss: {best_val_loss:.4f})")
        else:
            patience_counter += 1
            if patience_counter >= max_patience:
                print(f"\n   ⏹️  Early stopping triggered (patience={max_patience})")
                break
        
        print()

    print(f"{'='*60}")
    print("TRAINING COMPLETED")
    print(f"{'='*60}")
    print(f"\n✅ Best model saved to: models/weights/xlsr_diarizer_best.pt")
    print(f"   Best validation loss: {best_val_loss:.4f}\n")


if __name__ == "__main__":
    train()
