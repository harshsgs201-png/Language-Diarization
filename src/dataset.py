"""
PyTorch Dataset and DataLoader for Language Diarization (XLSR Embeddings).
Handles variable-length utterances with custom collate_fn and SpecAugment masking.
"""

import os
import glob
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
from collections import Counter


class HiACC_XLSR_Dataset(Dataset):
    """
    PyTorch Dataset for XLSR embeddings and frame-level language labels.
    Handles variable-length utterances with SpecAugment-style masking.
    """
    
    def __init__(self, processed_dir, is_training=True):
        """
        Args:
            processed_dir: Path to folder containing *_emb.npy and *_labels.npy
            is_training: If True, applies SpecAugment-style masking for augmentation
        """
        self.is_training = is_training
        
        # Grab all embedding files and sort them to match labels
        self.emb_files = sorted(glob.glob(os.path.join(processed_dir, "*_emb.npy")))
        self.lbl_files = sorted(glob.glob(os.path.join(processed_dir, "*_labels.npy")))
        
        assert len(self.emb_files) == len(self.lbl_files), \
            f"Mismatch between embeddings ({len(self.emb_files)}) and labels ({len(self.lbl_files)})!"

    def __len__(self):
        return len(self.emb_files)

    def apply_masking(self, embedding, time_mask_max=20, feat_mask_max=100):
        """
        Custom SpecAugment for (Time, 1024) embeddings.
        Randomly zeros out blocks of time (temporal masking) and features (channel masking).
        
        Args:
            embedding: Tensor of shape (Time, 1024)
            time_mask_max: Max length of temporal mask
            feat_mask_max: Max length of feature mask
            
        Returns:
            Masked embedding tensor
        """
        emb = embedding.clone()
        time_steps, num_features = emb.shape
        
        # 1. Time Masking (simulate dropped audio frames)
        if time_steps > time_mask_max:
            t_mask = np.random.randint(1, time_mask_max)
            t0 = np.random.randint(0, time_steps - t_mask)
            emb[t0:t0 + t_mask, :] = 0.0
            
        # 2. Feature (Channel) Masking (force network to use different embedding channels)
        f_mask = np.random.randint(1, feat_mask_max)
        f0 = np.random.randint(0, num_features - f_mask)
        emb[:, f0:f0 + f_mask] = 0.0
        
        return emb

    def __getitem__(self, idx):
        """
        Returns:
            emb_tensor: XLSR embedding of shape (Time, 1024)
            lbl_tensor: Frame-level labels of shape (Time,)
        """
        # Load arrays
        emb = np.load(self.emb_files[idx])
        lbl = np.load(self.lbl_files[idx])
        
        # Convert to tensors
        emb_tensor = torch.tensor(emb, dtype=torch.float32)
        lbl_tensor = torch.tensor(lbl, dtype=torch.long)
        
        # Apply augmentation only during training
        if self.is_training:
            emb_tensor = self.apply_masking(emb_tensor)
            
        return emb_tensor, lbl_tensor


def collate_fn(batch):
    """
    Custom collate function for variable-length sequences.
    Pads all sequences to the maximum length in the batch.
    
    Args:
        batch: List of (embedding, label) tuples
        
    Returns:
        padded_embeddings: Tensor of shape (Batch, Max_Time, 1024)
        padded_labels: Tensor of shape (Batch, Max_Time)
                       Padded values are -100 (ignored by CrossEntropyLoss)
    """
    embeddings, labels = zip(*batch)
    
    # Pad embeddings with 0.0 (these are feature values, 0 is neutral)
    padded_embeddings = pad_sequence(embeddings, batch_first=True, padding_value=0.0)
    
    # Pad labels with -100 (PyTorch's CrossEntropyLoss ignores this value automatically!)
    padded_labels = pad_sequence(labels, batch_first=True, padding_value=-100)
    
    return padded_embeddings, padded_labels


def compute_class_weights(processed_dir):
    """
    Calculate class weights for CrossEntropyLoss based on actual label distribution.
    Only counts non-padded labels (original frames, not padding added during batching).
    
    Args:
        processed_dir: Path to folder containing *_labels.npy files
        
    Returns:
        torch.Tensor of shape (3,) with weights for [EN, HI, Other]
    """
    label_files = sorted(glob.glob(os.path.join(processed_dir, "*_labels.npy")))
    
    # Count each class across ALL original labels (before any padding)
    class_counts = Counter({0: 0, 1: 0, 2: 0})
    
    for lbl_file in label_files:
        lbl = np.load(lbl_file)
        # Count labels in this file (all are original, not padding)
        unique, counts = np.unique(lbl, return_counts=True)
        for u, c in zip(unique, counts):
            class_counts[int(u)] += c
    
    total_frames = sum(class_counts.values())
    
    print("\n" + "="*60)
    print("CLASS DISTRIBUTION (Original Labels Only)")
    print("="*60)
    print(f"Total frames: {total_frames:,}\n")
    
    class_names = ["English (0)", "Hindi (1)", "Other/Silence (2)"]
    for cls in range(3):
        count = class_counts[cls]
        pct = 100.0 * count / total_frames
        print(f"{class_names[cls]:20s}: {count:8,} frames ({pct:6.2f}%)")
    
    # Compute weights as inverse frequency
    # Weight = total_frames / (num_classes * class_count)
    # This balances the classes so minority classes get higher weight
    weights = []
    for cls in range(3):
        if class_counts[cls] > 0:
            weight = total_frames / (3 * class_counts[cls])
        else:
            weight = 1.0
        weights.append(weight)
    
    weights_tensor = torch.tensor(weights, dtype=torch.float32)
    
    # Normalize so they sum to 1 (helps with training stability)
    weights_tensor = weights_tensor / weights_tensor.sum()
    
    print("\n" + "-"*60)
    print("COMPUTED CLASS WEIGHTS (Normalized)")
    print("-"*60)
    for cls, w in enumerate(weights_tensor):
        print(f"{class_names[cls]:20s}: {w:.6f}")
    print("="*60 + "\n")
    
    return weights_tensor


def get_dataloader(processed_dir, batch_size=32, shuffle=True, is_training=True):
    """
    Create DataLoader for the XLSR dataset.
    
    Args:
        processed_dir: Directory containing *_emb.npy and *_labels.npy
        batch_size: Batch size for training
        shuffle: Whether to shuffle data
        is_training: If True, applies SpecAugment masking
        
    Returns:
        DataLoader object
    """
    dataset = HiACC_XLSR_Dataset(processed_dir, is_training=is_training)
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=collate_fn,
        num_workers=0  # Keep as 0 for Windows compatibility
    )
    return dataloader


if __name__ == "__main__":
    # Example usage
    print("Loading dataset...")
    dataset = HiACC_XLSR_Dataset("data/processed", is_training=True)
    print(f"Dataset size: {len(dataset)} utterances\n")
    
    # Compute class weights
    print("Computing class weights...")
    weights = compute_class_weights("data/processed")
    
    # Create dataloader
    print("Creating dataloader...")
    dataloader = get_dataloader("data/processed", batch_size=32, shuffle=True, is_training=True)
    
    # Verify it works
    print("Verifying dataloader functionality...")
    for batch_emb, batch_lbl in dataloader:
        print(f"\nBatch shapes:")
        print(f"  Embeddings: {batch_emb.shape}  (Batch, Max_Time, 1024)")
        print(f"  Labels:     {batch_lbl.shape}  (Batch, Max_Time)")
        print(f"\nLabel values in batch: {torch.unique(batch_lbl).tolist()}")
        print(f"First sequence length: {(batch_lbl[0] != 2).sum().item()} original frames")
        break
