"""
PyTorch Dataset and DataLoader for Language Diarization.
"""

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from pathlib import Path


class LanguageDiarizationDataset(Dataset):
    """
    PyTorch Dataset for language diarization task.
    Loads Mel-spectrograms and corresponding labels.
    """

    def __init__(self, mel_spec_dir, labels_dir=None, transform=None):
        """
        Args:
            mel_spec_dir: Directory containing .npy Mel-spectrogram files
            labels_dir: Directory containing label files
            transform: Optional transforms to apply
        """
        self.mel_spec_dir = Path(mel_spec_dir)
        self.labels_dir = Path(labels_dir) if labels_dir else None
        self.transform = transform

        # Get list of spectrograms
        self.mel_spec_files = sorted(self.mel_spec_dir.glob("**/*_melspec.npy"))

    def __len__(self):
        return len(self.mel_spec_files)

    def __getitem__(self, idx):
        """
        Returns:
            mel_spec: Mel-spectrogram (C, T)
            label: Label tensor (optional)
        """
        mel_spec_path = self.mel_spec_files[idx]
        mel_spec = np.load(mel_spec_path)
        mel_spec = torch.FloatTensor(mel_spec)

        if self.transform:
            mel_spec = self.transform(mel_spec)

        # Load labels if available
        if self.labels_dir:
            label_path = self.labels_dir / f"{mel_spec_path.stem}.npy"
            if label_path.exists():
                label = np.load(label_path)
                label = torch.FloatTensor(label)
            else:
                label = torch.zeros(mel_spec.shape[-1])
        else:
            label = torch.zeros(mel_spec.shape[-1])

        return {
            "mel_spec": mel_spec,
            "label": label,
            "filename": mel_spec_path.stem,
        }


def get_dataloader(mel_spec_dir, labels_dir=None, batch_size=32, shuffle=True, num_workers=0):
    """
    Create DataLoader for the dataset.

    Args:
        mel_spec_dir: Directory with Mel-spectrograms
        labels_dir: Directory with labels
        batch_size: Batch size
        shuffle: Whether to shuffle data
        num_workers: Number of workers for data loading

    Returns:
        DataLoader object
    """
    dataset = LanguageDiarizationDataset(mel_spec_dir, labels_dir)
    dataloader = DataLoader(
        dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers
    )
    return dataloader


if __name__ == "__main__":
    # Example usage
    train_loader = get_dataloader("data/processed", batch_size=16, shuffle=True)
    for batch in train_loader:
        print(f"Batch shape: {batch['mel_spec'].shape}")
        print(f"Label shape: {batch['label'].shape}")
        break
