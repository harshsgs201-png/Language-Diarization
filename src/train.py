"""
Training script for Language Diarization model.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import argparse
from pathlib import Path
from tqdm import tqdm
import json

from model import create_model
from dataset import LanguageDiarizationDataset


def train_epoch(model, dataloader, criterion, optimizer, device):
    """
    Train for one epoch.

    Returns:
        Average loss for the epoch
    """
    model.train()
    total_loss = 0
    num_batches = 0

    for batch in tqdm(dataloader, desc="Training"):
        mel_specs = batch["mel_spec"].to(device)
        labels = batch["label"].to(device)

        # Add channel dimension if needed
        if mel_specs.dim() == 3:
            mel_specs = mel_specs.unsqueeze(1)

        # Forward pass
        outputs = model(mel_specs)
        loss = criterion(outputs.view(-1, outputs.shape[-1]), labels.view(-1).long())

        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        num_batches += 1

    return total_loss / num_batches


def validate(model, dataloader, criterion, device):
    """
    Validate the model.

    Returns:
        Average loss on validation set
    """
    model.eval()
    total_loss = 0
    num_batches = 0

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Validating"):
            mel_specs = batch["mel_spec"].to(device)
            labels = batch["label"].to(device)

            # Add channel dimension if needed
            if mel_specs.dim() == 3:
                mel_specs = mel_specs.unsqueeze(1)

            outputs = model(mel_specs)
            loss = criterion(outputs.view(-1, outputs.shape[-1]), labels.view(-1).long())

            total_loss += loss.item()
            num_batches += 1

    return total_loss / num_batches


def train(
    model_save_dir="models/saved_weights",
    mel_spec_dir="data/processed",
    labels_dir=None,
    batch_size=32,
    num_epochs=100,
    learning_rate=0.001,
    device="cuda" if torch.cuda.is_available() else "cpu",
):
    """
    Main training function.

    Args:
        model_save_dir: Directory to save model weights
        mel_spec_dir: Directory with Mel-spectrograms
        labels_dir: Directory with labels
        batch_size: Batch size
        num_epochs: Number of training epochs
        learning_rate: Learning rate
        device: Device to train on
    """
    print(f"Using device: {device}")

    # Create directories
    Path(model_save_dir).mkdir(parents=True, exist_ok=True)

    # Create model
    model = create_model(num_classes=2, device=device)
    print(f"Model created: {model}")

    # Create datasets and dataloaders
    train_dataset = LanguageDiarizationDataset(mel_spec_dir, labels_dir)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=5)

    # Training loop
    history = {"train_loss": []}

    for epoch in range(num_epochs):
        train_loss = train_epoch(model, train_loader, criterion, optimizer, device)
        history["train_loss"].append(train_loss)

        print(f"Epoch {epoch + 1}/{num_epochs} - Loss: {train_loss:.4f}")

        # Save checkpoint
        if (epoch + 1) % 10 == 0:
            checkpoint_path = Path(model_save_dir) / f"checkpoint_epoch_{epoch + 1}.pt"
            torch.save(model.state_dict(), str(checkpoint_path))
            print(f"Checkpoint saved to {checkpoint_path}")

        scheduler.step(train_loss)

    # Save final model
    final_model_path = Path(model_save_dir) / "model_final.pt"
    torch.save(model.state_dict(), str(final_model_path))
    print(f"Final model saved to {final_model_path}")

    # Save training history
    history_path = Path(model_save_dir) / "training_history.json"
    with open(str(history_path), "w") as f:
        json.dump(history, f)

    return model, history


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Language Diarization model")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size")
    parser.add_argument("--num_epochs", type=int, default=100, help="Number of epochs")
    parser.add_argument("--learning_rate", type=float, default=0.001, help="Learning rate")
    parser.add_argument("--mel_spec_dir", type=str, default="data/processed", help="Directory with mel-spectrograms")
    parser.add_argument("--model_save_dir", type=str, default="models/saved_weights", help="Directory to save models")

    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    train(
        model_save_dir=args.model_save_dir,
        mel_spec_dir=args.mel_spec_dir,
        batch_size=args.batch_size,
        num_epochs=args.num_epochs,
        learning_rate=args.learning_rate,
        device=device,
    )
