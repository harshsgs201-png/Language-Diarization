"""
Evaluation module for Language Diarization.
Calculates metrics like LDER (Language Diarization Error Rate) and JER (Joint Error Rate).
"""

import torch
import numpy as np
from pathlib import Path
from tqdm import tqdm

try:
    from pyannote.core import Annotation, Segment
    from pyannote.metrics.diarization import DiarizationErrorRate
except ImportError:
    print("Warning: pyannote not installed. Some evaluation metrics will not be available.")


def calculate_error_rate(predicted, ground_truth, frame_length=0.02):
    """
    Calculate Language Diarization Error Rate (LDER).

    Args:
        predicted: Predicted labels (array of shape (time_steps,))
        ground_truth: Ground truth labels (array of shape (time_steps,))
        frame_length: Duration of each frame in seconds

    Returns:
        Error rate (float between 0 and 1)
    """
    # Simple frame-level error rate
    errors = np.sum(predicted != ground_truth)
    total = len(predicted)
    return errors / total if total > 0 else 0.0


def evaluate_model(model, dataloader, device):
    """
    Evaluate model on a dataset.

    Args:
        model: Trained model
        dataloader: DataLoader with test data
        device: Device to evaluate on

    Returns:
        Dictionary with evaluation metrics
    """
    model.eval()
    all_predictions = []
    all_labels = []

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating"):
            mel_specs = batch["mel_spec"].to(device)
            labels = batch["label"].to(device)

            # Add channel dimension if needed
            if mel_specs.dim() == 3:
                mel_specs = mel_specs.unsqueeze(1)

            outputs = model(mel_specs)
            predictions = torch.argmax(outputs, dim=-1)

            all_predictions.extend(predictions.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    all_predictions = np.array(all_predictions)
    all_labels = np.array(all_labels)

    # Calculate metrics
    error_rate = calculate_error_rate(all_predictions, all_labels)

    metrics = {
        "error_rate": error_rate,
        "accuracy": np.mean(all_predictions == all_labels),
    }

    return metrics, all_predictions, all_labels


def print_metrics(metrics):
    """
    Print evaluation metrics.

    Args:
        metrics: Dictionary of metrics
    """
    print("\n" + "=" * 50)
    print("Evaluation Metrics")
    print("=" * 50)
    for key, value in metrics.items():
        print(f"{key:20s}: {value:.4f}")
    print("=" * 50)


if __name__ == "__main__":
    # Example usage
    from model import create_model
    from dataset import LanguageDiarizationDataset
    from torch.utils.data import DataLoader

    # Load model
    model = create_model(num_classes=2, device="cpu")
    model.load_state_dict(torch.load("models/saved_weights/model_final.pt", map_location="cpu"))

    # Create test dataloader
    test_dataset = LanguageDiarizationDataset("data/processed")
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    # Evaluate
    metrics, predictions, labels = evaluate_model(model, test_loader, "cpu")
    print_metrics(metrics)
