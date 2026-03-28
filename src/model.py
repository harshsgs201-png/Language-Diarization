"""
CRNN (Convolutional Recurrent Neural Network) model for Language Diarization.
"""

import torch
import torch.nn as nn


class CRNN(nn.Module):
    """
    CRNN model: Convolutional layers followed by RNN layers.
    Designed for temporal sequence modeling of Mel-spectrograms.
    """

    def __init__(self, input_channels=1, num_classes=2, hidden_size=128, num_layers=2):
        """
        Args:
            input_channels: Number of input channels (usually 1 for Mel-specs)
            num_classes: Number of output classes (languages)
            hidden_size: Hidden size for LSTM
            num_layers: Number of LSTM layers
        """
        super(CRNN, self).__init__()

        # Convolutional layers
        self.conv1 = nn.Conv2d(input_channels, 32, kernel_size=(3, 3), padding=(1, 1))
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=(3, 3), padding=(1, 1))
        self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=(3, 3), padding=(1, 1))
        self.bn3 = nn.BatchNorm2d(128)

        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool2d((2, 2))

        # Recurrent layers
        self.lstm = nn.LSTM(
            input_size=128 * 8,  # After conv and pooling
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
        )

        # Output layer
        self.fc = nn.Linear(hidden_size * 2, num_classes)

    def forward(self, x):
        """
        Args:
            x: Input tensor of shape (batch_size, channels, height, width)
               height = frequency bins, width = time steps

        Returns:
            output: Predictions of shape (batch_size, time_steps, num_classes)
        """
        # Convolutional blocks
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.maxpool(x)

        x = self.relu(self.bn2(self.conv2(x)))
        x = self.maxpool(x)

        x = self.relu(self.bn3(self.conv3(x)))
        x = self.maxpool(x)

        # Reshape for LSTM: (batch, channels, height, width) -> (batch, time, features)
        batch_size, channels, height, width = x.shape
        x = x.permute(0, 3, 1, 2)  # (batch, width, channels, height)
        x = x.reshape(batch_size, width, -1)  # (batch, time, features)

        # LSTM
        lstm_out, _ = self.lstm(x)

        # Output layer
        output = self.fc(lstm_out)

        return output


def create_model(num_classes=2, device="cpu"):
    """
    Create and return a CRNN model.

    Args:
        num_classes: Number of output classes
        device: Device to move model to

    Returns:
        Model on specified device
    """
    model = CRNN(input_channels=1, num_classes=num_classes)
    model = model.to(device)
    return model


if __name__ == "__main__":
    # Test model
    model = create_model(num_classes=2, device="cpu")
    x = torch.randn(4, 1, 64, 100)  # (batch, channels, freq_bins, time_steps)
    output = model(x)
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {output.shape}")
