"""
Phase 3: Language Diarization Model Architecture
Bi-LSTM with attention layer for frame-level language prediction from XLSR embeddings.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class XLSRDiarizer(nn.Module):
    """
    XLSR-based Frame-Level Language Classifier with Self-Attention.
    
    Input: XLSR embeddings (Batch, Time, 1024)
    Output: Class logits (Batch, Time, 3) for [English, Hindi, Other]
    """
    
    def __init__(self, input_dim=1024, hidden_dim=256, num_classes=3):
        """
        Args:
            input_dim: XLSR embedding dimension (always 1024)
            hidden_dim: LSTM hidden state dimension
            num_classes: Number of output classes (3: EN, HI, Other)
        """
        super(XLSRDiarizer, self).__init__()
        
        # Bi-directional LSTM for capturing temporal context
        # Bi-LSTM output will be: hidden_dim * 2 (forward + backward)
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=2,
            batch_first=True,
            bidirectional=True,
            dropout=0.3
        )
        
        # Self-attention over LSTM outputs to weight frame importance
        # Outputs a single attention score per frame
        self.attention = nn.Linear(hidden_dim * 2, 1)
        
        # Frame-level classifier
        # Takes weighted features and outputs 3 class logits per frame
        self.classifier = nn.Linear(hidden_dim * 2, num_classes)

    def forward(self, x):
        """
        Args:
            x: Input embeddings (Batch, Time, 1024)
        
        Returns:
            logits: Class logits (Batch, Time, 3)
        """
        # BiLSTM encoding
        # x shape: (Batch, Time, 1024)
        lstm_out, _ = self.lstm(x)
        # lstm_out shape: (Batch, Time, hidden_dim * 2 = 512)
        
        # Self-attention weighting
        # For each frame, compute attention score [0, 1]
        attn_logits = self.attention(lstm_out)  # (Batch, Time, 1)
        attn_weights = F.softmax(attn_logits, dim=1)  # Normalize across time
        # attn_weights shape: (Batch, Time, 1)
        
        # Apply attention: element-wise multiplication
        # This allows the model to focus on important frames/transitions
        context = lstm_out * attn_weights  # Broadcast multiply
        # context shape: (Batch, Time, 512)
        
        # Frame-level classification
        logits = self.classifier(context)
        # logits shape: (Batch, Time, 3)
        
        return logits


def create_model(device='cpu'):
    """
    Factory function to create XLSRDiarizer model.
    
    Args:
        device: 'cpu' or 'cuda'
    
    Returns:
        Model instance on specified device
    """
    model = XLSRDiarizer(
        input_dim=1024,
        hidden_dim=256,
        num_classes=3
    )
    return model.to(device)


if __name__ == "__main__":
    # Test forward pass
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    model = create_model(device=device)
    print(f"Model created on {device}\n")
    print(model)
    
    # Test with a batch
    batch_size = 4
    seq_len = 500
    input_dim = 1024
    
    x = torch.randn(batch_size, seq_len, input_dim).to(device)
    logits = model(x)
    
    print(f"\nTest Forward Pass:")
    print(f"  Input shape:  {x.shape}")
    print(f"  Output shape: {logits.shape}")
    print(f"  Expected:     ({batch_size}, {seq_len}, 3)")
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\nModel Parameters:")
    print(f"  Total:     {total_params:,}")
    print(f"  Trainable: {trainable_params:,}")



