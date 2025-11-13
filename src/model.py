# Model Architecture (Hybrid CNN-Transformer) [w MPS compatibility]
import torch
import torch.nn as nn

class ECGHybridModel(nn.Module):
    """
    ECGHybridModel(nn.Module)
    Hybrid CNN + Transformer model for ECG signal classification.
    Parameters
    ----------
    input_dim : int, optional
        Number of input channels/features per timestep (default: 2). The model
        expects input tensors with shape (batch_size, seq_len, input_dim).
    seq_len : int, optional
        Original input sequence length in timesteps (default: 1080). The internal
        architecture applies strided convolutions and pooling and trims the
        transformer branch to a fixed transformer_seq_len; if you change seq_len
        make sure preprocessing/downsampling results in a sequence at least as
        long as transformer_seq_len before trimming.
    num_classes : int, optional
        Number of output classes for the final classifier (default: 5).
    Summary
    -------
    This model combines a time-domain 1D-CNN branch and a Transformer encoder
    branch and fuses their representations for classification:
    - CNN branch:
      - Several Conv1d, ReLU and pooling layers producing a feature map which is
        adaptively pooled to length 16. Output per sample: flattened vector of
        size (64 * 16).
    - Transformer branch:
      - Inputs are downsampled (avg pooling) and trimmed to a fixed
        transformer_seq_len (default 256).
      - Each timestep is projected to a d_model=64 embedding, added to a learned
        positional encoding, and passed through a small TransformerEncoder.
      - Transformer outputs are permuted and adaptively pooled to length 16 and
        flattened, producing a vector of size (64 * 16).
    - Fusion & classification:
      - The CNN and Transformer flattened vectors are concatenated and projected
        through an MLP with dropout and ReLU non-linearities to produce logits
        of size (num_classes).
    Input / Output shapes
    ---------------------
    Input:
        x : torch.Tensor with shape (batch_size, seq_len, input_dim)
             Example default: (B, 1080, 2)
    CNN branch intermediate:
        - permuted to (B, input_dim, seq_len) for Conv1d processing
        - final adaptive pooled tensor: (B, 64, 16) -> flattened to (B, 64*16)
    Transformer branch intermediate:
        - downsampled then trimmed to (B, transformer_seq_len, input_dim)
        - embedded to (B, transformer_seq_len, 64)
        - transformer output: (B, transformer_seq_len, 64)
        - permuted and pooled to (B, 64, 16) -> flattened to (B, 64*16)
    Output:
        logits : torch.Tensor with shape (batch_size, num_classes)
    Implementation notes
    --------------------
    - transformer_seq_len is set inside the model (default 256) and must be <= the
      downsampled sequence length; the forward pass trims the downsampled sequence
      to this length.
    - Positional encodings are implemented as a learned parameter of shape
      (1, transformer_seq_len, 64).
    - AdaptiveAvgPool1d with output length 16 is used for both branches to ensure
      fixed-size feature vectors. The code organizes pooling/trimming to avoid
      non-divisible sizes (helpful for some backends such as MPS).
    - If you change pooling/stride/conv/kernel sizes or the input seq_len,
      recompute transformer_seq_len and verify divisibility constraints for the
      adaptive pools, or adjust pooling targets accordingly.
    - Move model and inputs to the same device (cpu/cuda/mps) and match dtypes.
    Returns
    -------
    Tensor
        Raw logits (not softmaxed) of shape (batch_size, num_classes).
    """
    def __init__(self, input_dim=2, seq_len=1080, num_classes=5):
        super().__init__()
        
        # CNN Branch
        # The CNN sequential block to guarantee a divisible output length.
        self.cnn = nn.Sequential(
            nn.Conv1d(input_dim, 32, kernel_size=16, stride=2, padding=7), # -> [B, 32, 540]
            nn.ReLU(),
            nn.MaxPool1d(4), # -> [B, 32, 135]
            nn.Conv1d(32, 64, kernel_size=8, stride=2, padding=3), # -> [B, 64, 67]
            # The small pooling layer to get a divisible dimension (67 -> 64).
            nn.AvgPool1d(kernel_size=4, stride=1) # -> [B, 64, 64]
        )
        self.cnn_pool = nn.AdaptiveAvgPool1d(16) # This is now safe: 64 % 16 == 0

        # Transformer Branch
        self.embedding = nn.Linear(input_dim, 64)
        # Define a target sequence length that is divisible by 16.
        self.transformer_seq_len = 256 
        # The positional encoding must match this new, trimmed length.
        self.pos_encoding = nn.Parameter(torch.randn(1, self.transformer_seq_len, 64))
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=64, nhead=4, dim_feedforward=256, batch_first=True, dropout=0.1
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=2)
        self.transformer_pool = nn.AdaptiveAvgPool1d(16) # This is also safe now: 256 % 16 == 0

        # Fusion & Classification
        self.fusion = nn.Linear(64*16 + 64*16, 128)
        self.classifier = nn.Sequential(
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, num_classes)
        )

    def forward(self, x):
        batch_size = x.size(0)

        # CNN branch - now produces a correctly-sized output
        cnn_out = self.cnn(x.permute(0, 2, 1))
        cnn_pooled = self.cnn_pool(cnn_out) # Safe on MPS
        cnn_flat = cnn_pooled.reshape(batch_size, -1)

        # Transformer branch - now uses a trimmed sequence
        x_down = nn.functional.avg_pool1d(x.permute(0, 2, 1), kernel_size=4).permute(0, 2, 1) # [B, 270, 2]

        # Trim the sequence from 270 to our target length of 256.
        x_down_trimmed = x_down[:, :self.transformer_seq_len, :]

        # Proceed with the trimmed, compatible tensor.
        x_emb = self.embedding(x_down_trimmed) + self.pos_encoding
        trans_out = self.transformer(x_emb) # Output is now [B, 256, 64]

        t = trans_out.permute(0, 2, 1)
        trans_pooled = self.transformer_pool(t) # Safe on MPS
        trans_flat = trans_pooled.reshape(batch_size, -1)

        # Fusion
        fused = torch.cat([cnn_flat, trans_flat], dim=1)
        features = self.fusion(fused)
        
        return self.classifier(features)


