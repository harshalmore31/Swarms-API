"""
FFTNet: An Efficient Alternative to Self-Attention

This implementation is based on the paper "The FFT Strikes Back: An Efficient Alternative to
Self-Attention" by Jacob Fein-Ashley.

The model uses adaptive spectral filtering to achieve global token mixing in O(n log n) time
instead of the O(n²) complexity of traditional self-attention mechanisms.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.fft import fft, ifft
from typing import Optional
from loguru import logger


class ShapeLogger:
    """
    A utility class to log tensor shapes during forward passes.
    Helps with debugging and understanding data flow.
    """

    @staticmethod
    def log_shape(
        tensor: torch.Tensor, name: str, level: str = "DEBUG"
    ) -> torch.Tensor:
        """Log the shape of a tensor and return the tensor unchanged."""
        if level.upper() == "DEBUG":
            logger.debug(f"{name} shape: {tensor.shape}")
        elif level.upper() == "INFO":
            logger.info(f"{name} shape: {tensor.shape}")
        return tensor


class ModReLU(nn.Module):
    """
    ModReLU activation for complex numbers as described in the Unitary RNN paper.

    This applies a ReLU-like threshold to the magnitude while preserving the phase.
    """

    def __init__(self, dim: int, num_heads: int = 1):
        """
        Initialize ModReLU with learnable bias parameters.

        Args:
            dim: The dimensionality of the bias parameter
            num_heads: Number of attention heads
        """
        super().__init__()
        self.num_heads = num_heads
        if num_heads > 1:
            # For multi-head case, we need a bias per head
            self.bias = nn.Parameter(torch.zeros(num_heads, 1, dim // num_heads))
        else:
            self.bias = nn.Parameter(torch.zeros(dim))
        self.shape_logger = ShapeLogger()
        logger.info(f"Initialized ModReLU with dim={dim}, num_heads={num_heads}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply ModReLU activation to complex tensor.

        Args:
            x: Complex tensor of shape [batch_size, num_heads, seq_len, head_dim] or [batch_size, seq_len, dim]

        Returns:
            Complex tensor after ModReLU activation
        """
        x = self.shape_logger.log_shape(x, "ModReLU input")
        # Get magnitude and phase
        magnitude = torch.abs(x)
        phase = torch.angle(x)

        # Apply bias and threshold to magnitude based on tensor shape
        if x.dim() == 4:  # [batch_size, num_heads, seq_len, head_dim]
            # The bias is already shaped for broadcasting
            thresholded_magnitude = F.relu(magnitude + self.bias)
        else:  # [batch_size, seq_len, dim]
            thresholded_magnitude = F.relu(
                magnitude + self.bias.unsqueeze(0).unsqueeze(0)
            )

        # Convert back to complex using new magnitude and original phase
        real_part = thresholded_magnitude * torch.cos(phase)
        imag_part = thresholded_magnitude * torch.sin(phase)

        # Reconstruct complex tensor
        result = torch.complex(real_part, imag_part)
        return self.shape_logger.log_shape(result, "ModReLU output")


class AdaptiveSpectralFilter(nn.Module):
    """
    Adaptive spectral filter that modulates Fourier coefficients based on a global context vector.
    """

    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        mlp_ratio: int = 4,
        init_values: float = 0.1,
        dropout: float = 0.0,
    ):
        """
        Initialize the adaptive spectral filter.

        Args:
            dim: Input dimension
            num_heads: Number of attention heads
            mlp_ratio: Ratio for hidden dimension in MLP
            init_values: Initial scaling for learnable modulation
            dropout: Dropout rate
        """
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim**-0.5
        self.shape_logger = ShapeLogger()

        # Base filter (fixed part)
        self.base_filter = nn.Parameter(torch.ones(1, num_heads, 1, self.head_dim))
        self.base_bias = nn.Parameter(torch.zeros(1, num_heads, 1, self.head_dim))

        # Context MLP to generate adaptive modulation parameters
        self.context_mlp = nn.Sequential(
            nn.Linear(dim, dim * mlp_ratio),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(
                dim * mlp_ratio, 2 * num_heads * self.head_dim
            ),  # For both scale and bias
            nn.Dropout(dropout),
        )

        # Initialize with small values to ensure stability during early training
        nn.init.normal_(self.base_filter, mean=1.0, std=init_values)
        nn.init.zeros_(self.base_bias)

        # Complex activation
        self.modrelu = ModReLU(dim, num_heads=num_heads)

        logger.info(
            f"Initialized AdaptiveSpectralFilter with dim={dim}, num_heads={num_heads}"
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply adaptive spectral filtering.

        Args:
            x: Input tensor of shape [batch_size, seq_len, dim]

        Returns:
            Filtered tensor of the same shape
        """
        x = self.shape_logger.log_shape(x, "AdaptiveSpectralFilter input")
        batch_size, seq_len, dim = x.shape

        # Compute global context vector
        context = torch.mean(x, dim=1)  # [batch_size, dim]
        context = self.shape_logger.log_shape(context, "Global context")

        # Generate adaptive modulation parameters
        modulation = self.context_mlp(context)  # [batch_size, 2 * num_heads * head_dim]
        modulation = self.shape_logger.log_shape(modulation, "Modulation parameters")

        # Split into scale and bias
        modulation = modulation.view(batch_size, 2, self.num_heads, self.head_dim)
        delta_scale = modulation[:, 0].unsqueeze(
            2
        )  # [batch_size, num_heads, 1, head_dim]
        delta_bias = modulation[:, 1].unsqueeze(
            2
        )  # [batch_size, num_heads, 1, head_dim]

        # Reshape x for multi-head processing
        x_reshaped = x.view(batch_size, seq_len, self.num_heads, self.head_dim)
        x_reshaped = x_reshaped.permute(
            0, 2, 1, 3
        )  # [batch_size, num_heads, seq_len, head_dim]

        # Apply FFT along sequence dimension
        x_freq = fft(x_reshaped, dim=2)  # [batch_size, num_heads, seq_len, head_dim]
        x_freq = self.shape_logger.log_shape(x_freq, "X in frequency domain")

        # Compute effective filter and bias
        effective_filter = self.base_filter * (1 + delta_scale)
        effective_bias = self.base_bias + delta_bias

        # Apply adaptive filtering in frequency domain
        filtered_x_freq = x_freq * effective_filter + effective_bias
        filtered_x_freq = self.shape_logger.log_shape(
            filtered_x_freq, "Filtered X in frequency domain"
        )

        # Apply ModReLU activation
        filtered_x_freq = self.modrelu(filtered_x_freq)

        # Apply inverse FFT
        x_filtered = ifft(filtered_x_freq, dim=2)

        # Take real part and reshape back
        x_filtered = x_filtered.real
        x_filtered = x_filtered.permute(
            0, 2, 1, 3
        ).contiguous()  # [batch_size, seq_len, num_heads, head_dim]
        x_filtered = x_filtered.view(
            batch_size, seq_len, dim
        )  # [batch_size, seq_len, dim]

        return self.shape_logger.log_shape(x_filtered, "AdaptiveSpectralFilter output")


class FFTNetLayer(nn.Module):
    """
    FFTNet layer that replaces self-attention with adaptive spectral filtering.

    Each layer consists of:
    1. Layer normalization
    2. Adaptive spectral filtering
    3. Residual connection
    4. Layer normalization
    5. MLP
    6. Residual connection
    """

    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        mlp_ratio: int = 4,
        dropout: float = 0.0,
        layer_scale_init_value: float = 1e-5,
    ):
        """
        Initialize the FFTNet layer.

        Args:
            dim: Input dimension
            num_heads: Number of attention heads
            mlp_ratio: Ratio for hidden dimension in MLP
            dropout: Dropout rate
            layer_scale_init_value: Initial value for layer scaling
        """
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.shape_logger = ShapeLogger()

        # Layer 1: Normalization + Adaptive Spectral Filtering + Residual
        self.norm1 = nn.LayerNorm(dim)
        self.spectral_filter = AdaptiveSpectralFilter(
            dim=dim, num_heads=num_heads, mlp_ratio=mlp_ratio, dropout=dropout
        )

        # Layer 2: Normalization + MLP + Residual
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim * mlp_ratio),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim * mlp_ratio, dim),
            nn.Dropout(dropout),
        )

        # Layer scaling parameters
        self.layer_scale_1 = nn.Parameter(layer_scale_init_value * torch.ones(dim))
        self.layer_scale_2 = nn.Parameter(layer_scale_init_value * torch.ones(dim))

        logger.info(f"Initialized FFTNetLayer with dim={dim}, num_heads={num_heads}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the FFTNet layer.

        Args:
            x: Input tensor of shape [batch_size, seq_len, dim]

        Returns:
            Output tensor of the same shape
        """
        x = self.shape_logger.log_shape(x, "FFTNetLayer input")

        # Layer 1: Normalization + Adaptive Spectral Filtering + Residual
        x = x + self.layer_scale_1.unsqueeze(0).unsqueeze(0) * self.spectral_filter(
            self.norm1(x)
        )

        # Layer 2: Normalization + MLP + Residual
        x = x + self.layer_scale_2.unsqueeze(0).unsqueeze(0) * self.mlp(self.norm2(x))

        return self.shape_logger.log_shape(x, "FFTNetLayer output")


class SpectralGating(nn.Module):
    """
    Spectral gating mechanism to selectively emphasize frequency components.
    """

    def __init__(self, dim: int, dropout: float = 0.0):
        """
        Initialize the spectral gating mechanism.

        Args:
            dim: Input dimension
            dropout: Dropout rate
        """
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.gate = nn.Sequential(
            nn.Linear(dim, dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim, dim),
            nn.Sigmoid(),
        )
        self.shape_logger = ShapeLogger()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply spectral gating to input tensor.

        Args:
            x: Input tensor of shape [batch_size, seq_len, dim]

        Returns:
            Gated tensor of the same shape
        """
        x = self.shape_logger.log_shape(x, "SpectralGating input")
        norm_x = self.norm(x)

        # Compute global context
        context = torch.mean(norm_x, dim=1, keepdim=True)
        context = self.shape_logger.log_shape(context, "SpectralGating context")

        # Generate gate values
        gate = self.gate(context)
        gate = self.shape_logger.log_shape(gate, "SpectralGating gate values")

        # Apply gate
        gated_x = x * gate

        return self.shape_logger.log_shape(gated_x, "SpectralGating output")


class FFTNet(nn.Module):
    """
    FFTNet model as described in the paper "The FFT Strikes Back: An Efficient Alternative to Self-Attention".

    This model replaces self-attention with adaptive spectral filtering using the Fast Fourier Transform (FFT).
    """

    def __init__(
        self,
        dim: int = 768,
        depth: int = 12,
        num_heads: int = 12,
        mlp_ratio: int = 4,
        dropout: float = 0.0,
        layer_scale_init_value: float = 1e-5,
        use_spectral_gating: bool = True,
    ):
        """
        Initialize the FFTNet model.

        Args:
            dim: Model dimension
            depth: Number of layers
            num_heads: Number of attention heads
            mlp_ratio: Ratio for hidden dimension in MLP
            dropout: Dropout rate
            layer_scale_init_value: Initial value for layer scaling
            use_spectral_gating: Whether to use spectral gating mechanism
        """
        super().__init__()
        self.dim = dim
        self.depth = depth
        self.num_heads = num_heads
        self.use_spectral_gating = use_spectral_gating
        self.shape_logger = ShapeLogger()

        # Create layers
        self.layers = nn.ModuleList(
            [
                FFTNetLayer(
                    dim=dim,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    dropout=dropout,
                    layer_scale_init_value=layer_scale_init_value,
                )
                for _ in range(depth)
            ]
        )

        # Optional spectral gating
        if use_spectral_gating:
            self.spectral_gating = nn.ModuleList(
                [SpectralGating(dim=dim, dropout=dropout) for _ in range(depth)]
            )

        self.norm = nn.LayerNorm(dim)

        logger.info(
            f"Initialized FFTNet with dim={dim}, depth={depth}, num_heads={num_heads}"
        )
        logger.info(f"Using spectral gating: {use_spectral_gating}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the FFTNet model.

        Args:
            x: Input tensor of shape [batch_size, seq_len, dim]

        Returns:
            Output tensor of the same shape
        """
        x = self.shape_logger.log_shape(x, "FFTNet input")

        for i, layer in enumerate(self.layers):
            # Apply layer
            x = layer(x)

            # Apply spectral gating if enabled
            if self.use_spectral_gating:
                x = self.spectral_gating[i](x)

        # Final normalization
        x = self.norm(x)

        return self.shape_logger.log_shape(x, "FFTNet output")


class FFTNetForSequenceClassification(nn.Module):
    """
    FFTNet model for sequence classification tasks.

    This model adds a classification head on top of the FFTNet encoder.
    """

    def __init__(
        self,
        num_classes: int,
        dim: int = 768,
        depth: int = 12,
        num_heads: int = 12,
        mlp_ratio: int = 4,
        dropout: float = 0.0,
        layer_scale_init_value: float = 1e-5,
        use_spectral_gating: bool = True,
        pool: str = "cls",  # 'cls' or 'mean'
    ):
        """
        Initialize the model for sequence classification.

        Args:
            num_classes: Number of classification classes
            dim: Model dimension
            depth: Number of layers
            num_heads: Number of attention heads
            mlp_ratio: Ratio for hidden dimension in MLP
            dropout: Dropout rate
            layer_scale_init_value: Initial value for layer scaling
            use_spectral_gating: Whether to use spectral gating mechanism
            pool: Pooling method ('cls' or 'mean')
        """
        super().__init__()
        self.pool = pool
        self.shape_logger = ShapeLogger()

        # FFTNet encoder
        self.encoder = FFTNet(
            dim=dim,
            depth=depth,
            num_heads=num_heads,
            mlp_ratio=mlp_ratio,
            dropout=dropout,
            layer_scale_init_value=layer_scale_init_value,
            use_spectral_gating=use_spectral_gating,
        )

        # Classification head
        self.classifier = nn.Linear(dim, num_classes)

        logger.info(
            f"Initialized FFTNetForSequenceClassification with num_classes={num_classes}, pool={pool}"
        )

    def forward(
        self, x: torch.Tensor, cls_token: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Forward pass for sequence classification.

        Args:
            x: Input tensor of shape [batch_size, seq_len, dim]
            cls_token: Optional classification token of shape [batch_size, 1, dim]

        Returns:
            Classification logits of shape [batch_size, num_classes]
        """
        x = self.shape_logger.log_shape(x, "Classification model input")

        # Prepend cls token if provided and using 'cls' pooling
        if self.pool == "cls" and cls_token is not None:
            x = torch.cat([cls_token, x], dim=1)

        # Apply FFTNet encoder
        x = self.encoder(x)

        # Pooling
        if self.pool == "cls" and cls_token is not None:
            x = x[:, 0]  # Use the first token (CLS)
        else:  # "mean" pooling
            x = torch.mean(x, dim=1)

        x = self.shape_logger.log_shape(x, "Pooled representation")

        # Classification
        logits = self.classifier(x)

        return self.shape_logger.log_shape(logits, "Classification logits")


class FFTNetViT(nn.Module):
    """
    Vision Transformer model using FFTNet instead of self-attention.

    This model follows the ViT architecture but uses FFTNet for token mixing.
    """

    def __init__(
        self,
        img_size: int = 224,
        patch_size: int = 16,
        in_channels: int = 3,
        num_classes: int = 1000,
        dim: int = 768,
        depth: int = 12,
        num_heads: int = 12,
        mlp_ratio: int = 4,
        dropout: float = 0.0,
        layer_scale_init_value: float = 1e-5,
        use_spectral_gating: bool = True,
    ):
        """
        Initialize the FFTNetViT model.

        Args:
            img_size: Input image size
            patch_size: Patch size for tokenization
            in_channels: Number of input channels
            num_classes: Number of classification classes
            dim: Model dimension
            depth: Number of layers
            num_heads: Number of attention heads
            mlp_ratio: Ratio for hidden dimension in MLP
            dropout: Dropout rate
            layer_scale_init_value: Initial value for layer scaling
            use_spectral_gating: Whether to use spectral gating mechanism
        """
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.dim = dim
        self.shape_logger = ShapeLogger()

        # Calculate number of patches
        num_patches = (img_size // patch_size) ** 2

        # Patch embedding
        self.patch_embed = nn.Conv2d(
            in_channels, dim, kernel_size=patch_size, stride=patch_size
        )

        # Position embedding and CLS token
        self.cls_token = nn.Parameter(torch.zeros(1, 1, dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, dim))

        # FFTNet encoder
        self.encoder = FFTNet(
            dim=dim,
            depth=depth,
            num_heads=num_heads,
            mlp_ratio=mlp_ratio,
            dropout=dropout,
            layer_scale_init_value=layer_scale_init_value,
            use_spectral_gating=use_spectral_gating,
        )

        # Classification head
        self.norm = nn.LayerNorm(dim)
        self.head = nn.Linear(dim, num_classes)

        # Initialize position embeddings
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.trunc_normal_(self.cls_token, std=0.02)

        logger.info(
            f"Initialized FFTNetViT with img_size={img_size}, patch_size={patch_size}, num_classes={num_classes}"
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the FFTNetViT model.

        Args:
            x: Input tensor of shape [batch_size, in_channels, img_size, img_size]

        Returns:
            Classification logits of shape [batch_size, num_classes]
        """
        x = self.shape_logger.log_shape(x, "FFTNetViT input")
        batch_size = x.shape[0]

        # Convert image to patches and flatten
        x = self.patch_embed(x)
        x = self.shape_logger.log_shape(x, "After patch embedding")
        x = x.flatten(2).transpose(1, 2)  # [B, num_patches, dim]
        x = self.shape_logger.log_shape(x, "Flattened patches")

        # Add position embeddings and CLS token
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        x = x + self.pos_embed
        x = self.shape_logger.log_shape(x, "After position embedding")

        # Apply FFTNet encoder
        x = self.encoder(x)

        # Classification from CLS token
        x = self.norm(x)
        x = x[:, 0]  # Take CLS token
        x = self.shape_logger.log_shape(x, "CLS token representation")

        # Classification head
        x = self.head(x)

        return self.shape_logger.log_shape(x, "Classification output")


# Example: Create a small FFTNetViT model
def create_fftnetvit_small(num_classes: int = 1000) -> FFTNetViT:
    """
    Create a small FFTNetViT model.

    Args:
        num_classes: Number of classification classes

    Returns:
        FFTNetViT model instance
    """
    model = FFTNetViT(
        img_size=224,
        patch_size=16,
        num_classes=num_classes,
        dim=384,
        depth=8,
        num_heads=6,
        mlp_ratio=4,
        dropout=0.1,
        use_spectral_gating=True,
    )
    logger.info(
        f"Created small FFTNetViT model with {sum(p.numel() for p in model.parameters()):,} parameters"
    )
    return model


# Example usage
if __name__ == "__main__":
    # Create a small FFTNetViT model
    model = create_fftnetvit_small()

    # Generate random input
    batch_size = 4
    x = torch.randn(batch_size, 3, 224, 224)

    # Forward pass
    with torch.no_grad():
        output = model(x)

    logger.info(f"Model output shape: {output.shape}")
