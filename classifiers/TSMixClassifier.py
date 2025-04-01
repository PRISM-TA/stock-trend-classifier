from classifiers.BaseClassifier import BaseClassifier
from classifiers.tsmixer.layers import MixerLayer, TimeBatchNorm2d, feature_to_time, time_to_feature

import torch
import torch.nn as nn
import torch.nn.functional as F


class TSMixerClassifier_V0(BaseClassifier):
    """TSMixer model for time series classification.

    This model uses a series of mixer layers to process time series data,
    followed by MLP layers for classification.
    """
    model_name="TSMixv0"

    def __init__(
        self,
        input_size: int,
        sequence_length: int = 20,
        num_classes: int = 3,
        hidden_size: int = 256,
        activation_fn: str = "relu",
        num_blocks: int = 2,
        dropout_rate: float = 0.2,
        ff_dim: int = 128,
        normalize_before: bool = True,
        norm_type: str = "batch",
    ):
        super().__init__()

        # Ensure input is compatible with 20-day structure
        assert input_size % sequence_length == 0, f"input_size must be divisible by {sequence_length}"
        self.input_channels = input_size // sequence_length  # Features per timestep

        # Transform activation_fn to callable
        activation_fn = getattr(F, activation_fn)

        # Transform norm_type to callable
        assert norm_type in {
            "batch",
            "layer",
        }, f"Invalid norm_type: {norm_type}, must be one of batch, layer."
        norm_type = TimeBatchNorm2d if norm_type == "batch" else nn.LayerNorm

        # Build mixer layers
        self.mixer_layers = self._build_mixer(
            num_blocks,
            self.input_channels,
            self.input_channels,  # Keep same number of channels throughout
            ff_dim=ff_dim,
            activation_fn=activation_fn,
            dropout_rate=dropout_rate,
            sequence_length=sequence_length,
            normalize_before=normalize_before,
            norm_type=norm_type,
        )

        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(sequence_length * self.input_channels, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_size, num_classes)
        )

    def _build_mixer(
        self, num_blocks: int, input_channels: int, output_channels: int, **kwargs
    ):
        """Build the mixer blocks for the model."""
        channels = [self.input_channels] * num_blocks

        return nn.Sequential(
            *[
                MixerLayer(input_channels=in_ch, output_channels=out_ch, **kwargs)
                for in_ch, out_ch in zip(channels[:-1], channels[1:])
            ]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass of the TSMixer classifier.

        Args:
            x (torch.Tensor): Input time series tensor of shape (batch_size, sequence_length, self.input_channels)

        Returns:
            torch.Tensor: Classification logits of shape (batch_size, num_classes)
        """

        batch_size = x.size(0)
        x = x.view(batch_size, self.input_channels, 20)

        # Convert to feature representation
        x = time_to_feature(x)  # (batch_size, self.input_channels, sequence_length)
        
        # Process through mixer layers
        x = self.mixer_layers(x)
        
        # Convert back to time representation
        x = feature_to_time(x)  # (batch_size, sequence_length, self.input_channels)
        
        # Flatten for classification
        x = x.reshape(x.shape[0], -1)
        
        # Classification
        return self.classifier(x)


if __name__ == "__main__":
    # Example usage
    model = TSMixerClassifier_V0(
        sequence_length=10,
        input_channels=2,
        num_classes=3
    )
    x = torch.randn(3, 10, 2)  # batch_size=3, sequence_length=10, self.input_channels=2
    y = model(x)
    print(y.shape)  # Should be (3, 3) - (batch_size, num_classes)
    # Example usage
    model = TSMixerClassifier_V0(
        sequence_length=10,
        input_channels=2,
        num_classes=3
    )
    x = torch.randn(3, 10, 2)  # batch_size=3, sequence_length=10, self.input_channels=2
    y = model(x)
    print(y.shape)  # Should be (3, 3) - (batch_size, num_classes)