from classifiers.BaseClassifier import BaseClassifier
import torch
from torch import nn

class CNNClassifier_V0_drop(BaseClassifier):
    def __init__(self, input_size: int, num_classes: int = 3):
        super().__init__()
        self.model_name = "CNNv0_drop"

        # Ensure input is compatible with 20-day structure
        assert input_size % 20 == 0, "input_size must be divisible by 20"
        self.num_features = input_size // 20  # Features per timestep

        # CNN architecture
        self.conv_layers = nn.Sequential(
            # First convolution block
            nn.Conv1d(in_channels=self.num_features, out_channels=32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2),
            
            # Second convolution block
            nn.Conv1d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(2, 2)
        )

        # Calculate flattened size after convolutions
        self.flatten = nn.Flatten()
        
        # Dense layers for classification
        self.dense = nn.Sequential(
            nn.Linear(64 * 5, 128),  # 64 channels * 5 timesteps after pooling
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        # Reshape input to (batch, features, timesteps)
        batch_size = x.size(0)
        x = x.view(batch_size, self.num_features, 20)
        
        # Process through CNN
        x = self.conv_layers(x)
        x = self.flatten(x)
        return self.dense(x)