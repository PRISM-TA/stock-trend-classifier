from classifiers.BaseClassifier import BaseClassifier
import torch
from torch import nn

class CNNClassifier_V1(BaseClassifier):
    def __init__(self, input_size: int, num_classes: int = 3):
        super().__init__()
        self.model_name = "CNNv1"

        # Ensure input is compatible with 20-day structure
        assert input_size % 20 == 0, "input_size must be divisible by 20"
        self.num_features = input_size // 20  # Features per timestep

        # CNN architecture with conv1d layers
        # Use ReLU in first layer, then LeakyReLU in subsequent layers
        self.conv_layers = nn.Sequential(
            # First convolution block with ReLU
            nn.Conv1d(in_channels=self.num_features, out_channels=32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2),
            
            # Second convolution block with LeakyReLU
            nn.Conv1d(32, 64, kernel_size=3, padding=1),
            nn.LeakyReLU(0.01),  # a=0.01 as suggested in the paper
            nn.MaxPool1d(2, 2),
            
            # Third convolution block with LeakyReLU
            nn.Conv1d(64, 128, kernel_size=3, padding=1),
            nn.LeakyReLU(0.01),
            nn.MaxPool1d(2, 2)
        )

        # Calculate flattened size after convolutions
        self.flatten = nn.Flatten()
        
        # Dense layers for classification with improved dropout
        self.dense = nn.Sequential(
            nn.Linear(128 * 2, 256),  # 128 channels * 2 timesteps after pooling
            nn.LeakyReLU(0.01),
            nn.Dropout(0.2),
            nn.Linear(256, 128),
            nn.LeakyReLU(0.01),
            nn.Dropout(0.5),  # Higher dropout in final fully connected layer as paper suggests
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