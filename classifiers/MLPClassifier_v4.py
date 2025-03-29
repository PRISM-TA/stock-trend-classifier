# Deeper MLP with skip connections 
from classifiers.BaseClassifier import BaseClassifier
import torch.nn as nn


class MLPClassifier_V4(BaseClassifier):
    def __init__(self, input_size: int, output_size: int = 3, hidden_size: int = 128, dropout_rate: float = 0.2):
        super().__init__()
        self.model_name = "MLPv4"
        
        # Initial layer
        self.layer1 = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout_rate)
        )
        
        # Middle layers with skip connections
        self.layer2 = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout_rate)
        )
        
        self.layer3 = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout_rate)
        )
        
        self.layer4 = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout_rate)
        )
        
        self.layer5 = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout_rate)
        )
        
        self.layer6 = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout_rate)
        )
        
        # Output layer
        self.output_layer = nn.Linear(hidden_size, output_size)
    
    def forward(self, x):
        # Initial layer
        x = self.layer1(x)
        
        # Layer 2 with skip
        residual = x
        x = self.layer2(x)
        x = x + residual  # Skip connection
        
        # Layer 3-4 block with skip
        residual = x
        x = self.layer3(x)
        x = self.layer4(x)
        x = x + residual  # Skip connection
        
        # Layer 5-6 block with skip
        residual = x
        x = self.layer5(x)
        x = self.layer6(x)
        x = x + residual  # Skip connection
        
        # Output
        x = self.output_layer(x)
        return x