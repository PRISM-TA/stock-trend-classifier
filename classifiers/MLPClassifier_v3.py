# Deeper MLP with hourglass structure
from classifiers.BaseClassifier import BaseClassifier
import torch.nn as nn


class MLPClassifier_V3(BaseClassifier):
    def __init__(self, input_size: int, output_size: int = 3, dropout_rate: float = 0.2):
        super().__init__()
        self.model_name = "MLPv3"

        self.model = nn.Sequential(
            # Expanding path
            nn.Linear(input_size, 128),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            
            nn.Linear(256, 512),  # Widest part
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            
            # Contracting path
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            
            # Output layer
            nn.Linear(64, output_size)
        )
    
    def forward(self, x):
        return self.model(x)