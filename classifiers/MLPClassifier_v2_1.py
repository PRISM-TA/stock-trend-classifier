# Deeper MLP with larger hidden size
from classifiers.BaseClassifier import BaseClassifier
import torch.nn as nn


class MLPClassifier_V2_1(BaseClassifier):
    def __init__(self, input_size: int, output_size: int = 3, hidden_size: int = 512, dropout_rate: float = 0.2):
        super().__init__()
        self.model_name = "MLPv2_1"

        self.model = nn.Sequential(
            # Layer 1
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            
            # Layer 2
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            
            # Layer 3
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            
            # Layer 4
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            
            # Layer 5
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            
            # Layer 6
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            
            # Output layer
            nn.Linear(hidden_size, output_size)
        )
    
    def forward(self, x):
        return self.model(x)