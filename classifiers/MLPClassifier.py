from classifiers.BaseClassifier import BaseClassifier
import torch.nn as nn


class MLPClassifier_V0(BaseClassifier):

    def __init__(self, input_size: int, output_size: int = 3, hidden_size: int = 128, dropout_rate: float = 0.2):
        super().__init__()
        self.model_name = "MLPv0"

        self.model = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_size, output_size)
        )
    
    def forward(self, x):
        return self.model(x)