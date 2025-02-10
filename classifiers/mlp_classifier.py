import torch.nn as nn


class MLPClassifier(nn.Module):
    def __init__(self, input_size: int, hidden_size: int = 128, dropout_rate: float = 0.2):
        super(MLPClassifier, self).__init__()
        
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
            nn.Linear(hidden_size, 3)  # Keep 3 classes since we know we have labels 0, 1, 2
        )
    
    def forward(self, x):
        return self.model(x)