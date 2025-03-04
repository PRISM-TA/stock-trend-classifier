from classifiers.BaseClassifier import BaseClassifier
import torch
from torch import nn

class LSTMClassifier_V0(BaseClassifier):
    def __init__(self, input_size: int, num_classes: int = 3):
        super().__init__()
        self.model_name = "LSTMv0"

        assert input_size % 20 == 0, "input_size must be divisible by 20"
        self.num_features = input_size // 20
        self.num_timesteps = 20
        
        # Simpler architecture with a single, more powerful LSTM
        self.lstm = nn.LSTM(
            input_size=self.num_features,
            hidden_size=128,
            num_layers=2,  # Two stacked LSTMs
            batch_first=True,
            dropout=0.3,  # Dropout between LSTM layers
            bidirectional=True  # Process sequences in both directions
        )
        
        # The output of bidirectional LSTM is doubled
        lstm_output_size = 128 * 2
        
        # Dense layers with BatchNorm
        self.dense = nn.Sequential(
            nn.BatchNorm1d(lstm_output_size),
            nn.Linear(lstm_output_size, 64),
            nn.ReLU(),
            nn.BatchNorm1d(64),
            nn.Dropout(0.3),
            nn.Linear(64, num_classes)
        )

    def forward(self, x):
        batch_size = x.size(0)
        x = x.view(batch_size, self.num_timesteps, self.num_features)
        
        # Process through LSTM
        lstm_out, (h_n, _) = self.lstm(x)
        
        # For bidirectional LSTM, h_n contains forward and backward hidden states
        # Concatenate the final forward and backward hidden states
        h_forward = h_n[-2, :, :]
        h_backward = h_n[-1, :, :]
        h_combined = torch.cat([h_forward, h_backward], dim=1)
        
        return self.dense(h_combined)