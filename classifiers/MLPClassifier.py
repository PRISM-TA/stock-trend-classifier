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
    
class MLPClassifier_V1(BaseClassifier):
    """With Leaky ReLU"""
    def __init__(self, input_size: int, output_size: int = 3, hidden_size: int = 128, dropout_rate: float = 0.2):
        super().__init__()
        self.model_name = "MLPv1"

        self.model = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.LeakyReLU(0.01),
            nn.Dropout(dropout_rate),
            
            nn.Linear(hidden_size, hidden_size),
            nn.LeakyReLU(0.01),
            nn.Dropout(dropout_rate),
            
            nn.Linear(hidden_size, hidden_size),
            nn.LeakyReLU(0.01),
            nn.Dropout(dropout_rate),
            
            nn.Linear(hidden_size, output_size)
        )
    
    def forward(self, x):
        return self.model(x)
    
class MLPClassifier_V2(BaseClassifier):
    """With Deeper Layers"""
    def __init__(self, input_size: int, output_size: int = 3, hidden_size: int = 128, dropout_rate: float = 0.2):
        super().__init__()
        self.model_name = "MLPv2"

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

class MLPClassifier_V3(BaseClassifier):
    """Larger hidden size"""
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

class MLPClassifier_V4(BaseClassifier):
    """MLPv2 with skip connections"""
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