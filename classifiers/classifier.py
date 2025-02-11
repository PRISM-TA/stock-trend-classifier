import torch
import torch.nn as nn


class BaseClassifier(nn.Module):
    model_name: str # internal identifier for the model
    device: str # which device to use for training / prediction
    
    def __init__(self):
        super().__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.to(self.device)
        pass

    def forward():
        pass

    def train_classifier(self, train_loader, val_loader, criterion, optimizer, num_epochs=100, early_stopping=True, patience=50)->None:
        best_loss = float('inf')
        patience_counter = 0
        
        for epoch in range(num_epochs):
            self.train()
            total_loss = 0
            
            for batch_features, batch_labels in train_loader:
                batch_features = batch_features.to(self.device)
                batch_labels = batch_labels.to(self.device).long().squeeze()
                
                optimizer.zero_grad()
                outputs = self(batch_features)
                
                # Handle single sample case
                if outputs.dim() == 2 and outputs.size(0) == 1:
                    outputs = outputs.squeeze(0)
                    batch_labels = batch_labels.squeeze(0)
                
                loss = criterion(outputs, batch_labels)
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
            
            # Validation
            self.eval()
            val_loss = 0
            with torch.no_grad():
                for val_features, val_labels in val_loader:
                    val_features = val_features.to(self.device)
                    val_labels = val_labels.to(self.device).long().squeeze()
                    
                    val_outputs = self(val_features)
                    val_loss += criterion(val_outputs, val_labels).item()
            
            if epoch % 50 == 0:
                print(f"Epoch {epoch:3d}: Loss = {total_loss/len(train_loader):.4f}")
            
            # Early stopping
            if val_loss < best_loss:
                best_loss = val_loss
                patience_counter = 0
            else:
                patience_counter += 1
                
            if patience_counter >= patience:
                print(f"Early stopping at epoch {epoch}")
                break

class MLPClassifier(BaseClassifier):
    def __init__(self, input_size: int, output_size: int = 3, hidden_size: int = 128, dropout_rate: float = 0.2):
        super().__init__()
        
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