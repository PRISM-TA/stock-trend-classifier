from models.BaseHyperParam import BaseHyperParam
import torch
import torch.nn as nn
import os
import json
from datetime import datetime


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

    def train_classifier(self, criterion: nn.Module, optimizer: torch.optim.Optimizer, param: BaseHyperParam, 
                     train_loader, val_loader=None, ticker=None, feature_set=None, window_num=None):
        best_loss = float('inf')
        patience_counter = 0
        
        # Initialize lists to store losses
        train_losses = []
        val_losses = []
        
        for epoch in range(param.num_epochs):
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
            
            # Calculate average training loss for this epoch
            avg_train_loss = total_loss / len(train_loader)
            train_losses.append(avg_train_loss)

            if val_loader is not None:
                self.eval()
                val_loss = 0
                with torch.no_grad():
                    for val_features, val_labels in val_loader:
                        val_features = val_features.to(self.device)
                        val_labels = val_labels.to(self.device).long().squeeze()
                        
                        val_outputs = self(val_features)
                        val_loss += criterion(val_outputs, val_labels).item()
                
                # Calculate average validation loss
                avg_val_loss = val_loss / len(val_loader)
                val_losses.append(avg_val_loss)
                
                if param.early_stopping:
                    # Early stopping logic
                    if avg_val_loss < best_loss:
                        best_loss = avg_val_loss
                        patience_counter = 0
                    else:
                        patience_counter += 1
                        
                    if patience_counter >= param.patience:
                        print(f"Early stopping at epoch {epoch}")
                        break
        
        # Save loss history at the end of training (outside the epoch loop)
        loss_file = self.save_loss_history(train_losses, val_losses, param, ticker=ticker, feature_set=feature_set, window_num=window_num)
        print(f"Training completed. Loss history saved to {loss_file}")
        
        return train_losses, val_losses
        

    def save_loss_history(self, train_losses, val_losses, param, ticker=None, feature_set=None, window_num=None):
        """
        Save training and validation losses to a JSON file in the loss_record folder
        
        Args:
            train_losses: List of training losses
            val_losses: List of validation losses
            param: Training hyperparameters
            ticker: The stock ticker (e.g., 'AAPL')
            feature_set: The feature set object
            window_num: The window number for staggered training
        """
        # Create loss_record directory if it doesn't exist
        os.makedirs('loss_record', exist_ok=True)
        
        # Create filename parts list starting with model name
        filename_parts = [self.model_name]
        
        # Add ticker if provided
        if ticker:
            filename_parts.append(ticker)
        
        # Add feature set acronym if provided
        if feature_set and hasattr(feature_set, 'set_name'):
            # Split by spaces and get first letter of each word
            feature_set_name = feature_set.set_name
            words = feature_set_name.replace('+', ' ').split()
            acronym = ''.join(word[0].upper() for word in words)
            
            # Add any parenthesized content
            if '(' in feature_set_name:
                start_idx = feature_set_name.find('(')
                end_idx = feature_set_name.find(')')
                if end_idx > start_idx:
                    acronym += feature_set_name[start_idx:end_idx+1]
            
            filename_parts.append(acronym)
        
        # Add window number
        if window_num is not None:
            filename_parts.append(f"window{window_num}")
        
        # Join parts with underscores
        filename = f"loss_record/{'_'.join(filename_parts)}.json"
        
        # Current timestamp for data inside the file
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Prepare the data to save
        feature_set_name = getattr(feature_set, 'set_name', str(feature_set)) if feature_set else None
        
        loss_data = {
            "model_name": self.model_name,
            "ticker": ticker,
            "feature_set": feature_set_name,
            "window_num": window_num,
            "hyperparameters": {
                "learning_rate": getattr(param, 'learning_rate', None),
                "batch_size": getattr(param, 'batch_size', None),
                "num_epochs": param.num_epochs,
                "early_stopping": param.early_stopping,
                "patience": param.patience if param.early_stopping else None
            },
            "train_loss": train_losses,
            "val_loss": val_losses if val_losses else None,
            "timestamp": timestamp
        }
        
        # Save to file
        with open(filename, 'w') as f:
            json.dump(loss_data, f, indent=4)
        
        print(f"Loss history saved to {filename}")
        return filename