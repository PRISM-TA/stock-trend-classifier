import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from sqlalchemy import select, func
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.utils.class_weight import compute_class_weight
from typing import Tuple, List, Dict
import warnings
from data_models.MarketData import MarketData
from data_models.EquityIndicators import EquityIndicators
from data_models.SupervisedClassifierDataset import SupClassifierDataset
from data_models.StaggeredTrainingParam import StaggeredTrainingParam
from lib.data_preprocessing import process_labels, process_equity_indicators
import traceback
from collections import defaultdict


class MarketDataset(Dataset):
    def __init__(self, features, labels):
        self.features = torch.FloatTensor(features)
        self.labels = torch.LongTensor(labels)

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]

    
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
    
def calculate_class_weights(labels):
    """Calculate balanced class weights dynamically for all present classes"""
    # Convert float labels to integers
    labels_int = labels.astype(int)
    # Get unique classes and their counts
    unique_classes = np.unique(labels_int)
    class_counts = np.bincount(labels_int)

    # Calculate weights for each present class
    class_weights = torch.FloatTensor(len(class_counts))
    for i in range(len(class_counts)):
        if class_counts[i] > 0:
            class_weights[i] = 1.0 / class_counts[i]
        else:
            class_weights[i] = 0.0

    # Normalize weights
    if class_weights.sum() > 0:
        class_weights = class_weights / class_weights.sum()
    else:
        # If all weights are zero, use equal weights
        class_weights = torch.ones(len(class_counts)) / len(class_counts)

    return class_weights


def train_model(model, train_loader, val_loader, criterion, optimizer, device, num_epochs=100, patience=50):
    best_loss = float('inf')
    patience_counter = 0
    
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        
        for batch_features, batch_labels in train_loader:
            batch_features = batch_features.to(device)
            batch_labels = batch_labels.to(device).long().squeeze()
            
            optimizer.zero_grad()
            outputs = model(batch_features)
            
            # Handle single sample case
            if outputs.dim() == 2 and outputs.size(0) == 1:
                outputs = outputs.squeeze(0)
                batch_labels = batch_labels.squeeze(0)
            
            loss = criterion(outputs, batch_labels)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
        
        # Validation
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for val_features, val_labels in val_loader:
                val_features = val_features.to(device)
                val_labels = val_labels.to(device).long().squeeze()
                
                val_outputs = model(val_features)
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
            
    return model

def evaluate_model(model, test_loader, device):
    model.eval()
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for features, labels in test_loader:
            features, labels = features.to(device), labels.to(device)
            outputs = model(features)
            _, predicted = torch.max(outputs.data, 1)
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    # Convert to numpy arrays
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    
    # Get unique classes
    classes = np.unique(np.concatenate([all_labels, all_preds]))
    
    # Calculate metrics
    report = classification_report(all_labels, all_preds, output_dict=True)
    conf_matrix = confusion_matrix(all_labels, all_preds)
    
    return report, conf_matrix


def analyze_features(features_df, labels_df):
    """Comprehensive analysis of features and their relationships to trend states"""
    print("\nAnalyzing feature relationships...")
    
    analysis_results = {}
    feature_importance = {}
    
    # Group features by type
    feature_groups = {
        'RSI': ['rsi_9_centered', 'rsi_14_centered', 'rsi_20_centered'],
        'SMA': ['sma_20_normalized', 'sma_50_normalized', 'sma_200_normalized'],
        'EMA': ['ema_20_normalized', 'ema_50_normalized', 'ema_200_normalized'],
        'MACD': ['macd_line_normalized', 'macd_signal_normalized', 'macd_histogram_normalized']
    }
    
    # Analyze each trend state
    for trend_state in [0, 1, 2]:
        state_mask = labels_df['label'] == trend_state
        state_name = {0: "Downtrend", 1: "Sideways", 2: "Uptrend"}[trend_state]
        print(f"\nAnalyzing {state_name} (Label {trend_state}):")
        
        state_features = features_df[state_mask]
        other_features = features_df[~state_mask]
        
        # Analyze by feature group
        for group_name, features in feature_groups.items():
            print(f"\n{group_name} Indicators:")
            group_results = {}
            
            for feature in features:
                state_values = state_features[feature]
                other_values = other_features[feature]
                
                # Basic statistics
                state_mean = state_values.mean()
                state_std = state_values.std()
                other_mean = other_values.mean()
                other_std = other_values.std()
                
                # Calculate separation power
                separation = abs(state_mean - other_mean) / (state_std + other_std) if (state_std + other_std) != 0 else 0
                
                # Calculate correlation with labels
                correlation = np.corrcoef(features_df[feature], labels_df['label'])[0,1]
                
                # Store results
                group_results[feature] = {
                    'mean': state_mean,
                    'std': state_std,
                    'separation': separation,
                    'correlation': correlation
                }
                
                # Update feature importance
                if feature not in feature_importance:
                    feature_importance[feature] = separation
                else:
                    feature_importance[feature] = max(feature_importance[feature], separation)
                
                print(f"  {feature}:")
                print(f"    Mean: {state_mean:.4f}")
                print(f"    Std: {state_std:.4f}")
                print(f"    Separation: {separation:.4f}")
                print(f"    Correlation: {correlation:.4f}")
            
            analysis_results[(state_name, group_name)] = group_results
    
    # Print overall feature importance
    print("\nOverall Feature Importance:")
    for feature, importance in sorted(feature_importance.items(), key=lambda x: x[1], reverse=True):
        print(f"{feature}: {importance:.4f}")
    
    return analysis_results, feature_importance


def staggered_training(session, param: StaggeredTrainingParam, model_name: str, feature_set: str):
    def get_data(session, offset, count, ticker):
        with session() as session:
            query = (
                select(MarketData, EquityIndicators, SupClassifierDataset)
                .join(
                    EquityIndicators,
                    (MarketData.ticker == EquityIndicators.ticker) &
                    (MarketData.report_date == EquityIndicators.report_date)
                ).join(
                    SupClassifierDataset,
                    (MarketData.ticker == SupClassifierDataset.ticker) &
                    (MarketData.report_date == SupClassifierDataset.end_date)
                )
                .where(MarketData.ticker == ticker)
            )
               
            query = (query
                    .order_by(MarketData.report_date)
                    .offset(offset)
                    .limit(count))
                    
            query_result = session.execute(query).all()
            feature_df = process_equity_indicators([(record[0], record[1]) for record in query_result])
            labels_df = process_labels([(record[2]) for record in query_result])
            return feature_df, labels_df
        

    def scale_data(feat_train_df, label_train_df, feat_pred_df, label_pred_df):
        scaler = StandardScaler()
        train_features = scaler.fit_transform(feat_train_df.values)
        train_labels = label_train_df['label'].values
        pred_features = scaler.transform(feat_pred_df.values)
        pred_labels = label_pred_df['label'].values
        return train_features, train_labels, pred_features, pred_labels

    def create_dataloader(train_features, train_labels, pred_features, pred_labels, min_batch_size=32):
        train_dataset = MarketDataset(train_features, train_labels)
        pred_dataset = MarketDataset(pred_features, pred_labels)
        batch_size = min(min_batch_size, len(train_dataset))
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False)
        pred_loader = DataLoader(pred_dataset, batch_size=1, shuffle=False)
        return train_loader, pred_loader
    
    def get_available_data_count(session, ticker, start_date=None):
        with session() as session:
            query = (
                select(
                    func.count(MarketData.report_date)
                ).where(MarketData.ticker == ticker)
            )
            
                
            query_result = session.execute(query).all()
            print(f"[DEBUG] available_data_count for {ticker}: {query_result[0][0]}")
            return query_result[0][0]
    
    max_epochs = 1000
    classifier_result_list = []

    available_data_count = get_available_data_count(session, param.ticker) - param.training_day_count
    if available_data_count < 0:
        print(f"[DEBUG] Not enough training data available for {param.ticker}")
    training_offset = 0
    prediction_offset = param.training_day_count

    while available_data_count > param.prediction_day_count:
        print(f"[DEBUG] Available data count: {available_data_count}")
        print(f"[DEBUG] Training on {param.ticker} from {training_offset} to {prediction_offset}")
        print(f"[DEBUG] Testing on {param.ticker} from {prediction_offset} to {prediction_offset + param.prediction_day_count}")

        feat_train_df, label_train_df = get_data(session, training_offset, param.training_day_count, param.ticker)
        feat_pred_df, label_pred_df = get_data(session, prediction_offset, param.prediction_day_count, param.ticker)
        train_features, train_labels, pred_features, pred_labels = scale_data(feat_train_df, label_train_df, feat_pred_df, label_pred_df)
        train_loader, pred_loader = create_dataloader(train_features, train_labels, pred_features, pred_labels)

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = MLPClassifier(input_size=train_features.shape[1]).to(device)
        class_weights = calculate_class_weights(train_labels)
        criterion = nn.CrossEntropyLoss(weight=class_weights.to(device))
        # criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.SGD(model.parameters(), lr=0.001)
        model = train_model(model, train_loader, train_loader, criterion, optimizer, device, num_epochs=max_epochs)

        model.eval()
        predictions = []
        actual_labels = []
        probabilities = []
        with torch.no_grad():
            for features, labels in pred_loader:
                features = features.to(device)
                outputs = model(features)
                # Get probabilities using softmax
                probs = torch.softmax(outputs, dim=1).cpu().numpy()[0]
                prediction = torch.argmax(outputs, dim=1).item()
                predictions.append(prediction)
                actual_labels.append(labels.item())
                probabilities.append(probs)
                print(f"Prediction: {prediction}, Actual Label: {labels.item()}, Probabilities: [{float(probs[0]):.4f}, {float(probs[1]):.4f}, {float(probs[2]):.4f}]")

        classifier_result = {
            'predictions': predictions,
            'actual_labels': actual_labels,
            'training_offset': training_offset,
            'prediction_offset': prediction_offset,
            'probabilities': probabilities, 
            'model': model_name,
            'feature_set': feature_set
        }
        classifier_result_list.append(classifier_result)

        training_offset+=param.prediction_day_count
        prediction_offset+=param.prediction_day_count
        available_data_count-=param.prediction_day_count
    
    return classifier_result_list