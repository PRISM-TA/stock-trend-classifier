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
from data.MarketData import MarketData
from data.EquityIndicators import EquityIndicators
from data.SupervisedClassifierDataset import SupClassifierDataset
import traceback
from collections import defaultdict
from data.StaggeredTrainingParam import StaggeredTrainingParam

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
    
    
def analyze_current_features(features_df, labels_df):
    """Detailed analysis of how each feature relates to each trend state"""
    print("Analyzing feature-label relationships...\n")
    
    analysis_results = {}
    feature_importance = {}
    
    # For each trend state (0: Downtrend, 1: Sideways, 2: Uptrend)
    for trend_state in [0, 1, 2]:
        state_mask = labels_df['label'] == trend_state
        state_name = {0: "Downtrend", 1: "Sideways", 2: "Uptrend"}[trend_state]
        print(f"\nAnalyzing {state_name} (Label {trend_state}):")
        
        # Get features for this trend state
        state_features = features_df[state_mask]
        other_features = features_df[~state_mask]
        
        # Calculate separation power for each feature
        state_analysis = {}
        for feature in features_df.columns:
            state_values = state_features[feature]
            other_values = other_features[feature]
            
            # Calculate mean and std for this feature in this state
            state_mean = state_values.mean()
            state_std = state_values.std()
            other_mean = other_values.mean()
            other_std = other_values.std()
            
            # Calculate separation power
            separation = abs(state_mean - other_mean) / (state_std + other_std) if (state_std + other_std) != 0 else 0
            
            state_analysis[feature] = {
                'mean': state_mean,
                'std': state_std,
                'separation': separation
            }
            
            # Update overall feature importance
            if feature not in feature_importance:
                feature_importance[feature] = separation
            else:
                feature_importance[feature] = max(feature_importance[feature], separation)
        
        # Sort features by separation power
        sorted_features = sorted(state_analysis.items(), key=lambda x: x[1]['separation'], reverse=True)
        
        # # Print top 5 most discriminative features for this state
        # print("\nTop 5 most discriminative features:")
        # for feature, metrics in sorted_features[:5]:
        #     print(f"{feature}:")
        #     print(f"  Separation power: {metrics['separation']:.4f}")
        #     print(f"  Mean: {metrics['mean']:.4f}")
        #     print(f"  Std: {metrics['std']:.4f}")
        
        analysis_results[trend_state] = state_analysis
    
    # Print overall most important features
    print("\nOverall most important features:")
    sorted_importance = sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)
    for feature, importance in sorted_importance[:10]:
        print(f"{feature}: {importance:.4f}")
    
    return analysis_results, feature_importance

def analyze_label_patterns(labels_df):
    """Analyze patterns in the labels themselves"""
    print("\nAnalyzing label patterns...")
    
    # Calculate transition probabilities
    transitions = defaultdict(lambda: defaultdict(int))
    prev_label = None
    
    for label in labels_df['label']:
        if prev_label is not None:
            transitions[prev_label][label] += 1
        prev_label = label
    
    # Print transition probabilities
    print("\nTransition probabilities:")
    for from_state in sorted(transitions.keys()):
        total = sum(transitions[from_state].values())
        print(f"\nFrom state {from_state}:")
        for to_state in sorted(transitions[from_state].keys()):
            prob = transitions[from_state][to_state] / total
            print(f"  To state {to_state}: {prob:.2%}")
    
    # Calculate state durations
    state_durations = defaultdict(list)
    current_state = None
    current_duration = 0
    
    for label in labels_df['label']:
        if current_state is None:
            current_state = label
            current_duration = 1
        elif label == current_state:
            current_duration += 1
        else:
            state_durations[current_state].append(current_duration)
            current_state = label
            current_duration = 1
    
    # Add final state duration
    if current_state is not None:
        state_durations[current_state].append(current_duration)
    
    # # Print duration statistics
    # print("\nState duration statistics:")
    # for state in sorted(state_durations.keys()):
    #     durations = state_durations[state]
    #     print(f"\nState {state}:")
    #     print(f"  Mean duration: {np.mean(durations):.1f} days")
    #     print(f"  Median duration: {np.median(durations):.1f} days")
    #     print(f"  Min duration: {min(durations)} days")
    #     print(f"  Max duration: {max(durations)} days")
    
    return transitions, state_durations

def remove_outliers(df):
    Q1 = df.quantile(0.25)
    Q3 = df.quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    return df[~((df < lower_bound) | (df > upper_bound))]

def process_market_data(raw_data) -> pd.DataFrame:
    """Process market data using existing preprocessing"""
    # Convert raw_data to DataFrame first
    data = pd.DataFrame([{
        'report_date': record[0].report_date,
        'rsi_9': record[1].rsi_9,
        'rsi_14': record[1].rsi_14,
        'rsi_20': record[1].rsi_20,
        'sma_20': record[1].sma_20,
        'sma_50': record[1].sma_50,
        'sma_200': record[1].sma_200,
        'ema_20': record[1].ema_20,
        'ema_50': record[1].ema_50,
        'ema_200': record[1].ema_200,
        'macd_12_26_9_line': record[1].macd_12_26_9_line,
        'macd_12_26_9_signal': record[1].macd_12_26_9_signal,
        'macd_12_26_9_histogram': record[1].macd_12_26_9_histogram,
        'rv_10': record[1].rv_10,
        'rv_20': record[1].rv_20,
        'rv_30': record[1].rv_30,
        'rv_60': record[1].rv_60,
        'hls_10': record[1].hls_10,
        'hls_20': record[1].hls_20
    } for record in raw_data])
    
    data.set_index('report_date', inplace=True)
    
    # Initialize processed data DataFrame
    processed_data = pd.DataFrame(index=data.index)
    
    # RSI
    processed_data['rsi_9_centered'] = (data['rsi_9'] - 50) / 50
    processed_data['rsi_14_centered'] = (data['rsi_14'] - 50) / 50
    processed_data['rsi_20_centered'] = (data['rsi_20'] - 50) / 50
    
    # SMA
    for period in [20, 50, 200]:
        col = f'sma_{period}'
        processed_data[f"{col}_sta"] = data[col].pct_change()
        # processed_data[f"{col}_sta_no_out"] = remove_outliers(processed_data[f"{col}_sta"])
        # processed_data[f'{col}_normalized'] = 2 * (
        #     processed_data[f'{col}_sta_no_out'] - processed_data[f'{col}_sta_no_out'].min()
        # ) / (processed_data[f'{col}_sta_no_out'].max() - processed_data[f'{col}_sta_no_out'].min()) - 1
        
    processed_data['sma_20_normalized'] = 50 * processed_data['sma_20_sta']
    processed_data['sma_50_normalized'] = 60 * processed_data['sma_50_sta'] # 70
    processed_data['sma_200_normalized'] = 80 * processed_data['sma_200_sta'] # 100
    
    # EMA
    for period in [20, 50, 200]:
        col = f'ema_{period}'
        processed_data[f"{col}_sta"] = data[col].pct_change()
        # processed_data[f"{col}_sta_no_out"] = remove_outliers(processed_data[f"{col}_sta"])
        # processed_data[f'{col}_normalized'] = 2 * (
        #     processed_data[f'{col}_sta_no_out'] - processed_data[f'{col}_sta_no_out'].min()
        # ) / (processed_data[f'{col}_sta_no_out'].max() - processed_data[f'{col}_sta_no_out'].min()) - 1
        
    processed_data['ema_20_normalized'] = 50 * processed_data['ema_20_sta']
    processed_data['ema_50_normalized'] = 70 * processed_data['ema_50_sta']
    processed_data['ema_200_normalized'] = 120 * processed_data['ema_200_sta']
    
    # MACD
    for component in ['line', 'signal', 'histogram']:
        col = f'macd_12_26_9_{component}'
        processed_data[f"macd_{component}_sta"] = data[col].pct_change()
        processed_data[f"macd_{component}_sta_no_out"] = remove_outliers(processed_data[f"macd_{component}_sta"])
        processed_data[f'macd_{component}_normalized'] = 2 * (
            processed_data[f"macd_{component}_sta_no_out"] - processed_data[f"macd_{component}_sta_no_out"].min()
        ) / (processed_data[f"macd_{component}_sta_no_out"].max() - processed_data[f"macd_{component}_sta_no_out"].min()) - 1
        
    # RV
    for period in [10, 20, 30, 60]:
        col = f'rv_{period}'
        processed_data[f"{col}_sta"] = data[col].pct_change()
        # processed_data[f"{col}_sta_no_out"] = remove_outliers(processed_data[f"{col}_sta"])
        # processed_data[f'{col}_normalized'] = 2 * (
        #     processed_data[f'{col}_sta_no_out'] - processed_data[f'{col}_sta_no_out'].min()
        # ) / (processed_data[f'{col}_sta_no_out'].max() - processed_data[f'{col}_sta_no_out'].min()) - 1\
            
    # HLS
    for period in [10, 20]:
        col = f'hls_{period}'
        processed_data[f"{col}_sta"] = data[col].pct_change()
        # processed_data[f"{col}_sta_no_out"] = remove_outliers(processed_data[f"{col}_sta"])
        # processed_data[f'{col}_normalized'] = 2 * (
        #     processed_data[f'{col}_sta_no_out'] - processed_data[f'{col}_sta_no_out'].min()
        # ) / (processed_data[f'{col}_sta_no_out'].max() - processed_data[f'{col}_sta_no_out'].min()) - 1
        
    processed_data['hls_10_normalized'] = 2 * (processed_data['hls_10_sta'])
    processed_data['hls_20_normalized'] = 3 * (processed_data['hls_20_sta'])
    
    # Select final features
    final_features = [
        'rsi_9_centered', 'rsi_14_centered', 'rsi_20_centered',
        'sma_20_normalized', 'sma_50_normalized', 'sma_200_normalized',
        'ema_20_normalized', 'ema_50_normalized', 'ema_200_normalized',
        'macd_line_normalized', 'macd_signal_normalized', 'macd_histogram_normalized'
        # 'rv_10_sta', 'rv_20_sta', 'rv_30_sta', 'rv_60_sta',
        # 'hls_10_normalized', 'hls_20_normalized'
    ]
    
    # Handle any NaN values
    result = processed_data[final_features].fillna(0)
    return result

def analyze_current_features(features_df, labels_df):
    """Analyze how each feature relates to trend states"""
    print("\nAnalyzing feature relationships...")
    
    analysis_results = {}
    feature_importance = {}
    
    # Group features by type
    feature_groups = {
        'RSI': ['rsi_9_centered', 'rsi_14_centered', 'rsi_20_centered'],
        'SMA': ['sma_20_normalized', 'sma_50_normalized', 'sma_200_normalized'],
        'EMA': ['ema_20_normalized', 'ema_50_normalized', 'ema_200_normalized'],
        'MACD': ['macd_line_normalized', 'macd_signal_normalized', 'macd_histogram_normalized']
        # 'RV': ['rv_10_sta', 'rv_20_sta', 'rv_30_sta', 'rv_60_sta'],
        # 'HLS': ['hls_10_normalized', 'hls_20_normalized']
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
            # print(f"\n{group_name} Indicators:")
            for feature in features:
                state_values = state_features[feature]
                other_values = other_features[feature]
                
                state_mean = state_values.mean()
                state_std = state_values.std()
                separation = abs(state_mean - other_values.mean()) / (state_std + other_values.std())
                
                # print(f"  {feature}:")
                # print(f"    Mean: {state_mean:.4f}")
                # print(f"    Std: {state_std:.4f}")
                # print(f"    Separation: {separation:.4f}")
                
                if feature not in feature_importance:
                    feature_importance[feature] = separation
                else:
                    feature_importance[feature] = max(feature_importance[feature], separation)
    
    return analysis_results, feature_importance

def process_labels(raw_labels) -> pd.DataFrame:
    """Process raw labels into labels DataFrame"""
    labels_data = []
    dates = []
    
    for item in raw_labels:
        labels_data.append({'label': item.label})
        dates.append(item.start_date)
    
    df = pd.DataFrame(labels_data, index=dates)
    return df


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

def get_data_for_period(session, 
                       start_date: datetime, 
                       end_date: datetime, 
                       ticker: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Get market data, indicators and labels for a specific period"""
    try:
        print(f"\nProcessing period: {start_date} to {end_date}")
        
        # Query labels first to determine actual date range
        labels_query = (
            select(
                SupClassifierDataset.start_date,
                SupClassifierDataset.end_date,
                SupClassifierDataset.label
            )
            .where(
                SupClassifierDataset.ticker == ticker,
                SupClassifierDataset.start_date.between(start_date, end_date)
            )
            .order_by(SupClassifierDataset.start_date)
        )
        
        labels = session.execute(labels_query).all()
        
        if not labels:
            raise ValueError(f"No labels found for period {start_date} to {end_date}")
            
        actual_end_date = labels[-1][0]
        print(f"Adjusting end date to: {actual_end_date} based on label availability")
        
        # Get market data and indicators
        query = (
            select(MarketData, EquityIndicators)
            .join(
                EquityIndicators,
                (MarketData.ticker == EquityIndicators.ticker) &
                (MarketData.report_date == EquityIndicators.report_date)
            )
            .where(
                MarketData.ticker == ticker,
                MarketData.report_date.between(start_date, actual_end_date)
            )
            .order_by(MarketData.report_date)
        )
        
        result = session.execute(query).all()
        
        if not result:
            raise ValueError(f"No market data found for period {start_date} to {actual_end_date}")
        
        print(f"Records found - Market data: {len(result)}, Labels: {len(labels)}")
        
        # Process the data into DataFrames
        features_df = process_market_data(result)
        labels_df = process_labels(labels)
        
        # Ensure alignment between features and labels
        common_dates = features_df.index.intersection(labels_df.index)
        features_df = features_df.loc[common_dates]
        labels_df = labels_df.loc[common_dates]
        
        # NEW CODE START: Feature Importance Analysis
        print("\nFeature Importance Analysis:")
        correlations = {}
        for column in features_df.columns:
            correlation = np.corrcoef(features_df[column], labels_df['label'])[0,1]
            correlations[column] = abs(correlation)
        
        # Sort and print feature importance
        sorted_features = dict(sorted(correlations.items(), key=lambda x: x[1], reverse=True))
        print("\nFeature Importance (by correlation with labels):")
        for feature, importance in sorted_features.items():
            print(f"{feature}: {importance:.4f}")
            
        feature_analysis, feature_importance = analyze_current_features(features_df, labels_df)
        transition_patterns, duration_patterns = analyze_label_patterns(labels_df)
        # NEW CODE END
        
        # Analyze class distribution
        label_counts = labels_df['label'].value_counts()
        print("\nClass distribution:")
        for label, count in label_counts.items():
            percentage = (count / len(labels_df)) * 100
            print(f"Class {label}: {count} samples ({percentage:.1f}%)")
        
        print(f"Final aligned shapes - Features: {features_df.shape}, Labels: {labels_df.shape}")
        
        return features_df, labels_df
        
    except Exception as e:
        print(f"Error in get_data_for_period: {str(e)}")
        raise
    
    
def analyze_class_distribution(train_labels, test_labels, verbose=True, 
                             max_shift_threshold=0.4,      # Increased from 0.2 to 0.4
                             min_class_ratio=0.05):        # Allow classes with 0 representation
    """
    Analyze class distribution in train and test sets with more flexible criteria
    """
    train_labels = train_labels.flatten()
    test_labels = test_labels.flatten()
    
    # Get distributions
    train_dist = pd.Series(train_labels).value_counts(normalize=True)
    test_dist = pd.Series(test_labels).value_counts(normalize=True)
    
    if verbose:
        print("\nClass Distribution Analysis:")
        print("Training Distribution:")
        for cls in sorted(train_dist.index):
            count = (train_labels == cls).sum()
            print(f"Class {cls}: {count} samples ({train_dist[cls]*100:.1f}%)")
        
        print("\nTesting Distribution:")
        for cls in sorted(test_dist.index):
            count = (test_labels == cls).sum()
            print(f"Class {cls}: {count} samples ({test_dist[cls]*100:.1f}%)")
    
    # Check if any class has too few samples
    for dist in [train_dist, test_dist]:
        if any(ratio < min_class_ratio for ratio in dist.values):
            if verbose:
                print(f"\nWARNING: Some classes have less than {min_class_ratio*100}% representation")
            return False
    
    # Calculate distribution shifts
    all_classes = sorted(set(train_dist.index) | set(test_dist.index))
    max_shift = 0
    total_shift = 0
    n_shifts = 0
    
    for cls in all_classes:
        train_ratio = train_dist.get(cls, 0)
        test_ratio = test_dist.get(cls, 0)
        shift = abs(train_ratio - test_ratio)
        max_shift = max(max_shift, shift)
        total_shift += shift
        n_shifts += 1
    
    avg_shift = total_shift / n_shifts
    
    if verbose:
        print(f"\nDistribution Shift Analysis:")
        print(f"Maximum shift: {max_shift:.2f}")
        print(f"Average shift: {avg_shift:.2f}")
    
    # Accept if either condition is met:
    # 1. Max shift is within threshold
    # 2. Average shift is within threshold/2
    if max_shift > max_shift_threshold and avg_shift > max_shift_threshold/2:
        if verbose:
            print(f"\nWARNING: Distribution shift too large (max: {max_shift:.2f}, avg: {avg_shift:.2f})")
        return False
    
    return True

def filter_by_min_samples(features_df, labels_df, min_samples=5, verbose=True):
    """Filter out classes with too few samples"""
    label_counts = labels_df['label'].value_counts()
    valid_classes = label_counts[label_counts >= min_samples].index
    
    if verbose:
        print("\nClass filtering:")
        print("Original distribution:")
        for cls in sorted(label_counts.index):
            print(f"Class {cls}: {label_counts[cls]} samples")
        
        excluded = set(label_counts.index) - set(valid_classes)
        if excluded:
            print(f"\nExcluding classes with < {min_samples} samples: {excluded}")
    
    mask = labels_df['label'].isin(valid_classes)
    return features_df[mask], labels_df[mask]

def rolling_window_training(db_session, 
                          ticker: str, #Which stock to train on
                          start_date: str = None, #Start date for what?
                          end_date: str = None, #End date for what?
                          train_window_size: int = 240, #The number of days used for training
                          predict_window_size: int = 20, #The number of days to predict for??
                          max_predictions: int = 60, #The number of predictions to make
                          min_samples_per_class: int = 2, # Why do we need this?
                          max_epochs: int = 100,
                          early_stopping_patience: int = 50) -> List[Dict]:
    """
    Args:
        train_window_size: Training window in calendar days
        predict_window_size: Number of days to predict trend for (e.g., 20 days ahead)
        max_predictions: Maximum number of predictions to make
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Get the valid date range from database
    query = (
        select(
            func.min(SupClassifierDataset.start_date).label('min_date'),
            func.max(SupClassifierDataset.start_date).label('max_date')
        )
        .where(SupClassifierDataset.ticker == ticker)
    )
    date_range = db_session.execute(query).first()
    
    if not date_range or not date_range.min_date or not date_range.max_date:
        raise ValueError(f"No data found for ticker {ticker}")

    db_start = date_range.min_date
    db_end = date_range.max_date
    
    # Convert string dates to datetime.date if provided as strings
    if start_date:
        if isinstance(start_date, str):
            start_date = datetime.strptime(start_date, '%Y-%m-%d').date()
        # Ensure start_date is not before database start
        start_date = max(start_date, db_start)
    else:
        start_date = db_start
        
    if end_date:
        if isinstance(end_date, str):
            end_date = datetime.strptime(end_date, '%Y-%m-%d').date()
        # Ensure end_date is not after database end
        end_date = min(end_date, db_end)
    else:
        end_date = db_end
    
    print(f"\nDatabase period: {db_start} to {db_end}")
    print(f"Selected period: {start_date} to {end_date}")
    
    # Get valid trading days for testing period
    valid_days_query = (
        select(EquityIndicators.report_date)
        .where(
            EquityIndicators.ticker == ticker,
            EquityIndicators.report_date.between(start_date, end_date),
            EquityIndicators.rsi_14.isnot(None),
            EquityIndicators.macd_12_26_9_line.isnot(None),
            EquityIndicators.sma_20.isnot(None),
            EquityIndicators.ema_20.isnot(None)
        )
        .order_by(EquityIndicators.report_date)
    )
    
    valid_trading_days = [date[0] for date in db_session.execute(valid_days_query).fetchall()]
    
    if not valid_trading_days:
        raise ValueError("No valid trading days found")
    
    print(f"Found {len(valid_trading_days)} valid trading days")
    # =============================================================================
    # ^^ This part should be wrapped into a function ^^

    # Initialize for training
    results = []
    predictions_made = 0
    
    # Skip prediction until the first window size is reached
    train_start_idx = 0 
    while (train_start_idx < len(valid_trading_days) and 
           valid_trading_days[train_start_idx] < start_date + timedelta(days=train_window_size)):
        train_start_idx += 1
    
    current_idx = train_start_idx

    while current_idx < len(valid_trading_days) and predictions_made < max_predictions:
        try:
            current_date = valid_trading_days[current_idx]
            print(f"\nMaking prediction for trading day: {current_date}")
            
            # Get training data (using calendar days for window)
            train_start = current_date - timedelta(days=train_window_size)
            train_features_df, train_labels_df = get_data_for_period(
                db_session, train_start, current_date, ticker
            )
            
            # Get prediction period data
            pred_end_date = current_date + timedelta(days=predict_window_size)
            pred_features_df, pred_labels_df = get_data_for_period(
                db_session, current_date, pred_end_date, ticker
            )
            
            if (train_features_df is None or len(train_features_df) < min_samples_per_class or 
                pred_features_df is None or len(pred_features_df) < 1):
                print("Insufficient data for training or prediction, skipping...")
                current_idx += 1
                continue
            
            # Prepare data
            scaler = StandardScaler()
            train_features = scaler.fit_transform(train_features_df.values)
            train_labels = train_labels_df['label'].values
            pred_features = scaler.transform(pred_features_df.values)
            pred_labels = pred_labels_df['label'].values
            
            # Create datasets and loaders
            train_dataset = MarketDataset(train_features, train_labels)
            pred_dataset = MarketDataset(pred_features, pred_labels)
            
            batch_size = min(32, len(train_dataset))
            train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
            pred_loader = DataLoader(pred_dataset, batch_size=1, shuffle=False)
            
            # Train model
            model = MLPClassifier(input_size=train_features.shape[1]).to(device)
            class_weights = calculate_class_weights(train_labels)
            criterion = nn.CrossEntropyLoss(weight=class_weights.to(device))
            optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
            
            model = train_model(model, train_loader, train_loader, criterion, optimizer, device,
                              num_epochs=max_epochs, patience=early_stopping_patience)
            
            # Make prediction
            model.eval()
            with torch.no_grad():
                pred_features_tensor = torch.FloatTensor(pred_features[0]).unsqueeze(0).to(device)
                outputs = model(pred_features_tensor)
                prediction = torch.argmax(outputs, dim=1).item()
            
            # Store result
            results.append({
                'date': current_date.strftime('%Y-%m-%d'),
                'prediction': prediction,
                'actual_label': pred_labels[0],
                'confidence': F.softmax(outputs, dim=1)[0][prediction].item()
            })
            
            predictions_made += 1
            current_idx += 1
            
        except Exception as e:
            print(f"Error making prediction: {str(e)}")
            traceback.print_exc()
            current_idx += 1
    
    # Calculate overall accuracy
    if results:
        predictions = [r['prediction'] for r in results]
        actuals = [r['actual_label'] for r in results]
        accuracy = accuracy_score(actuals, predictions)
        conf_matrix = confusion_matrix(actuals, predictions)
        
        print("\nFinal Results:")
        print(f"Total predictions: {len(results)}")
        print(f"Overall accuracy: {accuracy:.4f}")
        print("\nConfusion Matrix:")
        print(conf_matrix)
    
    return results


def staggered_training(session, param: StaggeredTrainingParam):
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
                .order_by(MarketData.report_date)
                .offset(offset)
                .limit(count)
            )
            query_result = session.execute(query).all()
            # print(f"[DEBUG] query_result: {repr(query_result)}")
            feature_df = process_market_data([(record[0], record[1]) for record in query_result])
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
    
    def get_available_data_count(session, ticker):
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
    prediction_result_list = []

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
        with torch.no_grad():
            for features, labels in pred_loader:
                features = features.to(device)
                outputs = model(features)
                prediction = torch.argmax(outputs, dim=1).item()
                predictions.append(prediction)
                actual_labels.append(labels.item())
                print(f"Prediction: {prediction}, Actual Label: {labels.item()}")

        prediction_result = {
            'predictions': predictions,
            'actual_labels': actual_labels,
            'training_offset': training_offset,
            'prediction_offset': prediction_offset
        }
        prediction_result_list.append(prediction_result)

        training_offset+=param.prediction_day_count
        prediction_offset+=param.prediction_day_count
        available_data_count-=param.prediction_day_count
    
    return prediction_result_list