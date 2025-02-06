import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from sqlalchemy import select, func
from sklearn.preprocessing import StandardScaler, QuantileTransformer
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.utils.class_weight import compute_class_weight
from data_models.MarketData import MarketData
from data_models.EquityIndicators import EquityIndicators
from data_models.SupervisedClassifierDataset import SupClassifierDataset
import scipy.stats as stats
import traceback
from collections import defaultdict

def remove_outliers(df):
    Q1 = df.quantile(0.25)
    Q3 = df.quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    return df[~((df < lower_bound) | (df > upper_bound))]

def process_equity_indicators(raw_data) -> pd.DataFrame:
    """Process technical indicators using existing preprocessing"""
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
    
    # Preprocessed features
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

def process_raw_market_data(market_data_records, lookback_days=4):
    """
    Process raw market data into a DataFrame with lookback days.
    
    Args:
        market_data_records: List of MarketData records
        lookback_days: Number of previous days to include
    
    Returns:
        DataFrame with features formatted as: 
        [open_t, close_t, low_t, high_t, volume_t, 
         open_t-1, close_t-1, low_t-1, high_t-1, volume_t-1, ...]
    """
    # Convert records to DataFrame
    df = pd.DataFrame([{
        'report_date': record.report_date,
        'open': record.open,
        'close': record.close,
        'low': record.low,
        'high': record.high,
        'volume': record.volume
    } for record in market_data_records])
    
    # Sort by date
    df = df.sort_values('report_date')
    feature_columns = []
    for i in range(lookback_days):
        day_suffix = f"_t-{i}" if i > 0 else "_t"
        for col in ['open', 'close', 'low', 'high', 'volume']:
            df[f'{col}{day_suffix}'] = df[col].shift(i)
            feature_columns.append(f'{col}{day_suffix}')
    
    return df[feature_columns].dropna()
    

def process_labels(raw_labels) -> pd.DataFrame:
    """Process raw labels into labels DataFrame"""
    labels_data = []
    dates = []
    
    for item in raw_labels:
        labels_data.append({'label': item.label})
        dates.append(item.start_date)
    
    df = pd.DataFrame(labels_data, index=dates)
    return df