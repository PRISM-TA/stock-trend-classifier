import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from sqlalchemy import select, func
from sklearn.preprocessing import StandardScaler, QuantileTransformer
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.utils.class_weight import compute_class_weight
# from data.MarketData import MarketData
# from data.EquityIndicators import EquityIndicators
# from data.SupervisedClassifierDataset import SupClassifierDataset
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as stats

data = pd.read_csv("~/download/classifier-data-labeller-main/utils/equity_indicators_202501192116.csv") # AAPL only
# data.describe()

# Load the dataset
# Replace 'your_dataset.csv' with the path to your dataset file
df = data.copy()

# # Display basic information about the dataset
# print(df.info())
# print(df.describe())

# # Plot the distribution for all features
# for column in df.columns:
#     plt.figure(figsize=(10, 6))
#     sns.histplot(df[column], kde=True)
#     plt.title(f'Distribution of {column}')
#     plt.xlabel(column)
#     plt.ylabel('Frequency')
#     plt.show()

### obv
data["obv_sta"] = data["obv"].pct_change()

def remove_outliers(df):
    Q1 = df.quantile(0.25)
    Q3 = df.quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    return df[~((df < lower_bound) | (df > upper_bound))]

data["obv_sta_no_out"] = remove_outliers(data["obv_sta"])

# def volatility_adjustment(data, window=252):
#     # Calculate rolling standard deviation
#     rolling_std = data.rolling(window=window).std()
#     # Adjust the series by dividing by its rolling std
#     adjusted_series = data / rolling_std
#     return adjusted_series

# data["obv_sta_no_out_vol_adj"] = volatility_adjustment(data["obv_sta_no_out"])

### rsi_9
data['rsi_9_centered'] = (data['rsi_9'] - 50) / 50

### rsi_14
# Center around 50 (RSI's midpoint) and scale
data['rsi_14_centered'] = (data['rsi_14'] - 50) / 50  # This will bound it roughly between -1 and 1

### rsi_20
# Center and scale RSI_20
data['rsi_20_centered'] = (data['rsi_20'] - 50) / 50

### sma_20
data["sma_20_sta"] = data["sma_20"].pct_change()
data["sma_20_sta_no_out"] = remove_outliers(data["sma_20_sta"])
# Normalize to [-1, 1] range
data['sma_20_normalized'] = 2 * (data['sma_20_sta_no_out'] - data['sma_20_sta_no_out'].min()) / (data['sma_20_sta_no_out'].max() - data['sma_20_sta_no_out'].min()) - 1

### sma_50
data["sma_50_sta"] = data["sma_50"].pct_change()
data["sma_50_sta_no_out"] = remove_outliers(data["sma_50_sta"])
# Normalize to [-1, 1] range
data['sma_50_normalized'] = 2 * (data['sma_50_sta_no_out'] - data['sma_50_sta_no_out'].min()) / (data['sma_50_sta_no_out'].max() - data['sma_50_sta_no_out'].min()) - 1

### sma_200
data["sma_200_sta"] = data["sma_200"].pct_change()
data["sma_200_sta_no_out"] = remove_outliers(data["sma_200_sta"])
# Normalize to [-1, 1] range
data['sma_200_normalized'] = 2 * (data['sma_200_sta_no_out'] - data['sma_200_sta_no_out'].min()) / (data['sma_200_sta_no_out'].max() - data['sma_200_sta_no_out'].min()) - 1

### ema_20
data["ema_20_sta"] = data["ema_20"].pct_change()
data["ema_20_sta_no_out"] = remove_outliers(data["ema_20_sta"])
# Normalize to [-1, 1] range
data['ema_20_normalized'] = 2 * (data['ema_20_sta_no_out'] - data['ema_20_sta_no_out'].min()) / (data['ema_20_sta_no_out'].max() - data['ema_20_sta_no_out'].min()) - 1

### ema_50
data["ema_50_sta"] = data["ema_50"].pct_change()
data["ema_50_sta_no_out"] = remove_outliers(data["ema_50_sta"])
# Normalize to [-1, 1] range
data['ema_50_normalized'] = 2 * (data['ema_50_sta_no_out'] - data['ema_50_sta_no_out'].min()) / (data['ema_50_sta_no_out'].max() - data['ema_50_sta_no_out'].min()) - 1

### ema 200
data["ema_200_sta"] = data["ema_200"].pct_change()
data["ema_200_sta_no_out"] = remove_outliers(data["ema_200_sta"])
# Normalize to [-1, 1] range
data['ema_200_normalized'] = 2 * (data['ema_200_sta_no_out'] - data['ema_200_sta_no_out'].min()) / (data['ema_200_sta_no_out'].max() - data['ema_200_sta_no_out'].min()) - 1

### macd_12_26_9_line
# First apply log transformation (adding a small constant to handle zeros)
data["macd_line_sta"] = data["macd_12_26_9_line"].pct_change()
data["macd_line_sta_no_out"] = remove_outliers(data["macd_line_sta"])

# Normalize to [-1, 1] range
data['macd_line_normalized'] = 2 * (data['macd_line_sta_no_out'] - data['macd_line_sta_no_out'].min()) / (data['macd_line_sta_no_out'].max() - data['macd_line_sta_no_out'].min()) - 1

### macd_12_26_9_signal
# First apply log transformation (adding a small constant to handle zeros)
data["macd_signal_sta"] = data["macd_12_26_9_signal"].pct_change()
data["macd_signal_sta_no_out"] = remove_outliers(data["macd_signal_sta"])

# Normalize to [-1, 1] range
data['macd_signal_normalized'] = 2 * (data['macd_signal_sta_no_out'] - data['macd_signal_sta_no_out'].min()) / (data['macd_signal_sta_no_out'].max() - data['macd_signal_sta_no_out'].min()) - 1

### macd_12_26_9_histogram
# First apply log transformation (adding a small constant to handle zeros)
data["macd_histogram_sta"] = data["macd_12_26_9_histogram"].pct_change()
data["macd_histogram_sta_no_out"] = remove_outliers(data["macd_histogram_sta"])

# Normalize to [-1, 1] range
data['macd_histogram_normalized'] = 2 * (data['macd_histogram_sta_no_out'] - data['macd_histogram_sta_no_out'].min()) / (data['macd_histogram_sta_no_out'].max() - data['macd_histogram_sta_no_out'].min()) - 1

### rv


