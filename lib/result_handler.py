import os
import pandas as pd
from datetime import datetime
from sqlalchemy import select
from data_models.MarketData import MarketData
from data_models.EquityIndicators import EquityIndicators
from data_models.SupervisedClassifierDataset import SupClassifierDataset

def initialize_results_file(ticker, output_path='results'):
    """Initialize the CSV file with headers"""
    if not os.path.exists(output_path):
        os.makedirs(output_path)
        
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    csv_filename = f"{output_path}/{ticker}_predictions_{timestamp}.csv"
    
    pd.DataFrame(columns=['window_num', 'prediction_date', 'actual_trend', 'predicted_trend']).to_csv(csv_filename, index=False)
    
    return csv_filename

def store_window_results(session_factory, window_num, prediction_results, ticker, csv_filename):
    with session_factory() as session:
        query = (
            select(
                MarketData.report_date,
                SupClassifierDataset.label  
            )
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
            .offset(prediction_results['prediction_offset'])
            .limit(len(prediction_results['predictions']))
        )
        
        results_df = pd.read_sql(query, session.bind)
        
        window_results = []
        for i in range(min(len(prediction_results['predictions']), len(results_df))):
            window_results.append({
                'window_num': window_num,
                'prediction_date': results_df.iloc[i]['report_date'].strftime('%Y-%m-%d'),
                'actual_trend': int(results_df.iloc[i]['label']),  
                'predicted_trend': int(prediction_results['predictions'][i])
            })
        
        window_df = pd.DataFrame(window_results)
        
        file_exists = os.path.isfile(csv_filename)
        window_df.to_csv(csv_filename, mode='a', header=not file_exists, index=False)
        
        print_results_summary(window_df, csv_filename, window_num)
        
        return window_df

def print_results_summary(window_df, csv_filename, window_num):
    window_accuracy = (window_df['actual_trend'] == window_df['predicted_trend']).mean()
    
    try:
        all_results = pd.read_csv(csv_filename)
        total_predictions = len(all_results)
        overall_accuracy = (all_results['actual_trend'] == all_results['predicted_trend']).mean()
        
        print(f"\nWindow {window_num} Results:")
        print(f"Window accuracy: {window_accuracy:.2%}")
        print(f"Overall accuracy: {overall_accuracy:.2%}")
        print(f"Total predictions: {total_predictions}")
        print(f"Predictions in this window: {len(window_df)}")
        
    except Exception as e:
        print(f"Error calculating overall statistics: {str(e)}")
        print(f"\nWindow {window_num}:")
        print(f"Window accuracy: {window_accuracy:.2%}")