import os
from datetime import datetime, timedelta
import torch
import pandas as pd
import traceback
from db.session import create_db_session
from models.mlp_model import rolling_window_training
from sqlalchemy import select, func
from data.MarketData import MarketData
from data.EquityIndicators import EquityIndicators
from data.SupervisedClassifierDataset import SupClassifierDataset
from utils.visualization import create_trend_visualization
from sklearn.metrics import accuracy_score, confusion_matrix

from dotenv import load_dotenv
load_dotenv()

db_session = create_db_session(
    user=os.getenv("DB_USER"),
    password=os.getenv("DB_PASSWORD"),
    host=os.getenv("DB_HOST"),
    database=os.getenv("DB_NAME")
)

def check_data_availability(session, ticker):
    # Check market data
    market_query = select(
        func.min(MarketData.report_date),
        func.max(MarketData.report_date),
        func.count(MarketData.report_date)
    ).where(MarketData.ticker == ticker)
    
    market_result = session.execute(market_query).first()
    print("\nMarket Data:")
    print(f"Date range: {market_result[0]} to {market_result[1]}")
    print(f"Total records: {market_result[2]}")
    
    # Check indicators
    indicators_query = select(
        func.min(EquityIndicators.report_date),
        func.max(EquityIndicators.report_date),
        func.count(EquityIndicators.report_date)
    ).where(EquityIndicators.ticker == ticker)
    
    indicators_result = session.execute(indicators_query).first()
    print("\nIndicators:")
    print(f"Date range: {indicators_result[0]} to {indicators_result[1]}")
    print(f"Total records: {indicators_result[2]}")
    
    # Check labels
    labels_query = select(
        func.min(SupClassifierDataset.start_date),
        func.max(SupClassifierDataset.start_date),
        func.count(SupClassifierDataset.start_date)
    ).where(SupClassifierDataset.ticker == ticker)
    
    labels_result = session.execute(labels_query).first()
    print("\nLabels:")
    print(f"Date range: {labels_result[0]} to {labels_result[1]}")
    print(f"Total records: {labels_result[2]}")
    
    return market_result, indicators_result, labels_result

def save_results(results, save_dir='results'):
    """Save daily prediction results to files"""
    os.makedirs(save_dir, exist_ok=True)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    # Convert results to DataFrame
    results_df = pd.DataFrame(results)
    results_df['date'] = pd.to_datetime(results_df['date'])
    
    # Save predictions CSV
    results_df.to_csv(f'{save_dir}/daily_predictions_{timestamp}.csv', index=False)
    
    # Calculate and save summary statistics
    summary_stats = {
        'total_predictions': len(results),
        'accuracy': (results_df['prediction'] == results_df['actual_label']).mean(),
        'start_date': results_df['date'].min(),
        'end_date': results_df['date'].max(),
        'predictions_by_class': results_df['prediction'].value_counts().to_dict(),
        'actuals_by_class': results_df['actual_label'].value_counts().to_dict(),
    }
    
    # Save detailed results
    with open(f'{save_dir}/detailed_results_{timestamp}.txt', 'w') as f:
        f.write("Summary Statistics:\n")
        f.write(f"Period: {summary_stats['start_date']} to {summary_stats['end_date']}\n")
        f.write(f"Total predictions: {summary_stats['total_predictions']}\n")
        f.write(f"Overall accuracy: {summary_stats['accuracy']:.4f}\n")
        
        f.write("\nPredictions by class:\n")
        for label, count in summary_stats['predictions_by_class'].items():
            f.write(f"Class {label}: {count}\n")
        
        f.write("\nActual labels by class:\n")
        for label, count in summary_stats['actuals_by_class'].items():
            f.write(f"Class {label}: {count}\n")
        
        # Calculate confusion matrix
        conf_matrix = confusion_matrix(
            results_df['actual_label'],
            results_df['prediction']
        )
        f.write("\nConfusion Matrix:\n")
        f.write(str(conf_matrix))

def main():
    # Set up device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Updated parameters for daily predictions
    params = {
        'ticker': 'AAPL',
        'train_window_size': 240,    # Training window (use previous 240 days for training)
        'predict_window_size': 20,   # Prediction window (predict 20-day trend)
        'max_predictions': 60,      # Make 60 daily predictions
        'min_samples_per_class': 2,
        'max_epochs': 100,
        'early_stopping_patience': 50,
        # Add date range parameters
        'start_date': '2010-01-01',  # Start of training period
        # 'start_date': '1997-10-23',
        # 'start_date': None,
        'end_date': None,           # End of testing period (None for latest available)
    }
    
    print("Starting daily predictions with parameters:")
    for key, value in params.items():
        print(f"{key}: {value}")
    
    try:
        with db_session() as session:
            print("\nChecking data availability...")
            market_result, indicators_result, labels_result = check_data_availability(session, params['ticker'])
            
            if all([market_result, indicators_result, labels_result]):
                # Run daily predictions with date range
                results = rolling_window_training(
                    db_session=session,
                    ticker=params['ticker'],
                    start_date=params['start_date'],
                    end_date=params['end_date'],
                    train_window_size=params['train_window_size'],
                    predict_window_size=params['predict_window_size'],
                    max_predictions=params['max_predictions'],
                    min_samples_per_class=params['min_samples_per_class'],
                    max_epochs=params['max_epochs'],
                    early_stopping_patience=params['early_stopping_patience']
                )
                
                # Save results
                save_results(results)

                
                # Print summary with more details
                print("\nPredictions completed!")
                print(f"Total predictions: {len(results)}")
                print(f"Training window: {params['train_window_size']} days")
                print(f"Prediction window: {params['predict_window_size']} days")
                print(f"Date range: {params['start_date']} to {params['end_date'] or 'latest'}")
                
                accuracy = sum(r['prediction'] == r['actual_label'] for r in results) / len(results)
                print(f"Overall accuracy: {accuracy:.4f}")
                
                # Print prediction distribution
                predictions = pd.DataFrame(results)
                print("\nPrediction distribution:")
                print(predictions['prediction'].value_counts())
                
                # Create visualization directories
                visualization_dir = 'visualizations'
                os.makedirs(visualization_dir, exist_ok=True)
                
                # Define time windows for visualization
                time_windows = [{
                    'name': 'full',
                    'start': params['start_date'],
                    'end': params['end_date']
                }]
                
                # Create visualizations
                for window in time_windows:
                    print(f"\nCreating visualization for {window['name']} period...")
                    try:
                        time_window = (window['start'], window['end'])
                        create_trend_visualization(
                            db_session=session,
                            ticker=params['ticker'],
                            results=results,
                            save_dir='visualizations',
                            time_window=time_window, 
                            overall_accuracy=accuracy
                        )
                        print(f"Visualization saved for {window['name']} period")
                        
                    except Exception as viz_error:
                        print(f"Error creating visualization for {window['name']}: {viz_error}")
                        traceback.print_exc()
                
            else:
                print("Error: Missing data in one or more tables")
            
    except Exception as e:
        print(f"Error during prediction: {e}")
        traceback.print_exc()
        raise e

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nPredictions interrupted by user")
    except Exception as e:
        print(f"An error occurred: {e}")
        traceback.print_exc()
    finally:
        print("\nExiting program")