from data_models.PredictionResults import PredictionResults
from sqlalchemy import select
from data_models.MarketData import MarketData
from data_models.EquityIndicators import EquityIndicators
from data_models.SupervisedClassifierDataset import SupClassifierDataset
import pandas as pd

# def upload_prediction_results(prediction_data, window_num, ticker, db_session):
#     """
#     Upload prediction results to database.
    
#     Args:
#         prediction_data: Dictionary containing predictions and metadata
#         window_num: Window number for these predictions
#         ticker: Stock ticker symbol
#         db_session: Database session context manager
#     """
#     print(f"\n[DEBUG] Storing results for window {window_num}")
#     print(f"[DEBUG] Ticker: {ticker}")
    
#     try:
#         with db_session() as session:
#             # Get dates and actual labels
#             query = (
#                 select(
#                     MarketData.report_date,
#                     SupClassifierDataset.label
#                 )
#                 .join(
#                     EquityIndicators,
#                     (MarketData.ticker == EquityIndicators.ticker) &
#                     (MarketData.report_date == EquityIndicators.report_date)
#                 ).join(
#                     SupClassifierDataset,
#                     (MarketData.ticker == SupClassifierDataset.ticker) &
#                     (MarketData.report_date == SupClassifierDataset.end_date)
#                 )
#                 .where(MarketData.ticker == ticker)
#                 .order_by(MarketData.report_date)
#                 .offset(prediction_data['prediction_offset'])
#                 .limit(len(prediction_data['predictions']))
#             )
            
#             results = session.execute(query).all()
#             print(f"[DEBUG] Retrieved {len(results)} records from database")

#             # Delete existing records for this ticker
#             session.query(PredictionResults)\
#                    .filter(
#                        PredictionResults.ticker == ticker
#                    ).delete()

#             # Create new records
#             new_records = []
#             for (pred_date, actual_label), pred_value in zip(results, prediction_data['predictions']):
#                 record = PredictionResults(
#                     prediction_date=pred_date,
#                     ticker=ticker,
#                     predicted_trend=int(pred_value),
#                     actual_trend=int(actual_label)
#                 )
#                 new_records.append(record)

#             # Bulk insert new records
#             session.bulk_save_objects(new_records)
            
#             # # Calculate window statistics
#             # window_accuracy = sum(
#             #     1 for r in new_records 
#             #     if r.actual_trend == r.predicted_trend
#             # ) / len(new_records) if new_records else 0

#             # Commit changes
#             session.commit()

#             # print(f"\nWindow {window_num} Results:")
#             # print(f"Window accuracy: {window_accuracy:.2%}")
#             # print(f"Predictions in this window: {len(new_records)}")
            
#             return pd.DataFrame([{
#                 # 'window_num': r.window_num,
#                 'prediction_date': r.prediction_date,
#                 'actual_trend': r.actual_trend,
#                 'predicted_trend': r.predicted_trend
#             } for r in new_records])
            
#     except Exception as e:
#         print(f"[ERROR] Error storing results for window {window_num}: {str(e)}")
#         session.rollback()
#         raise
    
def upload_prediction_results_batch(prediction_results_list, ticker, db_session):
    """
    Upload batch prediction results to database.
    
    Args:
        prediction_results_list: List of dictionaries containing predictions and metadata
        ticker: Stock ticker symbol
        db_session: Database session context manager
    """
    try:
        with db_session() as session:
            # Get dates and actual labels for predictions
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
            )
            
            dates_and_labels = session.execute(query).all()
            print(f"Retrieved {len(dates_and_labels)} dates from database")

            # Delete existing records for this ticker
            deleted_count = session.query(PredictionResults)\
                                 .filter(PredictionResults.ticker == ticker)\
                                 .delete()
            print(f"Deleted {deleted_count} existing records for ticker {ticker}")

            # Create new records
            all_predictions = []
            current_offset = 0
            
            for window_result in prediction_results_list:
                predictions = window_result['predictions']
                pred_offset = window_result['prediction_offset']
                
                # Get the relevant dates for this window
                window_dates = dates_and_labels[pred_offset:pred_offset + len(predictions)]
                
                for (pred_date, actual_label), pred_value in zip(window_dates, predictions):
                    record = PredictionResults(
                        prediction_date=pred_date,
                        ticker=ticker,
                        predicted_trend=int(pred_value),
                        actual_trend=int(actual_label)
                    )
                    all_predictions.append(record)

            # Bulk insert new records
            session.bulk_save_objects(all_predictions)
            session.commit()
            
            print(f"Successfully uploaded {len(all_predictions)} predictions across {len(prediction_results_list)} windows")
            return True
            
    except Exception as e:
        print(f"Error uploading predictions: {str(e)}")
        session.rollback()
        return False

def get_prediction_results(db_session, ticker=None, start_date=None, end_date=None):
    try:
        with db_session() as session:
            query = select(PredictionResults)
            
            if ticker:
                query = query.where(PredictionResults.ticker == ticker)
            if start_date:
                query = query.where(PredictionResults.prediction_date >= start_date)
            if end_date:
                query = query.where(PredictionResults.prediction_date <= end_date)
                
            results = session.execute(query).scalars().all()
            
            return pd.DataFrame([{
                # 'window_num': r.window_num,
                'prediction_date': r.prediction_date,
                'ticker': r.ticker,
                'predicted_trend': r.predicted_trend,
                'actual_trend': r.actual_trend
            } for r in results])
            
    except Exception as e:
        print(f"[ERROR] Error retrieving prediction results: {str(e)}")
        raise