from data_models.ClassifierResult import ClassifierResult
from sqlalchemy import select
from data_models.MarketData import MarketData
from data_models.EquityIndicators import EquityIndicators
from data_models.SupervisedClassifierDataset import SupClassifierDataset
import pandas as pd

    
def upload_classifier_result_batch(classifier_result_list, ticker, db_session):
    try:
        with db_session() as session:
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

            deleted_count = session.query(ClassifierResult)\
                                 .filter(ClassifierResult.ticker == ticker,
                                     ClassifierResult.model == classifier_result_list[0]['model'],
                                     ClassifierResult.feature_set == classifier_result_list[0]['feature_set'])\
                                 .delete()
            print(f"Deleted {deleted_count} existing records for ticker {ticker}, model {classifier_result_list[0]['model']} and feature set {classifier_result_list[0]['feature_set']}")

            all_predictions = []
            total_correct = 0
            total_predictions = 0
            
            for window_result in classifier_result_list:
                predictions = window_result['predictions']
                probabilities = window_result['probabilities']
                pred_offset = window_result['prediction_offset']
                
                window_dates = dates_and_labels[pred_offset:pred_offset + len(predictions)]
                
                for (pred_date, actual_label), pred_value, probs in zip(window_dates, predictions, probabilities):
                    record = ClassifierResult(
                        report_date=pred_date,
                        ticker=ticker,
                        model=window_result['model'],
                        feature_set=window_result['feature_set'],
                        downtrend_prob=float(f"{probs[0]:.4f}"),   
                        side_prob=float(f"{probs[1]:.4f}"),  
                        uptrend_prob=float(f"{probs[2]:.4f}"),
                        predicted_label=int(pred_value),
                        actual_label=int(actual_label)
                    )
                    all_predictions.append(record)
                    
                    # Update accuracy counters
                    if int(pred_value) == int(actual_label):
                        total_correct += 1
                    total_predictions += 1

            session.bulk_save_objects(all_predictions)
            session.commit()
            
            accuracy = (total_correct / total_predictions * 100) if total_predictions > 0 else 0
            print(f"Successfully uploaded {len(all_predictions)} predictions across {len(classifier_result_list)} windows")
            print(f"Total Accuracy: {accuracy:.2f}% ({total_correct}/{total_predictions} correct predictions)")
            return True
            
    except Exception as e:
        print(f"Error uploading predictions: {str(e)}")
        session.rollback()
        return False

def get_classifier_result(db_session, ticker=None, model=None, feature_set=None, start_date=None, end_date=None):
    try:
        with db_session() as session:
            query = select(ClassifierResult)
            
            if ticker:
                query = query.where(ClassifierResult.ticker == ticker)
            if model:
                query = query.where(ClassifierResult.model == model)
            if feature_set:
                query = query.where(ClassifierResult.feature_set == feature_set)
            if start_date:
                query = query.where(ClassifierResult.report_date >= start_date)
            if end_date:
                query = query.where(ClassifierResult.report_date <= end_date)
                
            results = session.execute(query).scalars().all()
            
            return pd.DataFrame([{
                'report_date': r.report_date,
                'ticker': r.ticker,
                'model': r.model,
                'feature_set': r.feature_set,
                'uptrend_prob': r.uptrend_prob,
                'side_prob': r.side_prob,
                'downtrend_prob': r.downtrend_prob,
                'predicted_label': r.predicted_label,
                'actual_label': r.actual_label
            } for r in results])
            
    except Exception as e:
        print(f"[ERROR] Error retrieving prediction results: {str(e)}")
        raise