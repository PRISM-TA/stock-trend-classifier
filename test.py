from lib.mlp_model import staggered_training
from data_models.StaggeredTrainingParam import StaggeredTrainingParam
from db.session import create_db_session
from lib.result_handler import initialize_results_file, store_window_results
import os
import pandas as pd
from dotenv import load_dotenv

def main():
    load_dotenv()
    
    db_session = create_db_session(
        user=os.getenv("DB_USER"),
        password=os.getenv("DB_PASSWORD"),
        host=os.getenv("DB_HOST"),
        database=os.getenv("DB_NAME")
    )

    param = StaggeredTrainingParam(
        training_day_count=240,
        prediction_day_count=60,
        ticker='AAPL'
    )

    csv_filename = initialize_results_file(param.ticker)
    print(f"Results will be saved to: {csv_filename}")
    print("\nStarting training and prediction process...")
    
    for window_num, window_results in enumerate(staggered_training(db_session, param), 1):
        store_window_results(
            session_factory=db_session,
            window_num=window_num,
            prediction_results=window_results,
            ticker=param.ticker,
            csv_filename=csv_filename
        )

    final_results = pd.read_csv(csv_filename)
    final_accuracy = (final_results['actual_trend'] == final_results['predicted_trend']).mean()
    print(f"\nTraining completed!")
    print(f"Total predictions: {len(final_results)}")
    print(f"Final accuracy: {final_accuracy:.2%}")

if __name__ == "__main__":
    main()
