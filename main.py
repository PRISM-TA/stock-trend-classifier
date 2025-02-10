from lib.mlp_model import staggered_training
from lib.classifier_result_handler import upload_classifier_result_batch
from models.StaggeredTrainingParam import StaggeredTrainingParam
from db.session import create_db_session
from dotenv import load_dotenv
import os

def main():
    load_dotenv()
    
    # Create database session
    db_session = create_db_session(
        user=os.getenv("DB_USER"),
        password=os.getenv("DB_PASSWORD"),
        host=os.getenv("DB_HOST"),
        database=os.getenv("DB_NAME")
    )

    param = StaggeredTrainingParam(
        training_day_count=240,
        prediction_day_count=60,
        ticker='MMM'
    )
    
    model_name = "MLP"
    # feature_set = "Processed technical indicators"
    feature_set = "Processed technical indicators"

    print("\nStarting training and prediction process...")
    # Get all classifier results
    classifier_result = staggered_training(db_session, param, model_name=model_name,
    feature_set=feature_set)
    
    if not classifier_result:
        print("No prediction results generated")
        return
        
    print(f"\nTraining complete. Got results for {len(classifier_result)} windows")
    
    # Upload all classifier results at once
    success = upload_classifier_result_batch(
        classifier_result_list=classifier_result,
        ticker=param.ticker,
        db_session=db_session
    )
    
    if success:
        print("\nSuccessfully uploaded all results to database")
    else:
        print("\nFailed to upload results to database")

    print("\nProcessing complete")

if __name__ == "__main__":
    main()