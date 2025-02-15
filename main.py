from lib.staggered_training import staggered_training
from lib.classifier_result_handler import upload_classifier_result_batch

from models.StaggeredTrainingParam import StaggeredTrainingParam
from models.BaseHyperParam import BaseHyperParam
from classifiers.factory.ClassifierFactory import ClassifierFactory
from classifiers.MLPClassifier import MLPClassifier_V0

from features.BaseFeatureSet import RMD20DRTI20D

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
        ticker='HD'
    )
    model_param = BaseHyperParam(
        num_epochs=1000,
        early_stopping=True,
        patience=50
    )
    classifier_factory = ClassifierFactory(MLPClassifier_V0)

    # feature_set = "Processed technical indicators"
    feature_set = RMD20DRTI20D()

    print("[DEBUG] Starting training and prediction process...")
    # Get all classifier results
    classifier_result = staggered_training(
        db_session,
        param,
        classifier_factory,
        model_param,
        feature_set
    )
    
    if not classifier_result:
        print("[ERROR] No prediction results generated")
        return
        
    print(f"[DEBUG] Training complete. Got results for {len(classifier_result)} windows")
    
    # Upload all classifier results at once
    success = upload_classifier_result_batch(
        classifier_result_list=classifier_result,
        ticker=param.ticker,
        db_session=db_session
    )
    
    if success:
        print("[DEBUG] Successfully uploaded all results to database")
    else:
        print("[ERROR] Failed to upload results to database")

    print("[DEBUG] Processing complete")

if __name__ == "__main__":
    main()