from lib.staggered_training import staggered_training
from lib.classifier_result_handler import upload_classifier_result_batch

from models.StaggeredTrainingParam import StaggeredTrainingParam
from models.BaseHyperParam import BaseHyperParam
from classifiers.factory.ClassifierFactory import ClassifierFactory
from classifiers.BaseClassifier import BaseClassifier
from classifiers.MLPClassifier import MLPClassifier_V0

from features.BaseFeatureSet import BaseFeatureSet, RMD20DRTI20D, RMD20DRTI, PTI, RTI, RTI20D, RMD20D

from db.session import create_db_session

from dotenv import load_dotenv
import os

############# Trial Setting ###############
ticker_list = ["AAPL", "UNH"]
model_list = [MLPClassifier_V0]
feature_list = [RMD20DRTI20D, RMD20DRTI]
upload_result = False # Whether to upload results to database
###########################################

def run_single_trial(db_session, ticker: str, model: BaseClassifier, feature_set: BaseFeatureSet, save_result: bool = False)->None:
    """Run a single trial for a given ticker, model, and feature set"""
    param = StaggeredTrainingParam(
        training_day_count=240,
        prediction_day_count=60,
        ticker=ticker
    )
    model_param = BaseHyperParam(
        num_epochs=1000,
        early_stopping=True,
        patience=50
    )
    classifier_factory = ClassifierFactory(model)
    feature_set = feature_set()

    print("[DEBUG] Starting training and prediction process...")
    trial_result = staggered_training(
        session=db_session,
        param=param,
        classifier_factory=classifier_factory,
        model_param=model_param,
        feature_set=feature_set
    )
    if not trial_result:
        print("[ERROR] No prediction results generated")
        return
        
    print(f"[DEBUG] Training complete. Got results for {len(trial_result)} windows")
    
    if save_result:
        # Upload all classifier results at once
        success = upload_classifier_result_batch(
            classifier_result_list=trial_result,
            ticker=param.ticker,
            db_session=db_session
        )
        
        if success:
            print("[DEBUG] Successfully uploaded all results to database")
        else:
            print("[ERROR] Failed to upload results to database")
            raise Exception("Failed to upload results to database")
    return

if __name__ == "__main__":
    load_dotenv()
    # Create database session
    db_session = create_db_session(
        user=os.getenv("DB_USER"),
        password=os.getenv("DB_PASSWORD"),
        host=os.getenv("DB_HOST"),
        database=os.getenv("DB_NAME")
    )
    print("[DEBUG] Starting trials ...")
    print("[INFO] Trials will run for tickers:", ticker_list)
    print("[INFO] Trials will run for models:", model_list)
    print("[INFO] Trials will run for features:", feature_list)
    print("====================================================================")
    print("[INFO] Total trials:", len(ticker_list) * len(model_list) * len(feature_list))
    for ticker in ticker_list:
        for classifier_model in model_list:
            for feature_set in feature_list:
                print(f"[DEBUG] Running trial for {ticker}, {classifier_model}, {feature_set}")
                run_single_trial(db_session, ticker, classifier_model, feature_set, upload_result)
    print("[DEBUG] Processing complete.")