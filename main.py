from lib.staggered_training import staggered_training
from lib.classifier_result_handler import upload_classifier_result

from models.StaggeredTrainingParam import StaggeredTrainingParam
from models.BaseHyperParam import BaseHyperParam
from classifiers.factory.ClassifierFactory import ClassifierFactory
from classifiers.BaseClassifier import BaseClassifier
from classifiers.MLPClassifier import MLPClassifier_V0
from classifiers.CNNClassifier import CNNClassifier_V0
from classifiers.LSTMClassifier import LSTMClassifier_V0

from features.BaseFeatureSet import BaseFeatureSet, RMD20DRTI20D, RMD20DRTI, PTI, PTI20D, RTI, RTI20D, RMD20D

from db.session import create_db_session

from dotenv import load_dotenv
import os
import itertools
import multiprocessing as mp

############# Trial Setting ###############
# ticker_list = [ "AXP"]
ticker_list = [ "AAPL", "AXP", "BA", "CAT", "CSCO", "CVX", "DD", "DIS", "GE", "HD", "IBM", "INTC", "JNJ", "JPM", "KO", "MCD", "MMM", "MRK", "MSFT", "NKE", "PFE", "PG", "TRV", "UNH", "UTX", "VZ", "WMT", "XOM"]
model_list = [CNNClassifier_V0]
feature_list = [PTI20D]
upload_result = True  # Whether to upload results to database
num_processes = 3     # Adjust based on GPU memory capacity
###########################################

def run_single_trial(ticker: str, model: BaseClassifier, feature_set_class: BaseFeatureSet, save_result: bool = False) -> None:
    """ 
    Run a single trial for a given ticker, model, and feature set.
    """
    # Create database session context manager
    db_session_context = create_db_session(
        user=os.getenv("DB_USER"),
        password=os.getenv("DB_PASSWORD"),
        host=os.getenv("DB_HOST"),
        database=os.getenv("DB_NAME")
    )
    
    # Use the context manager to handle the session
    with db_session_context() as db_session:
        param = StaggeredTrainingParam(
            training_day_count=240,
            prediction_day_count=60,
            ticker=ticker
        )
        model_param = BaseHyperParam(
            num_epochs=1200,
            early_stopping=True,
            patience=50
        )
        classifier_factory = ClassifierFactory(model)
        feature_set = feature_set_class()  # Instantiate the feature set here

        print(f"[INFO] Starting trial for {ticker} with {model.__name__} on {feature_set_class.__name__}")
        trial_result = staggered_training(
            session=db_session,
            param=param,
            classifier_factory=classifier_factory,
            model_param=model_param,
            feature_set=feature_set
        )
        
        if not trial_result:
            print(f"[ERROR] No prediction results generated for {ticker} with {model.__name__} on {feature_set_class.__name__}")
            return
        
        print(f"[INFO] Completed training for {ticker} with {model.__name__} on {feature_set_class.__name__}.")
        
        if save_result:
            success = upload_classifier_result(
                session=db_session,
                classifier_result_list=trial_result
            )
            
            if success:
                print(f"[INFO] Uploaded results for {ticker} with {model.__name__} on {feature_set_class.__name__}")
            else:
                print(f"[ERROR] Failed to upload results for {ticker} with {model.__name__} on {feature_set_class.__name__}")
                raise Exception("Failed to upload results to database")

if __name__ == "__main__":
    load_dotenv()
    mp.set_start_method('spawn')  # Resolve CUDA compatibility issues
    
    # Generate all parameter combinations
    params = list(itertools.product(ticker_list, model_list, feature_list, [upload_result]))
    
    print(f"[INFO] Starting parallel trials with {num_processes} processes...")
    with mp.Pool(processes=num_processes) as pool:
        pool.starmap(run_single_trial, params)
    
    print("[INFO] All trials completed.")