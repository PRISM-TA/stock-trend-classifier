from models.mlp_model import staggered_training
from data.StaggeredTrainingParam import StaggeredTrainingParam
from db.session import create_db_session

import os
from dotenv import load_dotenv
load_dotenv()

def main():
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

    staggered_training(db_session, param)

if __name__ == "__main__":
    main()