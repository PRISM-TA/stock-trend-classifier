from sqlalchemy import Column, Integer, String, Date, DateTime, func
from sqlalchemy.ext.declarative import declarative_base

Base = declarative_base()

class PredictionResults(Base):
    __tablename__ = 'prediction_results'
    __table_args__ = {'schema': 'fyp'}

    ticker = Column(String, primary_key=True)
    prediction_date = Column(Date, nullable=False)
    actual_trend = Column(Integer, nullable=False)
    predicted_trend = Column(Integer, nullable=False)
    # created_at = Column(DateTime, server_default=func.now())
    
def __repr__(self):
        return f"<PredictionResults(date={self.prediction_date}, ticker={self.ticker})>"