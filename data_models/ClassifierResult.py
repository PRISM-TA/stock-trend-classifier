from sqlalchemy import Column, Integer, String, Date, Float
from sqlalchemy.ext.declarative import declarative_base

Base = declarative_base()

class ClassifierResult(Base):
    __tablename__ = 'classifier_result'
    __table_args__ = {'schema': 'fyp'}

    report_date = Column(Date, nullable=False)
    
    ### Primary keys
    ticker = Column(String, primary_key=True)
    model = Column(String, primary_key=True)
    feature_set = Column(String, primary_key=True)
    
    uptrend_prob = Column(Float(4))
    side_prob = Column(Float(4))
    downtrend_prob = Column(Float(4))
    predicted_label = Column(Integer, nullable=False)
    actual_label = Column(Integer, nullable=False)
    
def __repr__(self):
        return f"<ClassifierResult(date={self.report_date}, ticker={self.ticker}, model={self.model}, feature_set={self.feature_set})>"