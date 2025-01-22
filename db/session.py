from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from contextlib import contextmanager
from typing import ContextManager
from sqlalchemy.orm import Session
from data.EquityIndicators import EquityIndicators
from data.MarketData import MarketData
from data.SupervisedClassifierDataset import SupClassifierDataset

@contextmanager
def create_db_session(
    user: str,
    password: str,
    host: str,
    port: str = "5432",
    database: str = "postgres",
    **kwargs
) -> ContextManager[Session]:
    """
    Create and return a database session context manager.
    
    Args:
        user: Database username
        password: Database password
        host: Database host
        port: Database port (default: "5432")
        database: Database name (default: "postgres")
        **kwargs: Additional arguments for create_engine
        
    Returns:
        Context manager that yields database session
    """
    # If no credentials provided, use default Supabase credentials
    if not all([user, password, host]):
        user = "postgres.rcwgtecxhtgblwkmlbpu"
        password = "CSCI4998_TJ01"
        host = "aws-0-ap-southeast-1.pooler.supabase.com"
        port = "5432"
        database = "postgres"
    
    # Create database URL
    database_url = f"postgresql://{user}:{password}@{host}:{port}/{database}"
    
    # Create SQLAlchemy engine and session
    engine = create_engine(database_url, **kwargs)
    SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
    
    try:
        db = SessionLocal()
        yield db
    finally:
        db.close()