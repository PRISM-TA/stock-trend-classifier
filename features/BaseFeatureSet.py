from models.EquityIndicators import EquityIndicators
from models.MarketData import MarketData
from models.SupervisedClassifierDataset import SupClassifierDataset

from lib.data_preprocessing import process_equity_indicators, process_20_day_equity_indicators, process_raw_equity_indicators, process_labels, process_20_day_raw_equity_indicators, process_raw_market_data
from sqlalchemy import select
import pandas as pd

class BaseFeatureSet:
    set_name: str
    def get_data(self, session, offset: int, count: int, ticker: str):
        raise NotImplementedError

class PTI(BaseFeatureSet):
    set_name: str = "processed technical indicators"
    def get_data(self, session, offset: int, count: int, ticker: str):
        with session as session:
            ### Processed technical indicators
            query = (
                select(MarketData, EquityIndicators, SupClassifierDataset)
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
            ).order_by(MarketData.report_date).offset(offset).limit(count)

            query_result = session.execute(query).all()

            # Processed technical indicators
            feature_df = process_equity_indicators([(record[0], record[1]) for record in query_result])          
            labels_df = process_labels([(record[2]) for record in query_result])
            
            return feature_df, labels_df
        
class PTI20D(BaseFeatureSet):
    set_name: str = "processed technical indicators (20 days)"
    def get_data(self, session, offset: int, count: int, ticker: str):
        with session as session:
            ### Processed technical indicators (20 days)
            query = (
                select(MarketData, EquityIndicators, SupClassifierDataset)
                .join(
                    EquityIndicators,
                    (MarketData.ticker == EquityIndicators.ticker) &
                    (MarketData.report_date == EquityIndicators.report_date)
                ).join(
                    SupClassifierDataset,
                    (MarketData.ticker == SupClassifierDataset.ticker) &
                    (MarketData.report_date == SupClassifierDataset.start_date)
                )
                .where(MarketData.ticker == ticker)
            ).order_by(MarketData.report_date).offset(offset).limit(count)

            query_result = session.execute(query).all()

            # Processed technical indicators (20 days)
            feature_df = process_20_day_equity_indicators([(record[0], record[1]) for record in query_result], lookback_days=20)      
            labels_df = process_labels([(record[2]) for record in query_result])
            
            return feature_df, labels_df
        
class RTI(BaseFeatureSet):
    set_name: str = "raw technical indicators"
    def get_data(self, session, offset: int, count: int, ticker: str):
        with session as session:
            ### Raw technical indicators
            query = (
                select(MarketData, EquityIndicators, SupClassifierDataset)
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
            ).order_by(MarketData.report_date).offset(offset).limit(count)

            query_result = session.execute(query).all()

            # Raw technical indicators
            feature_df = process_raw_equity_indicators([(record[0], record[1]) for record in query_result])    
            labels_df = process_labels([(record[2]) for record in query_result])
            
            return feature_df, labels_df

class RTI20D(BaseFeatureSet):
    set_name: str = "raw technical indicators (20 days)"
    def get_data(self, session, offset: int, count: int, ticker: str):
        with session as session:
            ### Raw technical indicators (20 days)
            query = (
                select(MarketData, EquityIndicators, SupClassifierDataset)
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
            ).order_by(MarketData.report_date).offset(offset).limit(count)

            query_result = session.execute(query).all()

            # Raw technical indicators (20 days)
            feature_df = process_20_day_raw_equity_indicators([(record[0], record[1]) for record in query_result], lookback_days=20)      
            labels_df = process_labels([(record[2]) for record in query_result])
            
            return feature_df, labels_df

class RMD20D(BaseFeatureSet):
    set_name: str = "Raw market data (20 days)"
    def get_data(self, session, offset: int, count: int, ticker: str):
        with session as session:        
            ## Raw market data (20 days):
            query = (
                select(MarketData, SupClassifierDataset)
                .join(
                    SupClassifierDataset,
                    (MarketData.ticker == SupClassifierDataset.ticker) &
                    (MarketData.report_date == SupClassifierDataset.end_date)
                )
                .where(MarketData.ticker == ticker)
            )
            
            query = (query
                    .order_by(MarketData.report_date)
                    .offset(offset)
                    .limit(count))
                    
            query_result = session.execute(query).all()
            market_data = [record[0] for record in query_result]
            labels = [record[1] for record in query_result]

            feature_df = process_raw_market_data(market_data, lookback_days=20)
            labels_df = process_labels(labels)
            
            return feature_df, labels_df
        
class RMD20DRTI(BaseFeatureSet):
    set_name: str = "Raw market data (20 days) + raw technical indicators"
    def get_data(self, session, offset, count, ticker):
        with session as session:        
            ### Combine market data and technical indicators:
            query = (
                select(MarketData, EquityIndicators, SupClassifierDataset)
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
            )
            
            query = (query
                    .order_by(MarketData.report_date)
                    .offset(offset)
                    .limit(count))
                    
            query_result = session.execute(query).all()

            # Process both types of features
            market_data = [record[0] for record in query_result]
            raw_market_feature_df = process_raw_market_data(market_data, lookback_days=20)
            
            ### Raw market data (20 days) + raw technical indicators
            raw_tech_feature_df = process_raw_equity_indicators([(record[0], record[1]) for record in query_result])
            
            labels_df = process_labels([(record[2]) for record in query_result])
            # Get the length of the shortest dataframe
            min_length = min(len(raw_market_feature_df), len(raw_tech_feature_df), len(labels_df))
            # Trim all dataframes to the same length from the end
            raw_market_feature_df = raw_market_feature_df.iloc[-min_length:]
            raw_tech_feature_df = raw_tech_feature_df.iloc[-min_length:]
            labels_df = labels_df.iloc[-min_length:]
            # Reset indexes before concatenating
            raw_market_feature_df.index = range(len(raw_market_feature_df))
            raw_tech_feature_df.index = range(len(raw_tech_feature_df))
            labels_df.index = range(len(labels_df))
            #print("Raw market data columns:", raw_market_feature_df.columns.tolist())
            #print("Technical indicator columns:", raw_tech_feature_df.columns.tolist())
            # Combine features
            feature_df = pd.concat([raw_market_feature_df, raw_tech_feature_df], axis=1)
            # Remove any duplicate columns if they exist
            feature_df = feature_df.loc[:,~feature_df.columns.duplicated()]

            #print("Final feature shape:", feature_df.shape)
            
            return feature_df, labels_df

class RMD20DRTI20D(BaseFeatureSet):
    set_name: str = "Raw market data (20 days) + raw technical indicators (20 days)"
    def get_data(self, session, offset, count, ticker):
        with session as session:        
            ### Combine market data and technical indicators:
            query = (
                select(MarketData, EquityIndicators, SupClassifierDataset)
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
            )
            
            query = (query
                    .order_by(MarketData.report_date)
                    .offset(offset)
                    .limit(count))
                    
            query_result = session.execute(query).all()

            # Process both types of features
            market_data = [record[0] for record in query_result]
            raw_market_feature_df = process_raw_market_data(market_data, lookback_days=20)
            
            ### Raw market data (20 days) + raw technical indicators (20 days)
            raw_tech_feature_df = process_20_day_raw_equity_indicators([(record[0], record[1]) for record in query_result], lookback_days=20)
            
            labels_df = process_labels([(record[2]) for record in query_result])
            # Get the length of the shortest dataframe
            min_length = min(len(raw_market_feature_df), len(raw_tech_feature_df), len(labels_df))
            # Trim all dataframes to the same length from the end
            raw_market_feature_df = raw_market_feature_df.iloc[-min_length:]
            raw_tech_feature_df = raw_tech_feature_df.iloc[-min_length:]
            labels_df = labels_df.iloc[-min_length:]
            # Reset indexes before concatenating
            raw_market_feature_df.index = range(len(raw_market_feature_df))
            raw_tech_feature_df.index = range(len(raw_tech_feature_df))
            labels_df.index = range(len(labels_df))
            #print("Raw market data columns:", raw_market_feature_df.columns.tolist())
            #print("Technical indicator columns:", raw_tech_feature_df.columns.tolist())
            # Combine features
            feature_df = pd.concat([raw_market_feature_df, raw_tech_feature_df], axis=1)
            # Remove any duplicate columns if they exist
            feature_df = feature_df.loc[:,~feature_df.columns.duplicated()]

            #print("Final feature shape:", feature_df.shape)
            
            return feature_df, labels_df