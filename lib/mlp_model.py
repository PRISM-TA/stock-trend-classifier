from models.MarketDataset import MarketDataset
from models.MarketData import MarketData
from models.EquityIndicators import EquityIndicators
from models.SupervisedClassifierDataset import SupClassifierDataset
from models.StaggeredTrainingParam import StaggeredTrainingParam

from classifiers.BaseClassifier import BaseClassifier
from classifiers.MLPClassifier import MLPClassifier
from lib.data_preprocessing import process_labels, process_20_day_raw_equity_indicators, process_raw_market_data

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

import numpy as np
import pandas as pd
from sqlalchemy import select, func
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix


def calculate_class_weights(labels):
    """Calculate balanced class weights dynamically for all present classes"""
    # Convert float labels to integers
    labels_int = labels.astype(int)
    # Get unique classes and their counts
    unique_classes = np.unique(labels_int)
    class_counts = np.bincount(labels_int)

    # Calculate weights for each present class
    class_weights = torch.FloatTensor(len(class_counts))
    for i in range(len(class_counts)):
        if class_counts[i] > 0:
            class_weights[i] = 1.0 / class_counts[i]
        else:
            class_weights[i] = 0.0

    # Normalize weights
    if class_weights.sum() > 0:
        class_weights = class_weights / class_weights.sum()
    else:
        # If all weights are zero, use equal weights
        class_weights = torch.ones(len(class_counts)) / len(class_counts)

    return class_weights

def train_model(model, train_loader, val_loader, criterion, optimizer, device, num_epochs=100, patience=50):
    best_loss = float('inf')
    patience_counter = 0
    
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        
        for batch_features, batch_labels in train_loader:
            batch_features = batch_features.to(device)
            batch_labels = batch_labels.to(device).long().squeeze()
            
            optimizer.zero_grad()
            outputs = model(batch_features)
            
            # Handle single sample case
            if outputs.dim() == 2 and outputs.size(0) == 1:
                outputs = outputs.squeeze(0)
                batch_labels = batch_labels.squeeze(0)
            
            loss = criterion(outputs, batch_labels)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
        
        # Validation
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for val_features, val_labels in val_loader:
                val_features = val_features.to(device)
                val_labels = val_labels.to(device).long().squeeze()
                
                val_outputs = model(val_features)
                val_loss += criterion(val_outputs, val_labels).item()
        
        if epoch % 50 == 0:
            print(f"Epoch {epoch:3d}: Loss = {total_loss/len(train_loader):.4f}")
        
        # Early stopping
        if val_loss < best_loss:
            best_loss = val_loss
            patience_counter = 0
        else:
            patience_counter += 1
            
        if patience_counter >= patience:
            print(f"Early stopping at epoch {epoch}")
            break
            
    return model

def staggered_training(session, param: StaggeredTrainingParam, model_name: str, feature_set: str):
    def get_data(session, offset, count, ticker):
        with session() as session:
            ### Technical indicators:
            # query = (
            #     select(MarketData, EquityIndicators, SupClassifierDataset)
            #     .join(
            #         EquityIndicators,
            #         (MarketData.ticker == EquityIndicators.ticker) &
            #         (MarketData.report_date == EquityIndicators.report_date)
            #     ).join(
            #         SupClassifierDataset,
            #         (MarketData.ticker == SupClassifierDataset.ticker) &
            #         (MarketData.report_date == SupClassifierDataset.end_date)
            #     )
            #     .where(MarketData.ticker == ticker)
            # )
               
            # query = (query
            #         .order_by(MarketData.report_date)
            #         .offset(offset)
            #         .limit(count))
                    
            # query_result = session.execute(query).all()
            
            # # Raw technical indicators
            # # feature_df = process_raw_equity_indicators([(record[0], record[1]) for record in query_result])
            # # Raw technical indicators (20 days)
            # # feature_df = process_20_day_raw_equity_indicators([(record[0], record[1]) for record in query_result], lookback_days=20)
            # # Processed technical indicators
            # # feature_df = process_equity_indicators([(record[0], record[1]) for record in query_result])
            
            # labels_df = process_labels([(record[2]) for record in query_result])
            # # print("Feature columns:", feature_df.columns.tolist())
            
            
            ### Raw market data (20 days):
            # query = (
            #     select(MarketData, SupClassifierDataset)
            #     .join(
            #         SupClassifierDataset,
            #         (MarketData.ticker == SupClassifierDataset.ticker) &
            #         (MarketData.report_date == SupClassifierDataset.end_date)
            #     )
            #     .where(MarketData.ticker == ticker)
            # )
            
            # query = (query
            #         .order_by(MarketData.report_date)
            #         .offset(offset)
            #         .limit(count))
                    
            # query_result = session.execute(query).all()
            # market_data = [record[0] for record in query_result]
            # labels = [record[1] for record in query_result]

            # feature_df = process_raw_market_data(market_data, lookback_days=20)
            # labels_df = process_labels(labels)
            
            
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
            # raw_tech_feature_df = process_raw_equity_indicators([(record[0], record[1]) for record in query_result])
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
        
    def scale_data(feat_train_df, label_train_df, feat_pred_df, label_pred_df):
        scaler = StandardScaler()
        train_features = scaler.fit_transform(feat_train_df.values)
        train_labels = label_train_df['label'].values
        pred_features = scaler.transform(feat_pred_df.values)
        pred_labels = label_pred_df['label'].values
        return train_features, train_labels, pred_features, pred_labels

    def create_dataloader(train_features, train_labels, pred_features, pred_labels, min_batch_size=32):
        train_dataset = MarketDataset(train_features, train_labels)
        pred_dataset = MarketDataset(pred_features, pred_labels)
        batch_size = min(min_batch_size, len(train_dataset))
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False)
        pred_loader = DataLoader(pred_dataset, batch_size=1, shuffle=False)
        return train_loader, pred_loader
    
    def get_available_data_count(session, ticker, start_date=None):
        with session() as session:
            query = (
                select(
                    func.count(MarketData.report_date)
                ).where(MarketData.ticker == ticker)
            )
            
                
            query_result = session.execute(query).all()
            print(f"[DEBUG] available_data_count for {ticker}: {query_result[0][0]}")
            return query_result[0][0]
    
    max_epochs = 1000
    classifier_result_list = []

    available_data_count = get_available_data_count(session, param.ticker) - param.training_day_count
    if available_data_count < 0:
        print(f"[DEBUG] Not enough training data available for {param.ticker}")
    training_offset = 0
    prediction_offset = param.training_day_count

    while available_data_count > param.prediction_day_count:
        print(f"[DEBUG] Available data count: {available_data_count}")
        print(f"[DEBUG] Training on {param.ticker} from {training_offset} to {prediction_offset}")
        print(f"[DEBUG] Testing on {param.ticker} from {prediction_offset} to {prediction_offset + param.prediction_day_count}")

        feat_train_df, label_train_df = get_data(session, training_offset, param.training_day_count, param.ticker)
        feat_pred_df, label_pred_df = get_data(session, prediction_offset, param.prediction_day_count, param.ticker)
        train_features, train_labels, pred_features, pred_labels = scale_data(feat_train_df, label_train_df, feat_pred_df, label_pred_df)
        train_loader, pred_loader = create_dataloader(train_features, train_labels, pred_features, pred_labels)

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = MLPClassifier(input_size=train_features.shape[1]).to(device)
        class_weights = calculate_class_weights(train_labels)
        criterion = nn.CrossEntropyLoss(weight=class_weights.to(device))
        # criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.SGD(model.parameters(), lr=0.001)
        model = train_model(model, train_loader, train_loader, criterion, optimizer, device, num_epochs=max_epochs)
        # model.train_classifier(
        #     train_loader=train_loader,  
        #     criterion=criterion, 
        #     optimizer=optimizer, 
        #     num_epochs=max_epochs,
        #     early_stopping=True,
        #     val_loader=train_loader,
        #     patience=50
        # )

        model.eval()
        predictions = []
        actual_labels = []
        probabilities = []
        with torch.no_grad():
            for features, labels in pred_loader:
                features = features.to(device)
                outputs = model(features)
                # Get probabilities using softmax
                probs = torch.softmax(outputs, dim=1).cpu().numpy()[0]
                prediction = torch.argmax(outputs, dim=1).item()
                predictions.append(prediction)
                actual_labels.append(labels.item())
                probabilities.append(probs)
                print(f"Prediction: {prediction}, Actual Label: {labels.item()}, Probabilities: [{float(probs[0]):.4f}, {float(probs[1]):.4f}, {float(probs[2]):.4f}]")

        classifier_result = {
            'predictions': predictions,
            'actual_labels': actual_labels,
            'training_offset': training_offset,
            'prediction_offset': prediction_offset,
            'probabilities': probabilities, 
            'model': model_name,
            'feature_set': feature_set
        }
        classifier_result_list.append(classifier_result)

        training_offset+=param.prediction_day_count
        prediction_offset+=param.prediction_day_count
        available_data_count-=param.prediction_day_count
    
    return classifier_result_list