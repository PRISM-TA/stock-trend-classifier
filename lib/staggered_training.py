from models.MarketDataset import MarketDataset
from models.MarketData import MarketData
from models.StaggeredTrainingParam import StaggeredTrainingParam
from models.BaseHyperParam import BaseHyperParam
from models.ClassifierResult import ClassifierResult

from classifiers.factory.ClassifierFactory import ClassifierFactory

from features.BaseFeatureSet import BaseFeatureSet

from lib.data_preprocessing import calculate_class_weights

from scripts.loss_chart import plot_consolidated_loss_chart

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

import os
from sqlalchemy import select, func
from sklearn.preprocessing import StandardScaler


def staggered_training(session, param: StaggeredTrainingParam, classifier_factory: ClassifierFactory, model_param: BaseHyperParam, feature_set: BaseFeatureSet):
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
    
    def get_available_data_count(session, ticker):
        with session as session:
            query = (
                select(
                    func.count(MarketData.report_date)
                ).where(MarketData.ticker == ticker)
            )
            
            query_result = session.execute(query).all()
            print(f"[DEBUG] available_data_count for {ticker}: {query_result[0][0]}")
            return query_result[0][0]

    def get_report_date(session, offset, count, ticker):
        with session as session:
            query = (
                select(MarketData.report_date)
                .where(MarketData.ticker == ticker)
                .order_by(MarketData.report_date)
                .offset(offset)
                .limit(count)
            )
            
            query_result = session.execute(query).all()
            return [record[0] for record in query_result]

    classifier_result_list = []
    loss_files = []  # List to collect all loss files for chart generation

    available_data_count = get_available_data_count(session, param.ticker) - param.training_day_count
    if available_data_count < 0:
        print(f"[DEBUG] Not enough training data available for {param.ticker}")
        return classifier_result_list
    
    training_offset = 0
    prediction_offset = param.training_day_count
    window_num = 0

    while available_data_count > param.prediction_day_count:
        print(f"[DEBUG] Available data count: {available_data_count}")
        print(f"[DEBUG] Training on {param.ticker} from {training_offset} to {prediction_offset}")
        print(f"[DEBUG] Testing on {param.ticker} from {prediction_offset} to {prediction_offset + param.prediction_day_count}")

        feat_train_df, label_train_df = feature_set.get_data(session, training_offset, param.training_day_count, param.ticker)
        feat_pred_df, label_pred_df = feature_set.get_data(session, prediction_offset, param.prediction_day_count, param.ticker)
        train_features, train_labels, pred_features, pred_labels = scale_data(feat_train_df, label_train_df, feat_pred_df, label_pred_df)
        train_loader, pred_loader = create_dataloader(train_features, train_labels, pred_features, pred_labels)

        report_date_list = get_report_date(session, prediction_offset, param.prediction_day_count, param.ticker)

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = classifier_factory.create_classifier(input_size=train_features.shape[1]).to(device)

        criterion = nn.CrossEntropyLoss(weight=calculate_class_weights(train_labels).to(device))
        # criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.SGD(model.parameters(), lr=0.001)

        # Train the model with window number
        train_losses, val_losses = model.train_classifier(
            criterion=criterion,
            optimizer=optimizer,
            train_loader=train_loader,  
            param=model_param,
            val_loader=train_loader,
            ticker=param.ticker,
            feature_set=feature_set,
            window_num=window_num
        )
        
        # Store the loss file path
        loss_file = f"loss_record/{model.model_name}_{param.ticker}_"
        if hasattr(feature_set, 'set_name'):
            # Create acronym for feature set
            feature_set_name = feature_set.set_name
            words = feature_set_name.replace('+', ' ').split()
            acronym = ''.join(word[0].upper() for word in words)
            
            # Add any parenthesized content
            if '(' in feature_set_name:
                start_idx = feature_set_name.find('(')
                end_idx = feature_set_name.find(')')
                if end_idx > start_idx:
                    acronym += feature_set_name[start_idx:end_idx+1]
            
            loss_file += f"{acronym}_"
        loss_file += f"window{window_num}.json"
        
        if os.path.exists(loss_file):
            loss_files.append(loss_file)

        model.eval()
        with torch.no_grad():
            for report_date, (features, label) in zip(report_date_list, pred_loader):
                features = features.to(device)
                outputs = model(features)
                # Get probabilities using softmax
                probs = torch.softmax(outputs, dim=1).cpu().numpy()[0]
                prediction = torch.argmax(outputs, dim=1).item()

                classifier_result_list.append(
                    ClassifierResult(
                        report_date=report_date,
                        ticker=param.ticker,
                        model=model.model_name,
                        feature_set=feature_set.set_name,
                        uptrend_prob=float(f"{probs[0]:.4f}"),
                        side_prob=float(f"{probs[1]:.4f}"),
                        downtrend_prob=float(f"{probs[2]:.4f}"),
                        predicted_label=int(prediction),
                        actual_label=int(label.item())
                    )
                )
        training_offset+=param.prediction_day_count
        prediction_offset+=param.prediction_day_count
        available_data_count-=param.prediction_day_count
    # After all windows are completed, generate the consolidated loss chart
    if loss_files:
        try:
            plot_consolidated_loss_chart(
                json_files=loss_files, 
                model_name=model.model_name, 
                ticker=param.ticker, 
                feature_set=feature_set,
                save_path='loss_charts'  # Explicitly pass a string for save_path
            )
        except Exception as e:
            print(f"Error generating loss chart: {e}")    
    return classifier_result_list