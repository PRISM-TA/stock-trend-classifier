from models.MarketDataset import MarketDataset
from models.MarketData import MarketData
from models.StaggeredTrainingParam import StaggeredTrainingParam
from models.BaseHyperParam import BaseHyperParam
from models.ClassifierResult import ClassifierResult

from classifiers.factory.ClassifierFactory import ClassifierFactory

from features.BaseFeatureSet import BaseFeatureSet

from lib.data_preprocessing import calculate_class_weights

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

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

    available_data_count = get_available_data_count(session, param.ticker) - param.training_day_count
    if available_data_count < 0:
        print(f"[DEBUG] Not enough training data available for {param.ticker}")
    training_offset = 0
    prediction_offset = param.training_day_count

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

        model.train_classifier(
            criterion=criterion,
            optimizer=optimizer,
            train_loader=train_loader,  
            param=model_param,
            val_loader=train_loader
        )

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
    
    return classifier_result_list