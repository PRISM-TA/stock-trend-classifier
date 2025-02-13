To run the code:

1. Install the packages in requirements.txt
2. To train model and store results to database：
Run the code with "python main.py" 
3. To get the accuracy report stored in accuracy_analysis_results folder：
Run the code with "python scripts/analyze_accuracy.py --ticker 'ticker' --model 'model' --features 'feature_set'" 
4. Before training: switch feature set in get_data method of staggered_training in staggered_training.py
Rename the ticker, model_name, and feature_set in main.py
5. Without specifying the days of data, the default number of day is one-day 
("Raw technical indicators" means the indicators on the report day only; 
"Raw technical indicators (20 days)" means there's a lookback window of 20 days)
6. The file names of accuracy_analysis_results follow this structure: ticker_model_featureSet
To shorten the name, we only pick the capital letters of the feature set, i.e. RMD(DRTI stands for Raw Market Data (20 days) + Raw Technical Indicators
