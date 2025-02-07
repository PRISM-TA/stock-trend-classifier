To run the code:

1. Install the packages in requirements.txt
2. To train model and store results to database：
Run the code with "python test.py" 
3. To get the accuracy report stored in accuracy_analysis_results folder：
Run the code with "python scripts/analyze_accuracy.py --ticker 'ticker' --model 'model' --features 'feature_set'" 
4. Before training: switch feature set in get_data method of staggered_training in mlp_model.py
Rename the ticker, model_name, and feature_set in test.py
