1. TrainingData.csv
The provided (given) training dataset containing bitstreams and corresponding class labels.
2. main.py
The main Python program where I did data preprocessing, trained the machine learning model RandomForest(Best Performing model), performed cross-validation, and saved the trained model for later use purposes. This also included a function to predict on an independent test dataset.
3. Documentation.pdf
The documentation file that included the comparative analysis of the different models (GradientBoosting, XGBoost, and RandomForest) based on their cross-validation and validation performance and choosing the best performing model.
4. best_trained_model.joblib
The trained RandomForest classifier model, which I identified as the best-performing model based on validation accuracy. This model is capable to be loaded and used for making predictions on new, unseen data.