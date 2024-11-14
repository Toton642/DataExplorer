import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import joblib

# Load the training dataset
df = pd.read_csv('/workspaces/DataExplorer/TrainingData.csv')
print(df.head())

# Preprocess the data
X = df['Bitstream'].apply(lambda x: [int(bit) for bit in x])  # Convert bitstreams to lists of integers
y = df['class']

# Pad the sequences to ensure equal length using numpy
max_length = max(len(seq) for seq in X)  # Find the maximum length of the bitstreams
X_padded = [seq + [0] * (max_length - len(seq)) for seq in X]  # Pad the bitstreams using list comprehension
X_padded = np.array(X_padded)  # Convert to NumPy array

# Split the data into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(X_padded, y, test_size=0.2, random_state=42)

# Train a classifier
clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X_train, y_train)

# Evaluate the classifier
y_pred = clf.predict(X_val)
accuracy = accuracy_score(y_val, y_pred)
print(f'Validation Accuracy: {accuracy:.2f}')

# Hyperparameter tuning with GridSearchCV
param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [None, 10, 20, 30],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}
grid_search = GridSearchCV(estimator=RandomForestClassifier(random_state=42), param_grid=param_grid, cv=5, n_jobs=-1, verbose=2)
grid_search.fit(X_train, y_train)

# Get the best model
best_clf = grid_search.best_estimator_
print(f'Best Parameters: {grid_search.best_params_}')
print(f'Best Score: {grid_search.best_score_}')

# Final training on all data with the best parameters
best_clf.fit(X_padded, y)

# Save the trained model to a file
joblib.dump(best_clf, '/workspaces/DataExplorer/best_trained_model.joblib')
print("Best trained model saved!!")

# Cross-validation scores
cv_scores = cross_val_score(best_clf, X_padded, y, cv=5)
print(f'Cross-Validation Scores: {cv_scores}')
print(f'Mean Cross-Validation Score: {cv_scores.mean():.2f}')

"""
# Function to predict on an independent test dataset
def predict_on_test_data(model_path, test_file_path, output_file_path, max_length):

    model = joblib.load(model_path)
    
  
    test_df = pd.read_csv(test_file_path)
    X_test = test_df['Bitstream'].apply(lambda x: [int(bit) for bit in x])
    
    
    X_test_padded = [seq + [0] * (max_length - len(seq)) for seq in X_test]
    X_test_padded = np.array(X_test_padded)  # Convert to NumPy array
    
    
    predictions = model.predict(X_test_padded)
    
   
    submission_df = pd.DataFrame({'ID': test_df.index, 'Predicted': predictions})
    submission_df.to_csv(output_file_path, index=False)
    print(f"Predictions saved to {output_file_path}")
"""
