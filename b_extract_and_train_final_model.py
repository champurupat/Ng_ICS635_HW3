import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.preprocessing import StandardScaler

# Load the full dataset
print("Loading full dataset...")
data = pd.read_csv("data/train.csv")

# Separate features and labels
X_full = data.iloc[:, 1:].values  # All columns except the first one
y_full = data.iloc[:, 0].values   # First column contains labels (e.g., 1.0, 0.0)

# Load the optimal hyperparameters from the CSV
print("Retrieving optimal hyperparameters from 'current_run/xgboost_optuna_results.csv'...")
results = pd.read_csv('current_run/xgboost_optuna_results.csv')

# Get the top row (best trial, rank_test_score == 1)
best_trial = results[results['rank_test_score'] == 1].iloc[0]

# Extract hyperparameters (strip 'params_' prefix from column names)
best_params = {}
for col in results.columns:
    if col.startswith('params_'):
        param_name = col.split('params_')[1]
        best_params[param_name] = best_trial[col]

# Add fixed parameters from the original setup
best_params['objective'] = 'binary:logistic'
best_params['eval_metric'] = 'auc'
best_params['random_state'] = 42

print("Optimal hyperparameters retrieved:")
print(best_params)

# Scale the full dataset
print("Scaling the full dataset...")
scaler = StandardScaler()
X_full_scaled = scaler.fit_transform(X_full)

# Train the model on the full dataset
print("Training XGBoost model on the full dataset...")
final_model = xgb.XGBClassifier(**best_params)
final_model.fit(X_full_scaled, y_full)  # No eval_set, train on all data

# Save the trained model
final_model.save_model('current_run/final_xgboost_model.json')
print("Model trained and saved as 'final_xgboost_model.json'")