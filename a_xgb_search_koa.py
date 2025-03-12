import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
import matplotlib.pyplot as plt
import seaborn as sns
import optuna
import os
import sys
import datetime

# Create a log file with timestamp
timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
log_filename = f"xgboost_optuna_{timestamp}.log"

# Create a class to capture print statements and write them to both console and log file
class Logger:
    def __init__(self, filename):
        self.terminal = sys.stdout
        self.log = open(filename, 'a')
        
    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)
        self.log.flush()  # Ensure log is written immediately
        
    def flush(self):
        self.terminal.flush()
        self.log.flush()

# Redirect stdout to our Logger class
sys.stdout = Logger(log_filename)

print(f"Log file created: {log_filename}")
print(f"Run started at: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

# Load the data
print("Loading data...")
data = pd.read_csv("data/train.csv")

# Separate features and labels
X = data.iloc[:, 1:].values  # All columns except the first one
y = data.iloc[:, 0].values   # First column contains labels

# Split the data into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Scale the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)

# Define the objective function for Optuna
def objective(trial):
    params = {
        'n_estimators': trial.suggest_int('n_estimators', 100, 3000),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
        'max_depth': trial.suggest_int('max_depth', 3, 15),
        'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
        'subsample': trial.suggest_float('subsample', 0.2, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.2, 1.0),
        'reg_alpha': trial.suggest_float('reg_alpha', 0.001, 10, log=True),
        'reg_lambda': trial.suggest_float('reg_lambda', 0.001, 10, log=True),
        'objective': 'binary:logistic',
        'eval_metric': 'auc',
        'random_state': 42,
        # 'n_jobs': 16
    }
    
    # Initialize and train the XGBoost model
    model = xgb.XGBClassifier(**params)
    model.fit(
        X_train_scaled, y_train,
        eval_set=[(X_val_scaled, y_val)],
        verbose=False  # Suppress training output
    )
    
    # Predict probabilities and compute AUC
    y_val_pred_proba = model.predict_proba(X_val_scaled)
    auc = roc_auc_score(y_val, y_val_pred_proba[:, 1])
    return auc

# Perform Bayesian optimization with Optuna
print("Starting Bayesian optimization with Optuna...")
study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=100)  # Adjust n_trials as needed
print("Bayesian optimization completed!")

# Print the best parameters and score
print("\nBest parameters found:")
print(study.best_params)
print(f"\nBest validation AUC: {study.best_value:.4f}")

# Train the best model with the optimal parameters
best_params = study.best_params
best_params['objective'] = 'binary:logistic'
best_params['eval_metric'] = 'auc'
best_params['random_state'] = 42
# best_params['n_jobs'] = 16
best_model = xgb.XGBClassifier(**best_params)
best_model.fit(
    X_train_scaled, y_train,
    eval_set=[(X_val_scaled, y_val)],
    verbose=True
)

# Evaluate on validation set
y_val_pred_proba = best_model.predict_proba(X_val_scaled)
val_auc = roc_auc_score(y_val, y_val_pred_proba[:, 1])
print(f"Validation set AUC: {val_auc:.4f}")

# Simulate grid search results for visualization (using trials from Optuna)
results = pd.DataFrame(study.trials_dataframe())
results['mean_test_score'] = results['value']  # AUC from validation set
results['rank_test_score'] = results['value'].rank(ascending=False).astype(int)
results = results.sort_values(by='rank_test_score')

# Save the results to a CSV file
# Create the directory with an increment if it already exists

base_dir = 'current_run'
dir_name = base_dir
counter = 1

while os.path.exists(dir_name):
    dir_name = f"{base_dir}_{counter}"
    counter += 1

os.makedirs(dir_name)
results.to_csv(f'{dir_name}/xgboost_optuna_results.csv', index=False)

# Plot the top 10 combinations
plt.figure(figsize=(12, 6))
sns.barplot(x='rank_test_score', y='mean_test_score', 
            data=results.head(10))
plt.title('Top 10 Parameter Combinations (AUC)')
plt.xlabel('Rank')
plt.ylabel('Validation AUC Score')

# Zoom into regime where difference is visible based on min and max
min_auc = results['mean_test_score'].min()
max_auc = results['mean_test_score'].max()
plt.ylim(min_auc - 0.01, max_auc + 0.01)

plt.tight_layout()
plt.savefig(f'{dir_name}/xgboost_optuna_results.png')

# Save the best model
best_model.save_model(f'{dir_name}/best_xgboost_model.json')

print("\nResults have been saved to:")
print("- xgboost_optuna_results.csv")
print("- xgboost_optuna_results.png")
print("- xgboost_feature_importance.png")
print("- best_xgboost_model.json")
# Print log file information
print(f"\nAll output has been logged to: {log_filename}")
print(f"Run completed at: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

# Restore original stdout and close log file
sys.stdout.log.close()
sys.stdout = sys.stdout.terminal

