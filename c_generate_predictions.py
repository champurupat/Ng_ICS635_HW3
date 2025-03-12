import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.preprocessing import StandardScaler

# Load the model trained on full data
model = xgb.XGBClassifier()
model.load_model('current_run/final_xgboost_model.json')

# Load and preprocess test data
test_data = pd.read_csv("data/test.csv")
X_test = test_data.values  # All columns are features
scaler = StandardScaler()
X_test_scaled = scaler.fit_transform(X_test)

# Generate predictions (binary classification, positive class probabilities)
y_pred_proba = model.predict_proba(X_test_scaled)[:, 1]

# Create submission DataFrame
submission = pd.DataFrame({
    'Id': np.arange(len(test_data)).astype(int),
    'Predicted': y_pred_proba
})
submission['Id'] = submission['Id'].apply(lambda x: '{:.18e}'.format(float(x)))

# Save to CSV
submission.to_csv('submission.csv', index=False)
print("Submission saved to 'submission.csv'")