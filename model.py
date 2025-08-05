import os
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import TimeSeriesSplit, RandomizedSearchCV
from scipy.stats import uniform, randint
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
import joblib
from data_processor import MLPreparation
import time
from deep_learning import DeepLearningClass

try:
    from xgboost import XGBRegressor
    xgb_available = True
except ImportError:
    xgb_available = False


# Defined paths
data_path = "data/processed/processed.csv"
model_dir = "models/trained_models/"
model_path = os.path.join(model_dir, "occupancy_model.pkl")

# Ensure model directory exists
os.makedirs(model_dir, exist_ok=True)

# Instantiate Data Preparation class with path to data
df = pd.read_csv(data_path)
ml_prepare = MLPreparation(df)
ml_df = ml_prepare.prepare_for_ml_model()

# Drop unused columns
final_ml_df = ml_df.copy()
final_ml_df.drop(columns=['used_spaces', 'quarter', 'month', 'week_of_year'], inplace=True)

# Splitting the data
X = final_ml_df.drop('Occupancy_Rates', axis=1)
y = final_ml_df['Occupancy_Rates']

# The columns of the DataFrame we want to remove from the scaling process
features_to_scale = X.drop(columns=["Summary_Date_Local", "total_spaces", "weather_code", "rain_sum",
                                    "showers_sum", "precipitation_sum", "temperature_2m_max", "temperature_2m_min",
                                    "snowfall_sum", "Day_Monday", "Day_Tuesday", "Day_Wednesday", "Day_Thursday", 
                                    "Day_Friday", "Month_sin", "Month_cos", "Quarter_sin", "Quarter_cos", 
                                    "Weeks_sin", "Weeks_cos", "Occupancy_Rates_lag_1", "Occupancy_Rates_lag_7", 
                                    "year", "is_holidays?"]).columns

# Here we copy to recover the previously dropped columns
X_scaled = X.copy()

# Testing Pipeline
scaler = StandardScaler()
# Scaling will affect only the features left in the to Scale DataFrame
X_scaled[features_to_scale] = scaler.fit_transform(X[features_to_scale])

# Finally drop the date column no longer needed to fit the LR model
X_scaled.drop(columns=['Summary_Date_Local'], inplace=True)
X.drop(columns=['Summary_Date_Local'], inplace=True)

train_size = int(len(final_ml_df) * 0.8)

X_train_scaled, X_test_scaled = X_scaled.iloc[:train_size], X_scaled.iloc[train_size:]
X_train_raw, X_test_raw = X.iloc[:train_size], X.iloc[train_size:]
y_train, y_test = y.iloc[:train_size], y.iloc[train_size:]

results = []
# Model Evaluation
def evaluate_model(model, X_train, y_train, X_test, y_test, model_name):
    time_start = time.time()
    model.fit(X_train, y_train)
    time_end = time.time()
    training_time = time_end - time_start

    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_pred, y_test)
    rmse = np.sqrt(mse)

    results.append({
        "Model": model,
        "Training Time (seconds)": training_time,
        "Model Name": model_name,
        "Model MSE": mse,
        "Model RMSE": rmse
    })

#

# Model 1. Train Linear Regression model scaled
evaluate_model(LinearRegression(), X_train_scaled, y_train, X_test_scaled, y_test, "Linear Regression")

# Model 2. Decision Tree Regressor
evaluate_model(DecisionTreeRegressor(), X_train_raw, y_train, X_test_raw, y_test, "Decision Tree Regressor")

# Model 3. Random Forest Regressor
evaluate_model(RandomForestRegressor(), X_train_raw, y_train, X_test_raw, y_test, "Random Forest Regressor")

# Model 4. XGBoost
max_depth = 4
gbrt = GradientBoostingRegressor(max_depth=max_depth, n_estimators=120)
gbrt.fit(X_train_raw, y_train)

errors = [mean_squared_error(y_test, y_pred)
         for y_pred in gbrt.staged_predict(X_test_raw)]
best_n_estimators = np.argmin(errors) + 1

gbrt_best = GradientBoostingRegressor(max_depth=max_depth, n_estimators=best_n_estimators)

# gbrt_best.fit(X_train, y_train)
evaluate_model(gbrt_best, X_train_raw, y_train, X_test_raw, y_test, "Gradient Boosting Regressor(Best Estimators)")

# Model 5. Same as 4 but randomised search best estimators
param_distributions = {
    'n_estimators': randint(100, 2000),
    'learning_rate': uniform(0.01, 0.15),
    'max_depth': randint(3, 8),
    'subsample': uniform(0.7, 0.3),
    'max_features': ['sqrt', 'log2', 0.8, 0.9, 1.0],
    'min_samples_leaf': randint(1, 20),
    'min_samples_split': randint(2, 40)
}

# Setting TimeSeriesSplit
tscv = TimeSeriesSplit(n_splits=5)

gbr_random_search = RandomizedSearchCV(
    estimator=GradientBoostingRegressor(random_state=42),
    param_distributions=param_distributions,
    n_iter=50,
    cv=tscv,
    scoring='neg_root_mean_squared_error',
    random_state=42,
    n_jobs=1,
    verbose=2
)
gbr_random_search.fit(X_train_raw, y_train)

evaluate_model(gbr_random_search.best_estimator_, X_train_raw, y_train, X_test_raw, y_test, "Gradient Boosting Regressor(Randomized Search)")

# Model 6. XGBoost (scaled)

if xgb_available:
    xgb = XGBRegressor(n_estimators=100, random_state=42)
    evaluate_model(xgb, X_train_scaled, y_train, X_test_scaled, y_test, "XGBoost")

# Model 7. LSTM (Deep Learning)
# Data Preparation for Time Series Deep Learning
# final_ml_df declared on the models page
dl_df = ml_df.copy().sort_values('Summary_Date_Local').set_index('Summary_Date_Local')

# Features and target (excluding date if it's the index or dropping it if is a column)
features = [col for col in dl_df.columns if col != 'Occupancy_Rates']
X = dl_df[features].values
y = dl_df['Occupancy_Rates'].values

# StandardScaler to all features
scaler_X = StandardScaler()
X_scaled_df = scaler_X.fit_transform(X)

scaler_y = StandardScaler()
y_scaled_target = scaler_y.fit_transform(y.reshape(-1, 1))

# Instantiate DL class and fit model 
dl = DeepLearningClass(X_scaled_df, y_scaled_target)
start_time_lstm = time.time()
dl_model, X_test_seq, y_test_seq = dl.model_return()
end_time_lstm = time.time()
lstm_training_time = end_time_lstm - start_time_lstm
# Making predictions on the lstm test set
y_pred_scaled = dl_model.predict(X_test_seq)
y_pred_original = scaler_y.inverse_transform(y_pred_scaled)

y_test_unscaled_for_rmse = scaler_y.inverse_transform(y_test_seq[:, 0].reshape(-1, 1))
lstm_mse = mean_squared_error(y_test_unscaled_for_rmse, y_pred_original[:, 0])
lstm_rsme = np.sqrt(lstm_mse)

results.append({
    "Model": dl_model,
    "Training Time (seconds)": lstm_training_time,
    "Model Name": "LSTM",
    "Model MSE": lstm_mse,
    "Model RMSE": lstm_rsme
})

results_df = pd.DataFrame(results)
results_df.to_csv("data/latest_model_evaluation.csv")

# Select best model
best_model_result = min(results, key=lambda x: x['Model RMSE'])
best_model = best_model_result["Model"]
best_name = best_model_result["Model Name"]
best_rmse = best_model_result["Model RMSE"]

# Save best model
joblib.dump(best_model, model_path)
print(f"Model saved to {model_path}")

# Save scaler if used
if best_name in ["Linear Regression", "XGBoost"]:
    scaler_path = os.path.join(model_dir, "scaler.pkl")
    joblib.dump(scaler, scaler_path)

# Output Summary
print(f"✅ Best model: {best_model} with RSME: {best_rmse:.4f}")
print(f"📦 Model saved to: {model_path}")
if best_name in ["Linear Regression", "XGBoost"]:
    print(f"📦 Scaler saved to {scaler_path}")
