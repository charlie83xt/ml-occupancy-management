import os
import pandas as pd
import numpy as np
import joblib
from datetime import datetime, timedelta
from pathlib import Path
import holidays
from data_processor import MLPreparation
from weather_forecast import produce_weather_forecast
from sklearn.metrics import mean_squared_error, root_mean_squared_error

# Processed data path
processed_path = "data/processed/processed.csv"
predictions_path = "data/processed/predictions.csv"

df = pd.read_csv(processed_path)
df['Summary_Date_Local'] = pd.to_datetime(df['Summary_Date_Local'])

# Function to generate future_dates excluding weekends 
def get_next_weekdays(start_date, num_days=5):
    weekdays = []
    current_date = start_date + timedelta(days=1)
    while len(weekdays) < num_days:
        if current_date.weekday() < 5:
            weekdays.append(current_date)
        current_date += timedelta(days=1)
    return weekdays

# Get last date and generate next 5 days
last_date = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0) + timedelta(days=1)
# last_date = df['Summary_Date_Local'].max()
print(last_date)
# future_dates = [last_date + timedelta(days=i) for i in range(1, 6)]
future_dates = get_next_weekdays(last_date)
##########################
# After: true_future_dates_list = get_next_weekdays(start_future_prediction_from, num_days=5)
print(f"\n--- Debug: true_future_dates_list ---")
print(future_dates)
#########################
# Call the weather forecast
start_date = (last_date + timedelta(days=1)).strftime("%Y-%m-%d")
end_date = (last_date + timedelta(days=8)).strftime("%Y-%m-%d")
weather_forecast = produce_weather_forecast(start_date, end_date)

# Normalising both dates before merge
weather_forecast['Summary_Date_Local'] = pd.to_datetime(weather_forecast['Summary_Date_Local']).dt.normalize()
future_dates = [d.normalize() for d in pd.to_datetime(future_dates)]

# Fetch weather forecast for next 5 days
weather_forecast = weather_forecast[weather_forecast["Summary_Date_Local"].isin(future_dates)]

# infer Transport data from historical average
transport_cols = ['TubeJourneyCount', 'BusJourneyCount']
transport_means = df[transport_cols].tail(30).mean()

# create future dataframe
future_df = weather_forecast.copy()
future_df['total_spaces'] = df['total_spaces'].iloc[-1]
future_df['used_spaces'] = np.nan
future_df['Occupancy_Rates'] = np.nan
future_df['TubeJourneyCount'] = transport_means['TubeJourneyCount']
future_df['BusJourneyCount'] = transport_means['BusJourneyCount']
future_df['DayOfWeek'] = future_df['Summary_Date_Local'].dt.day_name()

# Append to original data
combined_df = pd.concat([df, future_df], ignore_index=True)

# Prepare features
ml_prep = MLPreparation(combined_df)
prepared_df = ml_prep.prepare_for_ml_model()

# Extract Only Future Rows
future_rows = prepared_df[prepared_df["Summary_Date_Local"].isin(future_dates)].copy()

future_rows.drop(columns=['used_spaces', 'quarter', 'month', 'week_of_year'], inplace=True)

# Fill any remaining NaNs in lag features using previous values
for lag_col in ['Occupancy_Rates_lag_1', 'Occupancy_Rates_lag_7']:
    if future_rows[lag_col].isna().any():
        # Fill with median or last known value
        fill_value = prepared_df[lag_col].dropna().iloc[-1]
        future_rows[lag_col].fillna(fill_value, inplace=True)

# Load model
model_path = "models/trained_models/occupancy_model.pkl"
model = joblib.load(model_path)
model_name = type(model).__name__

# Apply scaler if needed
if model_name in ["LinearRegression", "XGBRegressor"]:
    scaler_path = "models/trained_models/scaler.pkl"
    if os.path.exists(scaler_path):
        scaler = joblib.load(scaler_path)

        # The columns of the DataFrame we want to remove from the scaling process
        features_to_scale = future_rows.drop(columns=["Summary_Date_Local", "total_spaces", "weather_code", "rain_sum",
                                            "showers_sum", "precipitation_sum", "temperature_2m_max", "temperature_2m_min",
                                            "snowfall_sum", "Day_Monday", "Day_Tuesday", "Day_Wednesday", "Day_Thursday", 
                                            "Day_Friday", "Month_sin", "Month_cos", "Quarter_sin", "Quarter_cos", 
                                            "Weeks_sin", "Weeks_cos", "Occupancy_Rates_lag_1", "Occupancy_Rates_lag_7", 
                                            "year", "is_holidays?", "Occupancy_Rates"]).columns
        
        # Apply Scaling
        future_rows[features_to_scale] = scaler.transform(future_rows[features_to_scale])

# Features used in training
feature_cols = ['total_spaces', 'TubeJourneyCount', 'BusJourneyCount', 'weather_code', 
                'rain_sum','showers_sum', 'precipitation_sum', 'temperature_2m_max',
                'temperature_2m_min', 'snowfall_sum', 'daylight_duration', 'Day_Friday',
                'Day_Monday', 'Day_Thursday', 'Day_Tuesday', 'Day_Wednesday', 'year',
                'is_holidays?', 'Month_sin','Month_cos', 'Quarter_sin', 'Quarter_cos', 
                'Weeks_sin', 'Weeks_cos','Occupancy_Rates_lag_1', 'Occupancy_Rates_lag_7'
]

# predict 
x_pred = future_rows[feature_cols]
future_rows['Predicted_Occupancy'] = model.predict(x_pred)

# Compare with last 5 actuals
recent_actuals = df[df['Summary_Date_Local'].isin(future_rows['Summary_Date_Local'])]
if not recent_actuals.empty:
    merged = pd.merge(recent_actuals, future_rows, on='Summary_Date_Local', suffixes=('_actual', '_pred'))
    if not merged.empty:
        rmse = root_mean_squared_error(merged['Occupancy_Rates_actual'], merged['Occupancy_Rates_pred'])
        df.loc[df['Summary_Date_Local'] == df['Summary_Date_Local'].max(), "model_accuracy"] = 100 - rmse
        df.to_csv(processed_path, index=False)

# Save predictions

if os.path.exists(predictions_path):
    existing_preds = pd.read_csv(predictions_path)
    combined_preds = pd.concat([existing_preds, future_rows], ignore_index=True)
    combined_preds.drop_duplicates(subset=["Summary_Date_Local"], keep='last', inplace=True)
else:
    combined_preds = future_rows

combined_preds.to_csv(predictions_path, index=False)

print(f"Predictions saved to {predictions_path}")
# print(recent_actuals)