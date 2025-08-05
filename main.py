import pandas as pd
import subprocess

from utils import (
    update_data_if_needed, 
    merge_return_full_df,
    prepare_occupancy_df,
    prepare_weather_df,
    prepare_transport_df,
    produce_and_save_analytics_df
)

from data_fetcher import fetch_occupancy_data, fetch_weather_data, fetch_transport_data

# Occupancy
occupancy_df = update_data_if_needed(
    datatype = 'occupancy',
    fetch_function = fetch_occupancy_data,
    save_folder = "data/raw_occupancy/",
)

# Weather
weather_df = update_data_if_needed(
    datatype = 'weather',
    fetch_function = fetch_weather_data,
    save_folder = 'data/raw_weather/'
)

if occupancy_df is not None and weather_df is not None:
    # Proceed to analytics and dashboard
    # run_dashboard(occupancy_df, weather_df)
    fetch_transport_data()
    transport_df = merge_return_full_df("data/raw_transport/")
    print("Data saved and ready for Analytics.")
else:
    print("Data not ready. Skipping analytics.")

# Call predictor.py to generate predictions
subprocess.run(["python", "predictor.py"])

# Process and merge data
#occupancy
occupancy = prepare_occupancy_df(occupancy_df)
# Weather
weather = prepare_weather_df(weather_df)
# Transport
transport = prepare_transport_df(transport_df)

#Merging df_daily, cleaned_df(journeys), df_weather into one dataset
merged = pd.merge(occupancy, transport, left_on='Summary_Date_Local', right_on='TravelDate', how='inner')
merged = pd.merge(merged, weather, left_on='Summary_Date_Local', right_on='time', how='inner')



analytics_df = pd.read_csv("data/processed/processed.csv")
if not analytics_df.empty:
    # call the dashboard_analytics
    print("Data processed and saved to data/processed/processed.csv")
    # Here processed csv is updated and ready to make predictions
    # Ready for analytics
    produce_and_save_analytics_df(merged, "data/processed/")
else:
    print("No analytics dashboard ready")



