from index import get_latest_end_date, save_new_data
from datetime import datetime, timedelta
import os
import pandas as pd
import configparser

# create a ConfigParser object
config = configparser.ConfigParser()
# Read the config file
config.read("config.ini")

iconics_key = config["secret"]["ICONICS_KEY"]
weather_key = config["weather.key"]["WEATHER_KEY"]

# Controller Function Template
# A generalised version of the logic for both occupancy and weather
def update_data_if_needed(datatype, fetch_function, save_folder): # building_nodes=None, api_key=None
    """
    Checks if new data is needed for a given datatype and fetches it if necessary.
    Returns a combined DataFrame from the saved files.
    """
    latest_extracted_date = get_latest_end_date(datatype)
    yesterday = datetime.now() - timedelta(days=1)

    if latest_extracted_date is None:
        # Default start date if no files exist
        latest_extracted_date = datetime.strptime("2023-01-01", "%Y-%m-%d")
    else:
        latest_extracted_date += timedelta(days=1)

    if latest_extracted_date <= yesterday:
        start_str = latest_extracted_date.strftime("%Y-%m-%d")
        end_str = yesterday.strftime("%Y-%m-%d")

        if datatype == 'occupancy':
            df = fetch_function(start_str, end_str, iconics_key=iconics_key, building_nodes=[4, 5])
        elif datatype == 'weather':
            df = fetch_function(start_str, end_str, api_key=weather_key)
        else:
            raise ValueError("Unsupported datatype")

        save_new_data(datatype, start_str, end_str, df)
    else:
        print(f"✅ {datatype.capitalize()} data is up to date")

    
    dfs = merge_return_full_df(save_folder)
    # files = [f for f in os.listdir(save_folder) if os.path.isfile(os.path.join(save_folder, f))]
    # dfs = [pd.read_csv(os.path.join(save_folder, file)) for file in files]
    return dfs

# Combines all files in folder into single dataframe
def merge_return_full_df(save_folder): 
    # folder_path = "journeys/" passed in
    files = [f for f in os.listdir(save_folder) if os.path.isfile(os.path.join(save_folder, f))]
    dfs = [pd.read_csv(os.path.join(save_folder, file)) for file in files]
    return pd.concat(dfs, ignore_index=True)

# Occupancy dataframe cleaner
def prepare_occupancy_df(df: pd.DataFrame) -> pd.DataFrame:
    # filter for rows where the schedule is 07:00 to 19:00 (WorkingHours)
    occupancy_filtered = df[(df['Period_Start'] == "07:00:00") & (df['Period_End'] == "19:00:00")]
    # Keeping Only Relevant Columns
    columns_to_keep = [
        'Summary_Date_Local', 'Node_Id', 'IsWorking_Day','Space_Id', 
        'Period_Start', 'Period_End','Space_Used', 'Space_Name'
    ]

    df_filtered = occupancy_filtered[columns_to_keep]
    # Let's convert the date column to DateTime format
    df_filtered['Summary_Date_Local'] = pd.to_datetime(df_filtered['Summary_Date_Local'])

    # Aggregating to daily level
    df_daily = df_filtered.groupby('Summary_Date_Local').agg(
        total_spaces=('Space_Id', 'nunique'), 
        used_spaces=('Space_Used', 'sum')
    ).reset_index()

    # Occupancy_rates
    df_daily['Occupancy_Rates'] = df_daily['used_spaces'] / df_daily['total_spaces']

    return df_daily

# Weather dataframe cleaner
def prepare_weather_df(df: pd.DataFrame) -> pd.DataFrame:
    # Address weather date format
    df['time'] = pd.to_datetime(df['time'])
    return df

# Transport dataframe cleaner
def prepare_transport_df(df: pd.DataFrame) -> pd.DataFrame:
    # Merging DayOfWeek and DayOFWeek columns
    if 'DayOFWeek' in df.columns and 'DayOfWeek' in df.columns:
        df['DayOfWeek'] = df['DayOfWeek'].combine_first(df['DayOFWeek'])
    elif 'DayOFWeek' in df.columns:
        df.rename(columns={'DayOFWeek':'DayOfWeek'}, inplace=True)

    # Address date
    df['TravelDate'] = pd.to_datetime(df['TravelDate'], format="%Y%m%d")

    # Remove duplicate rows based on the date column
    cleaned_df = df.drop_duplicates(subset=['TravelDate'])

    return cleaned_df

# Prepare and save the final dataset for Analytics
def produce_and_save_analytics_df(df: pd.DataFrame, save_folder):
    # Drop redundant columns
    merged_dataset = df.drop(columns=['TravelDate', 'DayOfWeek', 'time']) # 'DayOFWeek',
    merged_dataset['DayOfWeek'] = merged_dataset['Summary_Date_Local'].dt.day_name()
    merged_dataset = merged_dataset[merged_dataset['Summary_Date_Local'].dt.weekday < 5]
    weather_code_map = {
        # Clear
        0: "Clear", 1: "Clear", 2: "Clear", 3: "Clear", 
        # Fog
        45: "Fog", 48: "Fog", 
        # Rain
        51: "Rain", 53: "Rain", 55: "Rain", 56: "Rain", 57: "Rain", 
        61: "Rain", 63: "Rain", 65: "Rain", 66: "Rain", 67: "Rain",
        80: "Rain", 81: "Rain", 82: "Rain",
        # Snow
        71: "Snow", 73: "Snow", 75: "Snow", 77: "Snow",
        85: "Snow", 86: "Snow", 
        # Thunderstorms
        95: "Thunderstorm", 96: "Thunderstorm", 
        99: "Thunderstorm" 
    }
    merged_dataset['Weather Label'] = merged_dataset['weather_code'].map(weather_code_map)

    merged_dataset.to_csv(os.path.join(save_folder, "processed.csv"), index=False)