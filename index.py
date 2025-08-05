import os
import re
import sqlite3
from datetime import datetime, timedelta
import pandas as pd

# Directory structure
data_dirs = {
    'occupancy': 'data/raw_occupancy/',
    'weather': 'data/raw_weather/',
    'transport': 'data/raw_transport'
}

# Ensure directories exist
for dir_path in data_dirs.values():
    os.makedirs(dir_path, exist_ok =True)

def get_db_connection():
    # Setud database
    db_path = 'data/data_records.db'
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    return conn, cursor


# Create table if not exists
conn, cursor = get_db_connection()
cursor.execute('''
CREATE TABLE IF NOT EXISTS records (
    id INTEGER PRIMARY KEY,
    datatype TEXT,
    startdate TEXT,
    enddate TEXT,
    filename TEXT,
    timestamp TEXT
)
''')
conn.commit()
conn.close()

# Extract dates from filenames
def extract_dates(filename):
    match = re.search(r'(\d{4}-\d{2}-\d{2})_(\d{4}-\d{2}-\d{2})', filename)
    return match.groups() if match else (None, None)

# Get Latest end date
def get_latest_end_date(datatype):
    latest_date = None
    for filename in os.listdir(data_dirs[datatype]):
        _, enddate = extract_dates(filename)
        if enddate:
            enddate = datetime.strptime(enddate, "%Y-%m-%d")
            if not latest_date or enddate > latest_date:
                latest_date = enddate
    return latest_date

# Save new data and update DB
def save_new_data(datatype, startdate, enddate, data):
    conn, cursor = get_db_connection()
    filename = f"{datatype}_{startdate}_{enddate}.csv"
    filepath = os.path.join(data_dirs[datatype], filename)

    if isinstance(data, pd.DataFrame):
        data.to_csv(filepath, index=False)
    else:
        with open(filepath, 'w') as f:
            f.write(data)
            
    cursor.execute('''
    INSERT INTO records (datatype, startdate, enddate, filename, timestamp)
    VALUES (?, ?, ?, ?, ?)
    ''', (datatype, startdate, enddate, filename, datetime.now().isoformat()))
    conn.commit()
    conn.close()

# Placeholder: fetch data from API
def fetch_data_from_api(datatype, startdate, enddate):
    # replace with actual API call
    return f"{{'datatype': '{datatype}', 'start': '{startdate}', 'end': '{enddate}'}}"

# Placeholder: Download transport file
def download_transport_file():
    # replace with actual download logic
    return "date,location,transport_count\n2025-01-01,Station A,123\n..."

# Update transport data
def update_transport_data():
    conn, cursor = get_db_connection()
    current_year = datetime.now().year
    next_year = current_year + 1
    filename = f"transport_{current_year}_{next_year}.csv"
    filepath = os.path.join(data_dirs['transport'], filename)
    if not os.path.exists(filepath):
        data = download_transport_file()
        with open(filepath, 'w') as f:
            f.write(data)
        cursor.execute('''
        INSERT INTO records (datatype, startdate, enddate, filename, timestamp)
        VALUES (?, ?, ?, ?, ?)
        ''', ('transport', f'{current_year}-01-01', f'{current_year}-12-31', filename, datetime.now().isoformat()))
        conn.commit()
        conn.close()


# Update occupancy and weather data
def update_data(datatype):
    latest_end_date = get_latest_end_date(datatype)
    if latest_end_date:
        latest_end_date += timedelta(days=1)
    else:
        latest_end_date = datetime.strptime('2023-01-01', '%Y-%m-%d')
    yesterday = datetime.now() - timedelta(days=1)
    if latest_end_date <= yesterday:
        startdate = latest_end_date.strftime("%Y-%m-%d")
        enddate = yesterday.strftime("%Y-%m-%d")
        data = fetch_data_from_api(datatype, startdate, enddate)
        save_new_data(datatype, startdate, enddate, data)


# Run updates
# update_data('occupancy')
# update_data('weather')
# update_transport_data()

# close DB
conn.close()

print(" Data update process completed.")