import pandas as pd
import numpy as np
import holidays
from pathlib import Path
# I will need to navigate the raw_occupancy folder and create a single DF

# Data path
data_path = "data/processed/processed.csv"

class MLPreparation:

    def __init__(self, df: pd.DataFrame):
        self.df = df
        self.date_column = "Summary_Date_Local"
        self.target_column = "Occupancy_Rates"
    
    # Label Hot-encode categoric variables
    def encode_categoric(self):
        self.df = pd.get_dummies(self.df, columns=['DayOfWeek'], prefix='Day', dtype='int64')
        self.df.drop(columns=['Weather Label'], inplace=True)

    # Adding Time-Related Features for Seasonality
    def time_related_features(self):
        """
            The dataframe is expected to have a datetime column
        """
        # convert date column into datetime
        self.df[self.date_column] = pd.to_datetime(self.df[self.date_column])

        # Adding Time-Related Features for Seasonality
        self.df['year'] = self.df[self.date_column].dt.year
        self.df['month'] = self.df[self.date_column].dt.month
        self.df['week_of_year'] = self.df[self.date_column].dt.isocalendar().week
        self.df['quarter'] = self.df[self.date_column].dt.quarter


    # Function to add a holidays boolean column. If holiday: 1 else 0
    def column_holidays(self):
        # Create a dict-like object for England's public holidays
        uk_holidays = holidays.UK(subdiv="England")

        self.df['is_holidays?'] = self.df[self.date_column].apply(lambda x: 1 if x in uk_holidays else 0)
        # return self.df

    # This will conserve the cyclic nature of the dates when the timestamp column is removed
    def sine_cosine_transform(self):
        # Apply sine and cosine transformations
        #Months
        num_months = 12 # Number of unique months in the cycle
        self.df['Month_sin'] = np.sin(2 * np.pi * self.df['month'] / num_months)
        self.df['Month_cos'] = np.cos(2 * np.pi * self.df['month'] / num_months)
        #QUARTERS
        num_quarters = 4 # Number of unique months in the cycle
        self.df['Quarter_sin'] = np.sin(2 * np.pi * self.df['quarter'] / num_quarters)
        self.df['Quarter_cos'] = np.cos(2 * np.pi * self.df['quarter'] / num_quarters)
        #WEEKS
        num_weeks = 52 # Number of unique months in the cycle
        self.df['Weeks_sin'] = np.sin(2 * np.pi * self.df['week_of_year'] / num_weeks)
        self.df['Weeks_cos'] = np.cos(2 * np.pi * self.df['week_of_year'] / num_weeks)

        # return self.df

    # A function to understand the median occupancy as per day of week
    def median_by_day(self):
        median_occupancy_by_day = self.df.groupby(self.df[self.date_column].dt.weekday)[self.target_column].median()
        return median_occupancy_by_day

    # A function to fill null values with the median occupancy base don day of week
    def impute_lag_by_day(self, row, lag_col_name, median_map):
        if pd.isna(row[lag_col_name]):
            day_num = row[self.date_column].weekday()
            return median_map.get(day_num, np.nan)
        return row[lag_col_name]

    # Sorting in datetime order and adding lag columns
    def sorting_and_adding_lags(self):
        # Sort in datetime order
        self.df = self.df.sort_values(by=self.date_column).reset_index(drop=True)
        # Create Occupancy lag columns
        self.df[self.target_column+"_lag_1"] = self.df[self.target_column].shift(periods=1)
        self.df[self.target_column+"_lag_7"] = self.df[self.target_column].shift(periods=7)


    # Median average by day filling for missing values in lag columns
    def median_average_filling(self):
        median_occupancy_by_day = self.median_by_day()
        for lag_col in [self.target_column+"_lag_1", self.target_column+"_lag_7"]:
            self.df[lag_col] = self.df.apply(lambda row: self.impute_lag_by_day(row, lag_col, median_occupancy_by_day), axis=1)
        return self.df 


    def prepare_for_ml_model(self):
        # We will instantiate this class and will prepare the DataFrame ready for ML 
        self.encode_categoric()
        self.time_related_features()
        self.column_holidays()
        self.sine_cosine_transform()
        self.sorting_and_adding_lags()
        self.df = self.median_average_filling()

        return self.df

