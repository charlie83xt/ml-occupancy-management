import openmeteo_requests
import pandas as pd
import requests_cache
from retry_requests import retry

def produce_weather_forecast(start_date, end_date):
    # Setup the Open-Meteo API client with cache and retry on error
    cache_session = requests_cache.CachedSession('.cache', expire_after = 3600)
    retry_session = retry(cache_session, retries = 5, backoff_factor = 0.2)
    openmeteo = openmeteo_requests.Client(session = retry_session)

    # Make sure all required weather variables are listed here
    # The order of variables in hourly or daily is important to assign them correctly below
    url = "https://api.open-meteo.com/v1/forecast"
    params = {
        "latitude": 51.509865,
        "longitude": 0.118092,
        "daily": ["weather_code", "rain_sum", "showers_sum", "precipitation_sum", "temperature_2m_max", "temperature_2m_min", "snowfall_sum", "daylight_duration"],
        "timezone": "Europe/London",
        "start_date": f"{start_date}",
        "end_date": f"{end_date}"
    }
    responses = openmeteo.weather_api(url, params=params, verify=False)

    # Process first location. Add a for-loop for multiple locations or weather models
    response = responses[0]
    # print(f"Coordinates {response.Latitude()}°N {response.Longitude()}°E")
    # print(f"Elevation {response.Elevation()} m asl")
    # print(f"Timezone {response.Timezone()}{response.TimezoneAbbreviation()}")
    # print(f"Timezone difference to GMT+0 {response.UtcOffsetSeconds()} s")

    # Process daily data. The order of variables needs to be the same as requested.
    daily = response.Daily()

    daily_weather_code = daily.Variables(0).ValuesAsNumpy()
    daily_rain_sum = daily.Variables(1).ValuesAsNumpy()
    daily_showers_sum = daily.Variables(2).ValuesAsNumpy()
    daily_precipitation_sum = daily.Variables(3).ValuesAsNumpy()
    daily_temperature_2m_max = daily.Variables(4).ValuesAsNumpy()
    daily_temperature_2m_min = daily.Variables(5).ValuesAsNumpy()
    daily_snowfall_sum = daily.Variables(6).ValuesAsNumpy()
    daily_daylight_duration = daily.Variables(7).ValuesAsNumpy()

    daily_data = {"Summary_Date_Local": pd.date_range(
        start = pd.to_datetime(daily.Time(), unit = "s", utc = False),
        end = pd.to_datetime(daily.TimeEnd(), unit = "s", utc = False), #.tz_convert("Europe/London").normalize(),
        freq = pd.Timedelta(seconds = daily.Interval()),
        inclusive = "left"
    )}

    daily_data["weather_code"] = daily_weather_code
    daily_data["rain_sum"] = daily_rain_sum
    daily_data["showers_sum"] = daily_showers_sum
    daily_data["precipitation_sum"] = daily_precipitation_sum
    daily_data["temperature_2m_max"] = daily_temperature_2m_max
    daily_data["temperature_2m_min"] = daily_temperature_2m_min
    daily_data["snowfall_sum"] = daily_snowfall_sum
    daily_data["daylight_duration"] = daily_daylight_duration

    daily_dataframe = pd.DataFrame(data = daily_data)
    return daily_dataframe