
import pandas as pd
import numpy as np
import requests
import os
from datetime import datetime, timedelta
from dateutil import relativedelta
import json
import warnings
import configparser
warnings.filterwarnings(action='ignore')
import re
import time

import urllib.parse
from urllib.parse import unquote
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from webdriver_manager.chrome import ChromeDriverManager
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import StaleElementReferenceException, TimeoutException, WebDriverException


def fetch_occupancy_data(start_date: str, end_date: str, building_nodes: list, iconics_key: str, max_retries=3, delay_seconds=1) -> pd.DataFrame:
    """
        Retrieves and processes space utilization data month-by-month for the past 24 months.
        Returns a Pandas DataFrame with the accumulated data.

        Parameters:
        start_date (str): Start date in 'YYYY-MM-DD' format.
        end_date (str): End date in 'YYYY-MM-DD' format.
        building_nodes (list): List of building node IDs to query.
        iconics_key (str): API key for authorization.
        max_retries (int): Number of retry attempts for failed requests.
        delay_seconds (int): Delay between retries and requests. 
    """
    base_url = "https://api.ibss.arcadis.iconics.cloud/v1/{node}/SpacesDailySummary"
    headers = {
        'accept': '*/*',
        'Authorization': f'{iconics_key}'
    }
    responses = []

    start_dt = datetime.strptime(start_date, "%Y-%m-%d")
    end_dt = datetime.strptime(end_date, "%Y-%m-%d")

    # today = datetime.utcnow()
    # building_nodes = [4, 5] # You can expand this list if needed
    # start_date = (today - relativedelta.relativedelta(years=2)).replace(hour=0, minute=0, second=0, microsecond=0)
    
    for node in building_nodes:
        current_date = start_dt
        while current_date <= end_dt:
            # range_end = min(range_start + timedelta(days=6), today)
        # for months_back in range(24):
            # range_start = range_end - relativedelta.relativedelta(months=1)
            # range_end = range_start + timedelta(days=1)
            
            date_str = current_date.strftime("%Y-%m-%d")
            # end_date_str = range_start.strftime("%Y-%m-%d")
            
            params = {
                "startDate": date_str,
                "endDate": date_str,
                "schema": "false",
                "$filter": "Space_Work_Type eq 'DeskWork'",
                "isUtc": "false"
            }

            url = base_url.format(node=node)
            attempt = 0
            success = False

            while attempt < max_retries and not success:
                try:
                    response = requests.get(url, headers=headers, params=params, timeout=30, verify=False)
                    
                    if response.status_code == 200:
                        data = response.json()
                        responses.extend(data)
                        print(f"✅ Data collected for {date_str} (Node {node})")
                        success = True
                    elif response.status_code == 204:
                        print(f"ℹ️ No data for {date_str} (Node {node})")
                        success = True
                    else:
                        print(f"⚠️ Error {response.status_code} for {date_str} (Node {node}): {response.text}")
                        break
                except Exception as e:
                    attempt += 1
                    print(f"❌ Attempt {attempt} failed for {date_str} (Node {node}): {e}")
                    if attempt < max_retries:
                        time.sleep(delay_seconds * attempt)
                    else:
                        print(f"🚫 Skipping {date_str} (Node {node}) after {max_retries} failed attempts.")
                        
            current_date += timedelta(days=1)
            time.sleep(delay_seconds)
            
    df = pd.DataFrame(responses)
    return df

# Weather
# A clean and modular version of the weather logic that fits into the controller
def fetch_weather_data(start_date: str, end_date: str, api_key: str) -> pd.DataFrame:
        
    """
        Fetches historical weather data from Open-Meteo API for Central London.
        Returns a DataFrame with daily weather metrics.
    """
    LAT = 51.5074
    LON = -0.1278

    url = (
        f"https://historical-forecast-api.open-meteo.com/v1/forecast"
        f"?latitude={LAT}&longitude={LON}&start_date={start_date}&end_date={end_date}"
        f"&daily=weather_code,rain_sum,showers_sum,precipitation_sum,"
        f"temperature_2m_max,temperature_2m_min,snowfall_sum,daylight_duration"
        f"&timezone=Europe%2FLondon"
    )

    response = requests.get(url, verify=False)
    if response.status_code == 200:
        data = response.json()
        daily_data = data.get("daily", {})
        df = pd.DataFrame(daily_data)
        return df
    else:
        raise Exception(f"Failed to fetch weather data: {response.status_code} - {response.text}")

# Transport
"""
    🚇 Transport Logic
    Since the transport file is updated weekly:

    Run a simple check every Monday.
    Replace the file if the current year’s file is missing or outdated.
    No need for date range logic — just overwrite and log.
"""
def fetch_transport_data():
    """
    Fetches the latest transport data CSV file from the TfL website.

    This function navigates to the TfL crowding data page, identifies
    the link to the latest "Journeys" CSV file, extracts its download URL,
    and then uses the requests library to download the file.
    """
    # Target page and download directory
    target_dir = os.path.abspath("data/raw_transport/")
    os.makedirs(target_dir, exist_ok=True)

    options = webdriver.ChromeOptions()
    prefs = {
        "download.default_directory": target_dir, # Keep target_dir for downloads
        "download.prompt_for_download": False,
        "download.directory_upgrade": True,
        "safebrowsing.enabled": True
    }

    options.add_experimental_option("prefs", prefs)
    # options.add_argument("--headless=new")
    options.add_argument("--no-sandbox")
    options.add_argument("--disable-dev-shm-usage")
    options.add_argument("--disable-gpu")
    options.add_argument("--disable-extensions")
    options.add_argument("--remote-debugging-port=9222")

    driver = None # Initialize driver to None
    try:
        # Start the browser
        driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()), options=options)
        driver.get("https://crowding.data.tfl.gov.uk/")

        # This combines the robustness of WebDriverWait with the explicit delay from your working script.
        WebDriverWait(driver, 15).until(
            EC.presence_of_element_located((By.TAG_NAME, "a"))
        )
        time.sleep(3) # Add a short sleep, similar to your working notebook
        print("Page loaded and initial elements are present on the single page.")

        # *** DEBUGGING STEP: Print a snippet of the page source ***
        # This will show us what HTML content Selenium is actually seeing.
        # print("\n--- Page Source Snippet (first 1000 chars) ---")
        # print(driver.page_source[:1000])
        # print("--- End Page Source Snippet ---\n")

        # *** Find all links on the current page (the only page) ***
        links = driver.find_elements(By.TAG_NAME, "a")
        journey_links = []

        # Iterate through links to find those matching the "Journeys_YYYY_YYYY.csv" pattern
        for link in links:
            try:
                href = link.get_attribute("href")
                if href:
                    decoded_href = unquote(href)
                    # The regex should be robust enough for the space before .csv
                    match = re.search(r"Journeys_(\d{4})_(\d{4})\s*\.csv", decoded_href)
                    if match:
                        year = int(match.group(1))
                        journey_links.append((year, link, match.group(1), match.group(2)))
            except StaleElementReferenceException:
                print("Warning: A link element became stale during iteration. Skipping this link.")
                continue # Skip to the next link if it becomes stale

        # Download the latest file
        if journey_links:
            # Find the link with the latest starting year
            latest = max(journey_links, key=lambda x: x[0])
            latest_element = latest[1] # This is the WebElement
            year1, year2 = latest[2], latest[3]

            # Extract the href attribute as a string *before* any further DOM interaction
            # This string will remain valid even if the WebElement becomes stale.
            download_url = latest_element.get_attribute("href")

            print(f"Identified download URL: {download_url}")
            print("Attempting to scroll the link into view (optional but good practice)...")
            try:
                # because download_url is already a safe string.
                driver.execute_script("arguments[0].scrollIntoView(true);", latest_element)
                time.sleep(1) # Give a moment for scroll to complete
            except (StaleElementReferenceException, TimeoutException):
                print("Element became stale or timed out during scroll, but URL was already captured. Proceeding with download.")
                pass # We already have the URL, so we can proceed

            # Use the extracted URL string with requests.get
            response = requests.get(download_url, verify=False)

            if response.status_code == 200:
                filename = f"transport_{year1}_{year2}.csv"
                filepath = os.path.join(target_dir, filename)
                with open(filepath, "wb") as f:
                    f.write(response.content)
                print(f"File downloaded to: {filepath}")
            else:
                print(f"Failed to download file: Status Code {response.status_code} - {response.text}")
        else:
            print("No 'Journeys' CSV links found on the page. Please verify the link pattern or page content.")

    except TimeoutException:
        print("Error: Timed out waiting for page elements to load. Check network or element locators.")
    except WebDriverException as e:
        print(f"WebDriver error occurred: {e}. This might indicate issues with browser setup or connection.")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
    finally:
        if driver:
            driver.quit() # Close the browser safely


