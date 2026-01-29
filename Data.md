## Overview

This directory contains sample datasets for the Accra Traffic Prediction & ETA Engine. The data spans 30 days (January 1-30, 2026) with measurements from 6 AM to 9 PM daily.

## Files

### traffic_data.csv
**Records:** 1,440 entries (30 days × 16 hours × 3 roads)

**Columns:**
- `road` (string): Road segment name
  - Circle Rd
  - Spintex Rd
  - Independence Ave
- `timestamp` (datetime): Date and time of measurement (YYYY-MM-DD HH:MM:SS)
- `avg_speed` (float): Average traffic speed in km/h (10-60 km/h range)

**Patterns:**
- Rush hour slowdown (7-9 AM, 5-7 PM): ~50% of base speed
- Off-peak periods (6 AM, 10-11 AM, 8-9 PM): ~110% of base speed
- Weekend traffic: ~15% faster than weekdays
- Base speeds: Circle Rd (40 km/h), Spintex Rd (45 km/h), Independence Ave (42 km/h)

### weather_data.csv
**Records:** 480 entries (30 days × 16 hours)

**Columns:**
- `timestamp` (datetime): Date and time of measurement
- `rain` (float): Rainfall in millimeters (0.0 = no rain)
- `temp` (float): Temperature in Celsius (24-35°C range)
- `humidity` (int): Relative humidity percentage (50-85%)

**Patterns:**
- Temperature peaks around noon (34-35°C)
- Morning temperatures start at ~24-25°C
- Evening temperatures cool to ~26-27°C
- Rain occurs in ~10% of records (random)
- Humidity inversely correlated with temperature

### events_data.csv
**Records:** 184 entries

**Columns:**
- `timestamp` (datetime): Event occurrence time
- `event_type` (string): Type of event
  - `none`: Normal traffic conditions
  - `concert`: Concert or large gathering
  - `sports`: Sports event
  - `festival`: Festival or celebration
  - `market_day`: Market day with increased activity
  - `accident`: Traffic incident

**Patterns:**
- Regular events logged for rush hours
- Special events occur ~20% of days at various times
- Events typically during midday (12-2 PM) or evening (6-8 PM)

## Data Generation

The data was generated using realistic traffic patterns with:
- Time-of-day variations
- Weekday/weekend differences
- Weather impacts on traffic
- Random variations to simulate real-world conditions

## Usage

Load the data in Python:
```python
import pandas as pd

traffic = pd.read_csv('data/traffic_data.csv')
weather = pd.read_csv('data/weather_data.csv')
events = pd.read_csv('data/events_data.csv')

# Use this to merge datasets
df = pd.merge(traffic, weather, on='timestamp', how='left')
df = pd.merge(df, events, on='timestamp', how='left')
df['event_type'] = df['event_type'].fillna('none')
```

## Notes

- All timestamps are aligned for proper merging
- Missing event types should be filled with 'none'
- Data is synthetic but based on realistic traffic patterns
- Suitable for machine learning model training and testing
