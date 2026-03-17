# Implementation of Random Forest Algorithm for Weather Prediction
## AIM:
To write a program to predict daily temperature , PM2.5 pollution level and Energy based on environmental sensor data using Random Forest Algorithm.

## Problem Statement and Dataset

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1.Data Loading
2.Column Cleaning
3.Time Conversion
4.Sorting
5.Missing Handling
6.Feature Extraction (Time)
7.Cyclical Encoding
8.Lag Feature Creation
9.Data Cleaning
10.Data Saving
11.Feature Selection
12.Train-Test Split
13.Model Initialization
14.Model Training
15.Prediction
16.Model Evaluation
17.Visualization (Prediction)
18.Feature Importance
19.Latest Data Extraction
20.Future Predict

## Program:
```
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
import joblib

# Load dataset
df = pd.read_csv("weather-station-eee-block_2024_07_13.csv")
df.columns = df.columns.str.strip()
df['time'] = pd.to_datetime(df['time'], errors='coerce')

print("Original rows:", len(df))

# Only drop if target missing
df = df.dropna(subset=['tem', 'pm2_5'])

# Fill feature columns instead of dropping
df['hum'] = df['hum'].fillna(df['hum'].mean())
df['pressure'] = df['pressure'].fillna(df['pressure'].mean())
df['wind_speed'] = df['wind_speed'].fillna(df['wind_speed'].mean())
df['co2'] = df['co2'].fillna(df['co2'].mean())

# Sort by time
df = df.sort_values('time')

# Create lag features
df['Temp_Lag1'] = df['tem'].shift(1)
df['PM_Lag1'] = df['pm2_5'].shift(1)

# Only remove first row created by shift
df = df.iloc[1:]

print("Rows after preprocessing:", len(df))

# Features
X = df[['hum', 'pressure', 'wind_speed', 'co2',
        'Temp_Lag1', 'PM_Lag1']]

y_temp = df['tem']
y_pm = df['pm2_5']

print("Training samples:", len(X))

# Train models
model_temp = RandomForestRegressor(n_estimators=300, random_state=42)
model_pm = RandomForestRegressor(n_estimators=300, random_state=42)

model_temp.fit(X, y_temp)
model_pm.fit(X, y_pm)

# Save models
joblib.dump(model_temp, "temperature_model.pkl")
joblib.dump(model_pm, "pm25_model.pkl")

print("✅ Models trained and saved successfully!")
```
Program to implement the Random Forest Algorithm to predict daily temperature , PM2.5 pollution level and Energy based on environmental sensor data.
Developed by: mohamed athif
RegisterNumber:  25011374,212225040239
*/


## Output:
<img width="1088" height="124" alt="Screenshot 2026-03-17 093401" src="https://github.com/user-attachments/assets/7fa33f07-86bf-49c2-ac22-93b1b1269195" />

## Result:
The Random Forest model was successfully implemented to predict temperature and PM2.5 pollution levels from environmental sensor data. The trained models were saved successfully.
